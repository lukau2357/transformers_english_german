import torch
import youtokentome as yttm
import time

from torch.nn.init import xavier_uniform_
from torch.nn import functional as F
from prettytable import PrettyTable
from typing import List
from dataset import preprocess_line

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_time_steps, device, lookahead_mask_required = False, p = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.max_time_steps = max_time_steps
        self.d_k = d_model // num_heads
        self.device = device
        self.lookahead_mask_required = lookahead_mask_required
        self.p = p

        if lookahead_mask_required:
            self.register_buffer("mask", torch.tril(torch.ones((max_time_steps, max_time_steps), device = device)).view(1, 1, max_time_steps, max_time_steps))
        
        # Q, K, V mappings in all attention heads are initialized jointly with xavier uniform where fan_in = d_model and fan_out = d_model
        # In reality, dimensionality of Q, K, V per attention head is fan_in = d_model, fan_out = d_k. Perhaps consider independent initalization and then
        # concatenating obtained tensors.
        self.WQ = torch.nn.Parameter(xavier_uniform_(torch.zeros(d_model, d_model, device = device)))
        self.WV = torch.nn.Parameter(xavier_uniform_(torch.zeros(d_model, d_model, device = device)))
        self.WK = torch.nn.Parameter(xavier_uniform_(torch.zeros(d_model, d_model, device = device)))              
        self.WO = torch.nn.Parameter(xavier_uniform_(torch.zeros(d_model, d_model, device = device)))

        self.attention_droput = torch.nn.Dropout(p = p)
        self.residual_dropout = torch.nn.Dropout(p = p)

    def forward(self, X1, X2, padding_mask):
        """
        X1 and X2 are of shape (batch_size, time_steps, d_model). If we are working with self attention, queries, keys and values
        will come from the output of previous layer, in which case we have X1 = X2. If we are working with cross attention, queries 
        come from previous layer output, and keys and values come from output of parallel layer, in which case X1 represents
        output of previous layer and X2 represents output of parallel layer.
        """
        batch_size = X1.shape[0]
        t1 = X1.shape[1]
        t2 = X2.shape[1]

        assert t1 <= self.max_time_steps, f"MHA assertion error: t1: {t1} max_time_steps: {self.max_time_steps}"

        Q = X1.matmul(self.WQ).view((batch_size, t1, self.num_heads, self.d_k)).transpose(1, 2)
        K = X2.matmul(self.WK).view((batch_size, t2, self.num_heads, self.d_k)).transpose(1, 2)
        V = X2.matmul(self.WV).view((batch_size, t2, self.num_heads, self.d_k)).transpose(1, 2)

        # (batch_size, num_heads, t, t)
        softmax_in = Q.matmul(K.transpose(-1, -2) / (self.d_k) ** 0.5)
        if self.lookahead_mask_required:
            softmax_in = softmax_in.masked_fill(self.mask[:, :, :t1, :t1] == 0, float("-inf")).to(self.device)

        # Always mask out value similarities, that is why we apply view (batch_size, 1, 1, t2). 
        # We always prevent query similarity computation with padding token value. Even though 
        # when query represents a padding token and it's similarities with other non-padding token values
        # are computed, Linear layer processes data independently so they won't affect other computations.
        # Lastly, probabilities induced by padding tokens are ignored in the loss itself (ignore_index = 0).
        softmax_in = softmax_in.masked_fill(padding_mask, float("-inf"))
        w = F.softmax(softmax_in, dim = -1)
        w = self.attention_droput(w)
        # (batch_size, t, d_model)
        # Alternatively, one more dropout on final computation
        return self.residual_dropout(w.matmul(V).transpose(1, 2).contiguous().view((batch_size, t1, self.d_model)).matmul(self.WO))

class EncoderBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_time_steps, hidden_factor, device, p = 0.1):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_time_steps = max_time_steps
        self.device = device
        self.hidden_factor = hidden_factor # Dimensionality of hidden layer in MLP is hidden_factor * d_model
        self.p = p

        """
        Original Transformer paper uses PostLayerNorm layer norm positioning - layer norm is perform after residual connections are 
        summed. PreLayerNorm first applied LayerNorm to output of last layer and then aggregates with residual connection. A study has 
        been performed on PreLayerNorm scheme, and it shows that it leads to more stable gradients (and hence training) in initial stages.
        https://arxiv.org/abs/2002.04745. This implementation uses PreLayerNorm assignment of LayerNorm.
        """

        self.ln1 = torch.nn.LayerNorm(d_model, device = device)
        self.mha = MultiHeadAttention(d_model, num_heads, max_time_steps, device, lookahead_mask_required = False, p =p)
        self.ln2 = torch.nn.LayerNorm(d_model, device = device)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden_factor * d_model, device = device),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_factor * d_model, d_model, device = device))
        self.mlp_dropout = torch.nn.Dropout(p = p)

    def forward(self, X, padding_mask):
        t = self.ln1(X)
        X = X + self.mha(t, t, padding_mask)
        X = X + self.mlp_dropout(self.mlp(self.ln2(X)))
        return X

class DecoderBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_time_steps, hidden_factor, device, p = 0.1):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_time_steps = max_time_steps
        self.device = device
        self.hidden_factor = hidden_factor # Dimensionality of hidden layer in MLP is hidden_factor * d_model
        self.p = p

        self.ln1 = torch.nn.LayerNorm(d_model, device = device)
        self.mha1 = MultiHeadAttention(d_model, num_heads, max_time_steps, device, lookahead_mask_required = True, p = p)
        self.ln2_q = torch.nn.LayerNorm(d_model, device = device)
        self.ln2_kv = torch.nn.LayerNorm(d_model, device = device)
        self.mha2 = MultiHeadAttention(d_model, num_heads, max_time_steps, device, lookahead_mask_required = False, p = p)
        self.ln_mlp = torch.nn.LayerNorm(d_model, device = device)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden_factor * d_model, device = device),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_factor * d_model, d_model, device = device))
        self.mlp_dropout = torch.nn.Dropout(p = p)

    def forward(self, X, enc_out, padding_mask_dec, padding_mask_enc):
        tq = self.ln1(X)
        tq = X + self.mha1(tq, tq, padding_mask_dec)
        # assume enc_out is layer normalized at the end of encoder
        tq = tq + self.mha2(self.ln2_q(tq), enc_out, padding_mask_enc)
        tq = tq + self.mlp_dropout(self.mlp(self.ln_mlp(tq)))
        return tq

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size_source, 
                 vocab_size_target, 
                 d_model, 
                 num_layers, 
                 num_heads, 
                 max_time_steps_encoder,
                 max_time_steps_decoder,
                 hidden_factor, 
                 device,
                 padding_idx, 
                 p = 0.1):
        
        super(Transformer, self).__init__()
        self.vocab_size_source = vocab_size_source
        self.vocab_size_target = vocab_size_target
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_time_steps_encoder = max_time_steps_encoder
        self.max_time_steps_decoder = max_time_steps_decoder
        self.hidden_factor = hidden_factor
        self.device = device
        self.padding_idx = padding_idx
        self.p = p

        self.encoder_dropout_initial = torch.nn.Dropout(p = p)
        self.decoder_dropout_initial = torch.nn.Dropout(p = p)
        self.encoder_we = torch.nn.Embedding(vocab_size_source, d_model, device = device, padding_idx = padding_idx)
        self.encoder_pe = torch.nn.Embedding(max_time_steps_encoder, d_model, device = device)
        self.decoder_we = torch.nn.Embedding(vocab_size_target, d_model, device = device, padding_idx = padding_idx)
        self.decoder_pe = torch.nn.Embedding(max_time_steps_decoder, d_model, device = device)

        self.encoder = torch.nn.ModuleList([EncoderBlock(d_model, num_heads, max_time_steps_encoder, hidden_factor, device, p = p) for _ in range(num_layers)])
        self.decoder = torch.nn.ModuleList([DecoderBlock(d_model, num_heads, max_time_steps_decoder, hidden_factor, device, p = p) for _ in range(num_layers)])

        self.last_ln = torch.nn.LayerNorm(d_model, device = device)
        self.encoder_ln = torch.nn.LayerNorm(d_model, device = device)
        self.last_linear = torch.nn.Linear(d_model, vocab_size_target, bias = False, device = device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Xavier normal initialization for Linear layers
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # Token/positional embeddings initlaizaiton
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0, std = 0.02)

    def count_parameters(self, sort_by_capacity = True):
        # Using pretty table to represent trainable parameters of the model.
        table = PrettyTable(["Layer", "Number of parameters"])
        total_params = 0
        params_list = []

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue

            params = parameter.numel()
            params_list.append((params, name))
            total_params += params

        if sort_by_capacity:
            params_list = sorted(params_list, reverse = True)
        
        for key, value in params_list:
            table.add_row([value, key])

        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def forward(self, X_source, X_target):
        """
        X_source -> (batch_size, t1)
        X_target -> (batch_size, t2)
        """
        t1 = X_source.shape[1]
        t2 = X_target.shape[1]

        # (batch_size, 1, 1, t1) -> broadcast by rows and all attention heads. (batch_size, 1, 1, t2) for decoder padding mask.
        # For each batch, row of size t1 gets repeated num_heads * t1 times when broadcast in encoder, and num_heads * t2 when broadcast in decoder.
        padding_mask_encoder = (X_source == self.padding_idx).unsqueeze(1).unsqueeze(2)
        padding_mask_decoder = (X_target == self.padding_idx).unsqueeze(1).unsqueeze(2)

        pos_1 = torch.arange(0, t1, dtype = torch.int32, device = self.device)
        pos_2 = torch.arange(0, t2, dtype = torch.int32, device = self.device)

        X_source = self.encoder_dropout_initial(self.encoder_we(X_source) + self.encoder_pe(pos_1))
        for i in range(self.num_layers):
            X_source = self.encoder[i].forward(X_source, padding_mask_encoder)

        X_source = self.encoder_ln(X_source)
        # Layer norm of encoder output
        X_target = self.decoder_dropout_initial(self.decoder_we(X_target) + self.decoder_pe(pos_2))

        for i in range(self.num_layers):
            X_target = self.decoder[i].forward(X_target, X_source, padding_mask_decoder, padding_mask_encoder)

        return self.last_linear(self.last_ln(X_target))
    
    def forward_encoder(self, X_source : torch.Tensor):
        # Used during inferece, no dropout will be used, even though model should be put into eval mode.
        t1 = X_source.shape[-1]
        padding_mask_encoder = (X_source == self.padding_idx).unsqueeze(1).unsqueeze(2)
        pos_1 = torch.arange(0, t1, dtype = torch.int32, device = self.device)
        X_source = self.encoder_dropout_initial(self.encoder_we(X_source) + self.encoder_pe(pos_1))

        for i in range(self.num_layers):
            X_source = self.encoder[i].forward(X_source, padding_mask_encoder)

        X_source = self.encoder_ln(X_source)
        return X_source
    
    def forward_decoder(self, X_target : torch.Tensor, context : torch.Tensor, encoder_mask : torch.Tensor):
        t2 = X_target.shape[-1]
        padding_mask_decoder = (X_target == self.padding_idx).unsqueeze(1).unsqueeze(2)
        pos_2 = torch.arange(0, t2, dtype = torch.int32, device = self.device)

        X_target = self.decoder_dropout_initial(self.decoder_we(X_target) + self.decoder_pe(pos_2))

        for i in range(self.num_layers):
            X_target = self.decoder[i].forward(X_target, context, padding_mask_decoder, encoder_mask)

        return self.last_linear(self.last_ln(X_target))

    @torch.no_grad()
    def inference_greedy(self, source_sentence : str, source_tokenizer : yttm.BPE, target_tokenizer : yttm.BPE, source_tokens : torch.Tensor = None) -> str:
        """
        Translates the given source sentence using greedy search.
        """
        encoder_tokens = torch.tensor([source_tokenizer.encode([source_sentence])[0]]).to(self.device) if source_tokens is None else source_tokens.to(self.device)
        # Input truncation if necessary
        encoder_tokens = encoder_tokens[:, -self.max_time_steps_encoder:]
        encoder_mask = (encoder_tokens == self.padding_idx).unsqueeze(1).unsqueeze(2)
        context = self.forward_encoder(encoder_tokens)
        decoder_tokens = torch.tensor([[target_tokenizer.subword_to_id("<BOS>")]]).to(self.device)

        while True:
            decoder_out = torch.nn.functional.log_softmax(self.forward_decoder(decoder_tokens, context, encoder_mask), dim = -1)
            next_token = torch.argmax(decoder_out, dim = -1)[:, -1]
            # End of sequence predicted as next token, translation is finished.
            if next_token == target_tokenizer.subword_to_id("<EOS>"):
                break

            decoder_tokens = torch.cat((decoder_tokens, next_token.unsqueeze(0)), dim = -1)
            # Length truncation if necessary
            if decoder_tokens.shape[1] > self.max_time_steps_decoder:
                decoder_tokens = decoder_tokens[:, 1:]
            
        # Decode target sentence, while disregarding beginning of sequence token
        out_sentence = target_tokenizer.decode(decoder_tokens.squeeze(0).tolist(), ignore_ids = [target_tokenizer.subword_to_id("<BOS>")])[0]
        return out_sentence
        
    @torch.no_grad()
    def inference_beam(self, source_sentence : str, source_tokenizer : yttm.BPE, target_tokenizer : yttm.BPE, 
                       beam_size : int = None, source_tokens : torch.Tensor = None, alpha = 0.75) -> str:
        """
        Beam search sampling. Beam search in domain of NLP differs from beam search in classical artifical intelligence:
            - First, there is a memory constraint, size of open set is at most beam_size in NLP while in classical AI it is not bounded.
            - In NLP all beams are terminated while in classical AI the shortest path is returned 
              (in terms of path length not considering the const function of the problem). This is because beam search in NLP has a tendency to pick shorter seuqneces
              which is not desireable.
            - Using parallelism, in NLP beam search all paths are expanded at once (resembelling BFS) thanks to data batching and parallel computing platforms like cuda. 
              This can of course be done for AI beam search, however vanilla formulation of beam search in classical AI does not include parallelism.
            - Since objective of NLP beam search is to find the sequence which maximizes conditional probability (or equivalently, sequence which maximizes log-likelihood,
              better for numerical stability), naturally shorter sequences will be preferable over longer ones. To circumvent this, one possibility is to scale
              log-likelihood by (sequence_length)^(-alpha) where alhpa is in [0, 1].
            - Lastly, since completeness of beam search is not guaranteed, we extend the beams at most self.max_time_steps_decoder times. If non of the sequences
              generated up to self.max_time_steps_decoder terminated, we just return the most probable sequence up to this point.
        """

        if beam_size is None:
            print("WARNING: beam_size parameter not given, using default beam_size of 10.")

        encoder_tokens = torch.tensor([source_tokenizer.encode([source_sentence])[0]]).to(self.device) if source_tokens is None else source_tokens.to(self.device)
        # Input truncation if necessary
        encoder_tokens = encoder_tokens[:, -self.max_time_steps_encoder:]
        encoder_mask = (encoder_tokens == self.padding_idx).unsqueeze(1).unsqueeze(2)
        context = self.forward_encoder(encoder_tokens)

        beams = torch.tensor([[target_tokenizer.subword_to_id("<BOS>")]]).to(self.device)
        prev_probs = torch.zeros((1, 1), device = self.device)
        V = target_tokenizer.vocab_size()
        candidates = []

        # <BOS> Token was already added
        for i in range(self.max_time_steps_decoder - 1):
            eos_mask = beams[:, -1] == target_tokenizer.subword_to_id("<EOS>")
            if sum(eos_mask) > 0:
                id = int(torch.argmax(prev_probs.masked_fill(~eos_mask, float("-inf"))).item())
                infered_sequence = beams[id]
                infered_sequence_prob = prev_probs[id]
                candidates.append((infered_sequence_prob, infered_sequence))
                # Remove terminated sequences from further consideration.
                beams = beams[~eos_mask]
                prev_probs = prev_probs[~eos_mask]

                # Every beam completed, terminate search.
                if beams.shape[0] == 0:
                    break

            new_beams_probs = torch.nn.functional.log_softmax(self.forward_decoder(beams, context.expand(beams.shape[0], -1, -1), encoder_mask)[:, -1], dim = -1) + prev_probs
            # Need to flatten new_beams_keys before doing topk to find global maxima, not maxima over beams. 
            _, indices = torch.topk(new_beams_probs.flatten(), beam_size, dim = -1)
            # Restore original indices after flatten and topk
            i = torch.floor(indices / V).to(torch.int32)
            j = (indices % V)
            beams = torch.cat((beams[i, :], j.view((beam_size, 1))), 1)
            # Length truncation is not necessary since the loop is run at most self.max_time_steps_decoder - 1 times, with <BOS> already being added.
            prev_probs = new_beams_probs[i, j].unsqueeze(1)
        
        # If none of the beams terminated, return most likely sequence up to this point.
        if len(candidates) == 0:
            print("WARNING: None of the beams generated terminating sequence, outputing most likely sequence up to current point.")
            candidates = [(prev_probs[i], beams[i]) for i in range(beams.shape[0])]
        # Normalize log likelihood by sequence_length ** alhpa since non-normalized beam search has tendency to choose shorter sequences.        
        infered_sequence = max(candidates, key = lambda x : x[0] / (x[1].shape[0] ** alpha))[1]
        out_sentence = target_tokenizer.decode(infered_sequence.tolist(), ignore_ids = [target_tokenizer.subword_to_id("<BOS>"), target_tokenizer.subword_to_id("<EOS>")])[0]
        return out_sentence

    @torch.no_grad()
    def inference_topk(self, source_sentence : str, source_tokenizer : yttm.BPE, target_tokenizer : yttm.BPE, k : int = None, source_tokens : torch.Tensor = None, temperature : int = 1):
        """
        Translate the given source sentence using top k sampling.
        """
        if k is None:
            print("WARNING: k parameter not given, using default k of 10.")
            k = 10

        encoder_tokens = torch.tensor([source_tokenizer.encode([source_sentence])[0]]).to(self.device) if source_tokens is None else source_tokens.to(self.device)
        # Input truncation if necessary
        encoder_tokens = encoder_tokens[:, -self.max_time_steps_encoder:]
        encoder_mask = (encoder_tokens == self.padding_idx).unsqueeze(1).unsqueeze(2)
        context = self.forward_encoder(encoder_tokens)
        decoder_tokens = torch.tensor([[target_tokenizer.subword_to_id("<BOS>")]]).to(self.device)
        
        while True:
            decoder_out = torch.nn.functional.softmax(self.forward_decoder(decoder_tokens, context, encoder_mask)[:, -1] / temperature, dim = -1)
            tk = torch.topk(decoder_out, k, dim = -1)
            i = torch.multinomial(tk.values.squeeze(0), 1)
            next_token = tk.indices[:, i]
            
            if next_token == target_tokenizer.subword_to_id("<EOS>"):
                break

            decoder_tokens = torch.cat((decoder_tokens, next_token), dim = -1)
            # Length truncation if necessary
            if decoder_tokens.shape[1] > self.max_time_steps_decoder:
                decoder_tokens = decoder_tokens[:, 1:]

        out_sentence = target_tokenizer.decode(decoder_tokens.squeeze(0).tolist(), ignore_ids = [target_tokenizer.subword_to_id("<BOS>")])[0]
        return out_sentence
    
    def live_inference(self, source_tokenizer : yttm.BPE, target_tokenizer : yttm.BPE, inference_methods : List[str], k : int = None, beam_size : int = None):
        print(f"Performing live inference with methods: {inference_methods}")
        if "topk" in inference_methods or "beam" in inference_methods:
            print("Hyperparameters")
            if "topk" in inference_methods:
                print(f"TopK: k = {k}")
            
            if "beam" in inference_methods:
                print(f"Beam search: beam_size = {beam_size}")
        
        while True:
            source_sentence = input("Enter an English sentence:")
            source_sentence = preprocess_line(source_sentence)
            print(f"Source sentence after preprocessing: {source_sentence}\n")
            print("Model generated translation/s:")

            for method in inference_methods:
                start = time.time()
                if method == "greedy":
                    print(f"Greedy inference:\n{self.inference_greedy(source_sentence, source_tokenizer, target_tokenizer)}")
                
                elif method == "topk":
                    print(f"TopK inference:\n{self.inference_topk(source_sentence, source_tokenizer, target_tokenizer, k = k)}")
                
                elif method == "beam":
                    print(f"Beam inference:\n{self.inference_beam(source_sentence, source_tokenizer, target_tokenizer, beam_size = beam_size)}")
                
                end = time.time()
                print(f"Inference time: {end - start} seconds\n")

            print("-" * 200)
                    
    @classmethod
    def load(cls, path, device) :
        d = torch.load(path, map_location = device)

        vocab_size_source = d["vocab_size_source"]
        vocab_size_target = d["vocab_size_target"]
        d_model = d["d_model"]
        num_layers = d["num_layers"]
        num_heads = d["num_heads"]
        max_time_steps_encoder = d["max_time_steps_encoder"]
        max_time_steps_decoder = d["max_time_steps_decoder"]
        hidden_factor = d["hidden_factor"]
        padding_idx = d["padding_idx"]
        state_dict = d["state_dict"]
        p = d["p_dropout"]

        res = cls(vocab_size_source, vocab_size_target, d_model, num_layers, num_heads, max_time_steps_encoder, max_time_steps_decoder, hidden_factor, device, padding_idx, p = p)
        res.load_state_dict(state_dict)
        return res
    
    @classmethod
    def from_dict(cls, d, device):

        vocab_size_source = d["vocab_size_source"]
        vocab_size_target = d["vocab_size_target"]
        d_model = d["d_model"]
        num_layers = d["num_layers"]
        num_heads = d["num_heads"]
        max_time_steps_encoder = d["max_time_steps_encoder"]
        max_time_steps_decoder = d["max_time_steps_decoder"]
        hidden_factor = d["hidden_factor"]
        padding_idx = d["padding_idx"]
        state_dict = d["state_dict"]
        p = d["p_dropout"]

        res = cls(vocab_size_source, vocab_size_target, d_model, num_layers, num_heads, max_time_steps_encoder, max_time_steps_decoder, hidden_factor, device, padding_idx, p = p)
        res.load_state_dict(state_dict)
        return res
    
    def save(self, path : str, dict_export : bool = False):
        d = {
            "vocab_size_source": self.vocab_size_source,
            "vocab_size_target": self.vocab_size_target,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "max_time_steps_encoder": self.max_time_steps_encoder,
            "max_time_steps_decoder": self.max_time_steps_decoder,
            "hidden_factor": self.hidden_factor,
            "padding_idx": self.padding_idx,
            "state_dict": self.state_dict(),
            "p_dropout": self.p
        }

        if dict_export:
            return d
        
        torch.save(d, path)