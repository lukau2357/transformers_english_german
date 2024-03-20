import os
import regex as re
import numpy as np 
import torch
import youtokentome as yttm
import tqdm
import matplotlib.pyplot as plt
import linecache

from torch.utils.data import Dataset

plt.style.use("ggplot")

REPEATED_WS = re.compile("([ \t\n\r]){2,}")
PARENTHESES = re.compile("([\(\)\{\}\[\]\|])")
# For english try keeping apostrohpes together with given suffixes, while ignoring casing. This is applied to the target language as well.
QUOTES = re.compile('[\"`„”“]|(\'(?!(?i:[sdmt]|ll|ve|re)))')
# WORD_INTERPUNCTION = re.compile('(\p{L}*)([^\s\p{L}\p{N}]+)(\p{L}*)')

def preprocess_line(line : str):
    line = line.lower()
    line = re.sub(PARENTHESES, "", line)
    line = re.sub(QUOTES, "", line)
    line = re.sub(REPEATED_WS, "\\1", line)
    # line = re.sub(WORD_INTERPUNCTION, "\\1 \\2 \\3", line)
    line = line.strip()
    return line

def preprocessing_write(source_language_filename : str, target_language_filename : str, source_language_filename_preprocessed : str, target_language_filename_preporcessed : str, length_heuristic : bool = True):
    print("Preprocessing source sentences.")

    with open(source_language_filename, "r", encoding = "utf-8") as f:
        lines = f.readlines()
        source_sentences = []

        for line in tqdm.tqdm(lines):
            if len(line) <= 1:
                print(f"Enteref with {line}")
                continue
            
            line = preprocess_line(line)
            source_sentences.append(line)
    
    print("Preprocessing target sentences.")
    with open(target_language_filename, "r", encoding = "utf-8") as f:
        lines = f.readlines()
        target_sentences = []

        for line in tqdm.tqdm(lines):
            if len(line) <= 1:
                continue

            line = preprocess_line(line)
            target_sentences.append(line)
            

    print(f"Source sentences: {len(source_sentences)} Target sentences: {len(target_sentences)}")
    assert len(source_sentences) == len(target_sentences), f"unequal number of sentences accross datasets! {len(source_sentences)} {len(target_sentences)}"

    sp_counter = 0
    problematic_ids = {}

    """
    If provided dataset can have missmatching sentences, it's possible to apply the following heuristic:
        Let m be character length of source sentence and let n be character length of target sentence. If m > 2n or n > 2m
        we deem the given sentence pair invalid and do now include it in the final dataset.
    """

    if length_heuristic:
        print("Computing problematic sentences")
        for i, (e, g) in tqdm.tqdm(enumerate(zip(source_sentences, target_sentences))):
            if len(g) > 2 * len(e) or len(e) > 2 * len(g):
                # print("Problematic sentence pair")
                # print(e)
                # print()
                # print(g)
                # print()
                sp_counter += 1
                problematic_ids[i] = True

        print(f"Number of problematic sentence pairs: {sp_counter}. Percentage: {(sp_counter / len(source_sentences) * 100)}")

    with open(source_language_filename_preprocessed, "w+", encoding = "utf-8") as f:
        for i, s in enumerate(source_sentences):
            if i not in problematic_ids:
                f.write(s + "\n")
        
        # Remove trailing newline character
        f.truncate(f.tell() - len(os.linesep))

    with open(target_language_filename_preporcessed, "w+", encoding = "utf-8") as f:
        for i, s in enumerate(target_sentences):
            if i not in problematic_ids:
                f.write(s + "\n")
        
        f.truncate(f.tell() - len(os.linesep))

def truncate_length_and_tokenize(source_filename, source_tokenizer, length_source, target_filename, target_tokenizer, length_target):
    d = {}
    sentences_source, sentences_target = [], []

    print("Reading source sentences for filtering")
    with open(source_filename, "r", encoding = "utf-8") as f:
        for i, line in tqdm.tqdm(enumerate(f)):
            ids = source_tokenizer.encode([line.strip()])[0]
            l = len(ids)
            sentences_source.append(" ".join([str(x) for x in ids])) 

            if l <= length_source:
                d[i] = [True, False]
            
            else:
                d[i] = [False, False]
    
    print("Reading target sentences for filtering")
    with open(target_filename, "r", encoding = "utf-8") as f:
        for i, line in tqdm.tqdm(enumerate(f)):            
            ids = target_tokenizer.encode([line.strip()], bos = True, eos = True)[0]
            l = len(ids)
            sentences_target.append(" ".join([str(x) for x in ids]))

            if l <= length_target:
                d[i][1] = True
    
    print("Performing joint filtering over source and target sentences")

    with open(source_filename + "_tokenized", "w+") as sf, open(target_filename + "_tokenized", "w+") as df:
        for i, (source_ids, target_ids) in tqdm.tqdm(enumerate(zip(sentences_source, sentences_target))):
            if d[i][0] and d[i][1]:
                sf.write(source_ids + "\n")
                df.write(target_ids + "\n")
    
        df.truncate(df.tell() - len(os.linesep))
        sf.truncate(sf.tell() - len(os.linesep))

def tokenize(source_filename : str, target_filename : str, source_tokenizer : yttm.BPE, target_tokenizer : yttm.BPE):
    print("Pre-tokenizing source sentences.")
    with open(source_filename, "r", encoding = "utf-8") as sf, open(source_filename + "_tokenized", "w+") as tf:
        for line in tqdm.tqdm(sf):
            tokens = source_tokenizer.encode([line.strip()])[0]     
            tokens = " ".join([str(x) for x in tokens])
            tf.write(tokens + "\n")
        
        tf.truncate(tf.tell() - len(os.linesep))

    print("Pre-tokenizing target sentences.")
    with open(target_filename, "r", encoding = "utf-8") as sf, open(target_filename + "_tokenized", "w+") as tf:
        for line in tqdm.tqdm(sf):
            tokens = target_tokenizer.encode([line.strip()], bos = True, eos = True)[0]
            tokens = " ".join([str(x) for x in tokens])
            tf.write(tokens + "\n")
        
        tf.truncate(tf.tell() - len(os.linesep))

def max_sequence_lengths(source_filename, target_filename):
    max_source, max_target = -1, -1
    with open(source_filename, "r") as f:
        for line in f:
            t = line.strip().split()
            max_source = max(max_source, len(t))
    
    with open(target_filename, "r") as f:
        for line in f:
            t = line.strip().split()
            max_target = max(max_target, len(t))
    
    return max_source, max_target

def train_bpe_yttm(source_language_filename, source_language_tokenizer_filename, target_language_filename, target_language_tokenizer_filename, vocab_size):
    # youtokentome.BPE.train(data, model, vocab_size, coverage, n_threads = -1, pad_id = 0, unk_id = 1, bos_id = 2, eos_id = 3)
    yttm.BPE.train(source_language_filename, source_language_tokenizer_filename, vocab_size)
    yttm.BPE.train(target_language_filename, target_language_tokenizer_filename, vocab_size)

def dataset_statistics(english_bpe, source_language_filename_preprocessed, german_bpe, target_language_filename_preprocessed):
    max_english_len, max_german_len, max_english_id, max_german_id = -1, -1, -1, -1
    max_english_line, max_german_line = "", ""

    print("Computing longest source sentence (in token length)")
    with open(source_language_filename_preprocessed, "r", encoding = "utf-8") as f:
        for i, line in tqdm.tqdm(enumerate(f)):
            l = len(english_bpe.encode([line])[0])
            if l > max_english_len:
                max_english_len = l
                max_english_line = line
                max_english_id = i
    
    print("Computing longest target sentence (in token length)")
    with open(target_language_filename_preprocessed, "r", encoding = "utf-8") as f:
        for i, line in tqdm.tqdm(enumerate(f)):
            l = len(german_bpe.encode([line], bos = True, eos = True)[0])
            if l > max_german_len:
                max_german_len = l
                max_german_line = line
                max_german_id = i

    print(f"Maximum Source sequence length: {max_english_len}")
    print(f"Maximum Target sequence length: {max_german_len}")
    print(f"Longest Source sentence (in token length): {max_english_line}")
    print(f"Longest Target sentence (in token length): {max_german_line}")
    print(f"Maximum Source sentence ID: {max_english_id}")
    print(f"Maximum Target sentence ID: {max_german_id}")

def split_dataset(source_filename, target_filename, train = 0.8, portion = None):
    source_train = source_filename + "_train"
    source_val = source_filename + "_val"
    target_train = target_filename + "_train"
    target_val = target_filename + "_val"
    mul = 1 if portion is None else portion

    print("Splitting source language into train-validation.")
    with open(source_filename, "r") as f, open(source_train, "w+") as tf, open(source_val, "w+") as vf:
        l = int(mul * len(f.readlines()))
        gap = int(l * train)
        f.seek(0)

        for i, line in tqdm.tqdm(enumerate(f)):
            if i > l:
                continue

            if i <= gap:
                tf.write(line)
            
            else:
                vf.write(line)

    print("Splitting target language into train-validation.")
    with open(target_filename, "r") as f, open(target_train, "w+") as tf, open(target_val, "w+") as vf:
        l = int(mul * len(f.readlines()))
        gap = int(l * train)
        f.seek(0)

        for i, line in tqdm.tqdm(enumerate(f)):
            if i > l:
                continue
            
            if i <= gap:
                tf.write(line)
            
            else:
                vf.write(line)
        
class TranslationDataset(Dataset):
    def __init__(self, source_filename, target_filename):
        super(Dataset, self).__init__()
        self.source_filename = source_filename
        self.target_filename = target_filename
        self.N = 0

        with open(source_filename, "r") as f:
            for _ in f:
                self.N += 1

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Lazyloading with linecache, avoid reading entire dataset into memory.
        source_ids = linecache.getline(self.source_filename, idx + 1).strip().split()
        target_ids = linecache.getline(self.target_filename, idx + 1).strip().split()
        return torch.tensor([int(x) for x in source_ids], dtype = torch.int32), torch.tensor([int(x) for x in target_ids], dtype = torch.int32)

"""
Creating batch pipeline:
    - Sample b pairs of English to German sentences.
    - Tokenize sentences with correspodning tokenizers.
    - Let e be the list of tokenized English sentences, perform padding on these sentences to obtain tensor e.
    - Do the same for g.
    - Return e, g[:-1], g[1:]. German sentences are shifted to right.
"""

def translation_dataset_collate(batch):
    """
    Collation of samples from TranslationDataset into a torch Tensor.
    Returns padded source language sentences, padded target language sentences, padded target language sentences shifted by 1 and 
    original lengths of padded target language sentences. Lengths are required to disregard padding tokens in loss computation.
    """
    b = len(batch)
    X_e = [item[0] for item in batch]
    X_g = [item[1][:-1] for item in batch] + [item[1][1:] for item in batch]

    X_e = torch.nn.utils.rnn.pad_sequence(X_e, batch_first = True, padding_value = 0)
    X_g = torch.nn.utils.rnn.pad_sequence(X_g, batch_first = True, padding_value = 0)
    return X_e.int(), X_g[:b].int(), X_g[b:].long()

def decode_batch(batch : torch.Tensor, tokenizer : yttm.BPE, ignore_ids : list[int] = []):
    batch = batch.tolist()
    return tokenizer.decode(batch, ignore_ids = ignore_ids)

def length_statistics(filename, tokenizer, q = 1, bos = False, eos = False, plot = False, verbose = False, plot_title_label = "", dataset_label = ""):
    """
    Compute sequence length statistics for the given portion of the dataset and save corresponding distribution as a figure.
    """
    stats = []

    print(f"Processing {filename} for length distribution.")
    with open(filename, "r", encoding = "utf-8") as f:
        for line in tqdm.tqdm(f):
            stats.append(len(tokenizer.encode([line], bos = bos, eos = eos)[0]))

    p = int(np.quantile(stats, q))
    if q < 1:
        print(f"{q}-th quantile: {p}")

    if verbose:
        print(f"Printing sentences with token length greater than {q}-th quantile")
        with open(filename, "r", encoding = "utf-8") as f:
            for line in tqdm.tqdm(f):
                l = len(tokenizer.encode([line], bos = bos, eos = eos)[0])
                if l > p:
                    print(line, l)
                    print()
    
    if plot:
        fig, ax = plt.subplots()
        ax.hist(stats, edgecolor = "black", linewidth = 1.2, log = True)
        if q < 1:
            ax.axvline(p, color = "black", label = f"{q}-th quantile = {p}")
        ax.set_title(f"Tokenized sentence length statistics for {plot_title_label}")
        ax.set_xlabel("Bins")
        ax.set_ylabel("Distribution (Log scale)")
        ax.legend()
        if not os.path.exists("figs"):
            os.mkdir("figs")

        plt.savefig(os.path.join("figs", f"{plot_title_label.lower()}_{dataset_label}_sequence_length_distribution.png"))
    
    return p

def data_pipeline(source_filename : str, target_filename : str, source_tokenizer_filename : str, target_tokenizer_filename : str, vocab_size_max : int,
                  train = False, length_truncation = True, 
                  dataset_label : str= "", 
                  source_plot_title_label : str= "",
                  target_plot_title_label : str= "",
                  quantile : float = 0.999,
                  require_splitting : bool = False,
                  train_threshold : float = 0.8,
                  length_heuristic : bool = True,
                  data_portion : float = None):

    source_filename_preprocessed = source_filename + "_v2"
    target_filename_preprocessed = target_filename + "_v2"
    preprocessing_write(source_filename, target_filename, source_filename_preprocessed, target_filename_preprocessed, length_heuristic = length_heuristic)

    if train:
        train_bpe_yttm(source_filename_preprocessed, source_tokenizer_filename, target_filename_preprocessed, target_tokenizer_filename, vocab_size_max)

    source_tokenizer = yttm.BPE(source_tokenizer_filename)
    target_tokenizer = yttm.BPE(target_tokenizer_filename)

    if train:
        print("Dataset statistics before length filtering")
        dataset_statistics(source_tokenizer, source_filename_preprocessed, target_tokenizer, target_filename_preprocessed)
        q = quantile
        q_english = length_statistics(source_filename_preprocessed, source_tokenizer, q = q, plot = True, plot_title_label = source_plot_title_label, dataset_label = dataset_label)
        q_german = length_statistics(target_filename_preprocessed, target_tokenizer, q = q, bos = True, eos = True, plot = True, plot_title_label = target_plot_title_label, dataset_label = dataset_label)

        if length_truncation:
            truncate_length_and_tokenize(source_filename_preprocessed, source_tokenizer, q_english, target_filename_preprocessed, target_tokenizer, q_german)

        else:
            tokenize(source_filename_preprocessed, target_filename_preprocessed, source_tokenizer, target_tokenizer)

        if require_splitting:
             split_dataset(source_filename_preprocessed + "_tokenized", target_filename_preprocessed + "_tokenized", train = train_threshold, portion = data_portion)

    else:
        tokenize(source_filename_preprocessed, target_filename_preprocessed, source_tokenizer, target_tokenizer)

    if train:
        return source_tokenizer, target_tokenizer, q_english, q_german