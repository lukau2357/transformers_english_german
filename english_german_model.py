import os
import torch
import youtokentome as yttm
import os

from torch.utils.data import DataLoader
from model import Transformer
from train import train, load_state, LambdaLRWrapper, eval_bleu_serialize
from dataset import data_pipeline, TranslationDataset, translation_dataset_collate, max_sequence_lengths, preprocess_line, split_dataset

CHECKPOINT_DIRECTORY = "english_german_1"
MODEL_LABEL = "english_german_1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = os.path.join("..", "datasets", "multi_30k")
BPE_DIR = "english_german_bpes"

if not os.path.exists(BPE_DIR):
    os.mkdir(BPE_DIR)

SOURCE_LANGUAGE_TOKENIZER_FILENAME = os.path.join(".", BPE_DIR, "english_bpe.yttm")
TARGET_LANGUAGE_TOKENIZER_FILENAME = os.path.join(".", BPE_DIR, "german_bpe.yttm")

SOURCE_LANGUAGE_FILENAME_TRAIN = os.path.join(DATASET_DIR, "train.en")
TARGET_LANGUAGE_FILENAME_TRAIN = os.path.join(DATASET_DIR, "train.de")
SOURCE_LANGUAGE_FILENAME_PREPROCESSED_TRAIN = SOURCE_LANGUAGE_FILENAME_TRAIN + "_v2"
TARGET_LANGUAGE_FILENAME_PREPROCESSED_TRAIN = TARGET_LANGUAGE_FILENAME_TRAIN + "_v2"
SOURCE_LANGUAGE_TOKENIZED_TRAIN = SOURCE_LANGUAGE_FILENAME_PREPROCESSED_TRAIN + "_tokenized"
TARGET_LANGUAGE_TOKENIZED_TRAIN= TARGET_LANGUAGE_FILENAME_PREPROCESSED_TRAIN + "_tokenized"

SOURCE_LANGUAGE_FILENAME_VAL = os.path.join(DATASET_DIR, "test_2016_flickr.en")
TARGET_LANGUAGE_FILENAME_VAL = os.path.join(DATASET_DIR, "test_2016_flickr.de")
SOURCE_LANGUAGE_FILENAME_PREPROCESSED_VAL = SOURCE_LANGUAGE_FILENAME_VAL + "_v2"
TARGET_LANGUAGE_FILENAME_PREPROCESSED_VAL = TARGET_LANGUAGE_FILENAME_VAL + "_v2"
SOURCE_LANGUAGE_TOKENIZED_VAL = SOURCE_LANGUAGE_FILENAME_PREPROCESSED_VAL + "_tokenized"
TARGET_LANGUAGE_TOKENIZED_VAL= TARGET_LANGUAGE_FILENAME_PREPROCESSED_VAL + "_tokenized"

VOCAB_SIZE_MAX = 30000

HYPERPARAMETERS = {
    "d_model": 512,
    "num_heads": 8,
    "hidden_factor": 4,
    "num_layers": 6,
    "batch_size": 128,
    "learning_rate": 5e-4,
    "epochs": 20
}

if __name__ == "__main__":
    torch.manual_seed(1337)

    # Preprocessing train data and training tokenizers
    '''
    print("Preprocessing train data")

    source_tokenizer, target_tokenizer, max_source_length, max_target_length = data_pipeline(SOURCE_LANGUAGE_FILENAME_TRAIN, 
                TARGET_LANGUAGE_FILENAME_TRAIN, 
                SOURCE_LANGUAGE_TOKENIZER_FILENAME,
                TARGET_LANGUAGE_TOKENIZER_FILENAME,
                VOCAB_SIZE_MAX,
                train = True,
                quantile = 1,
                target_plot_title_label = "English",
                source_plot_title_label = "German",
                dataset_label = "multi_30k",
                length_heuristic = False,
                length_truncation = False)

    data_pipeline(SOURCE_LANGUAGE_FILENAME_VAL, 
                TARGET_LANGUAGE_FILENAME_VAL, 
                SOURCE_LANGUAGE_TOKENIZER_FILENAME,
                TARGET_LANGUAGE_TOKENIZER_FILENAME,
                VOCAB_SIZE_MAX,
                train = False,
                quantile = 1,
                target_plot_title_label = "English",
                source_plot_title_label = "German",
                dataset_label = "multi_30k",
                length_heuristic = False,
                length_truncation = False)

    HYPERPARAMETERS["max_source_sequence_length"] = max_source_length
    HYPERPARAMETERS["max_target_sequence_length"] = max_target_length
    HYPERPARAMETERS["vocab_size_source"] = source_tokenizer.vocab_size()
    HYPERPARAMETERS["vocab_size_target"] = target_tokenizer.vocab_size()

    train_dataset = TranslationDataset(SOURCE_LANGUAGE_TOKENIZED_TRAIN, TARGET_LANGUAGE_TOKENIZED_TRAIN)
    val_dataset = TranslationDataset(SOURCE_LANGUAGE_TOKENIZED_VAL, TARGET_LANGUAGE_TOKENIZED_VAL)
    
    train_loader = DataLoader(train_dataset, HYPERPARAMETERS["batch_size"], shuffle = True, collate_fn = translation_dataset_collate)                  
    val_loader = DataLoader(val_dataset, HYPERPARAMETERS["batch_size"], shuffle = True, collate_fn = translation_dataset_collate)
    '''
    # On multiple iterations executes faster than waiting for data processing
    '''
    source_tokenizer = yttm.BPE(SOURCE_LANGUAGE_TOKENIZER_FILENAME)
    target_tokenizer = yttm.BPE(TARGET_LANGUAGE_TOKENIZER_FILENAME)
    max_source_length, max_target_length = max_sequence_lengths(SOURCE_LANGUAGE_TOKENIZED_TRAIN, TARGET_LANGUAGE_TOKENIZED_TRAIN)

    HYPERPARAMETERS["max_source_sequence_length"] = max_source_length
    HYPERPARAMETERS["max_target_sequence_length"] = max_target_length
    HYPERPARAMETERS["vocab_size_source"] = source_tokenizer.vocab_size()
    HYPERPARAMETERS["vocab_size_target"] = target_tokenizer.vocab_size()

    train_dataset = TranslationDataset(SOURCE_LANGUAGE_TOKENIZED_TRAIN, TARGET_LANGUAGE_TOKENIZED_TRAIN)
    val_dataset = TranslationDataset(SOURCE_LANGUAGE_TOKENIZED_VAL, TARGET_LANGUAGE_TOKENIZED_VAL)
    
    train_loader = DataLoader(train_dataset, HYPERPARAMETERS["batch_size"], shuffle = True, collate_fn = translation_dataset_collate)                  
    val_loader = DataLoader(val_dataset, HYPERPARAMETERS["batch_size"], shuffle = True, collate_fn = translation_dataset_collate)

    model = Transformer(HYPERPARAMETERS["vocab_size_source"], 
                        HYPERPARAMETERS["vocab_size_target"],
                        HYPERPARAMETERS["d_model"],
                        HYPERPARAMETERS["num_layers"],
                        HYPERPARAMETERS["num_heads"],
                        HYPERPARAMETERS["max_source_sequence_length"],
                        HYPERPARAMETERS["max_target_sequence_length"],
                        HYPERPARAMETERS["hidden_factor"],
                        DEVICE,
                        source_tokenizer.subword_to_id("<PAD>"))
    '''

    # Training the model
    '''
    betas = (HYPERPARAMETERS.get("beta_1", 0.99), HYPERPARAMETERS.get("beta_2", 0.999))
    optimizer = torch.optim.Adam(model.parameters(), 
                                 HYPERPARAMETERS["learning_rate"],
                                 betas, 
                                 weight_decay = 0 if "weight_decay" not in HYPERPARAMETERS else HYPERPARAMETERS["weight_decay"])
        
    scheduler = LambdaLRWrapper(optimizer, 5, 0.95)
    model.count_parameters()
    
    train(model, HYPERPARAMETERS["epochs"], train_loader, val_loader, optimizer, DEVICE,
          checkpoint_dir = CHECKPOINT_DIRECTORY, 
          model_label = MODEL_LABEL,
          lr_scheduler = scheduler,
          label_smoothing = 0.05)
    '''

    # Proceeding from checkpoint
    '''
    d = load_state(CHECKPOINT_DIRECTORY, MODEL_LABEL, "checkpoint", DEVICE, Transformer, torch.optim.Adam)
    model = d["model"]
    optimizer = d["optimizer"]
    lr_scheduler = d["lr_scheduler"]

    train(model, HYPERPARAMETERS["epochs"], train_loader, val_loader, optimizer, DEVICE,
          checkpoint_dir = CHECKPOINT_DIRECTORY,
          model_label = MODEL_LABEL,
          lr_scheduler = lr_scheduler,
          last_epoch = d["epoch"],
          best_validation_loss = d["best_validation_loss"],
          label_smoothing = d["label_smoothing"],
          max_grad_norm = d["max_grad_norm"],
          epoch_evaluation_period = d["epoch_evaluation_period"])

    # Loading the model, performing final evaluation and inference
    '''
    '''
    d = load_state(CHECKPOINT_DIRECTORY, MODEL_LABEL, "best", DEVICE, Transformer, torch.optim.Adam)
    model = d["model"]
    val_dataset = TranslationDataset(SOURCE_LANGUAGE_TOKENIZED_VAL, TARGET_LANGUAGE_TOKENIZED_VAL)
    val_loader = DataLoader(val_dataset, HYPERPARAMETERS["batch_size"], shuffle = True, collate_fn = translation_dataset_collate)
    source_tokenizer = yttm.BPE(SOURCE_LANGUAGE_TOKENIZER_FILENAME)
    target_tokenizer = yttm.BPE(TARGET_LANGUAGE_TOKENIZER_FILENAME)
    k = 10
    beam_size = 10
    eval_bleu_serialize(model, DEVICE, val_loader, source_tokenizer, target_tokenizer, ["greedy", "topk", "beam"], CHECKPOINT_DIRECTORY, beam_size = beam_size , k = k, metric = "bleu")
    eval_bleu_serialize(model, DEVICE, val_loader, source_tokenizer, target_tokenizer, ["greedy", "topk", "beam"], CHECKPOINT_DIRECTORY, beam_size = beam_size , k = k)
    '''

    # Live inference of loaded model
    d = d = load_state(CHECKPOINT_DIRECTORY, MODEL_LABEL, "best", DEVICE, Transformer, torch.optim.Adam)
    model = d["model"]
    source_tokenizer = yttm.BPE(SOURCE_LANGUAGE_TOKENIZER_FILENAME)
    target_tokenizer = yttm.BPE(TARGET_LANGUAGE_TOKENIZER_FILENAME)
    k = 10
    beam_size = 10
    model.eval()
    model.live_inference(source_tokenizer, target_tokenizer, ["greedy", "topk", "beam"], k = k, beam_size = beam_size)