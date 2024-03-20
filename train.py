import torch
import os
import csv
import matplotlib.pyplot as plt
import tqdm
import time
import json
import evaluate
import youtokentome as yttm

from torch.utils.checkpoint import checkpoint
from typing import Type, List
from model import Transformer

plt.style.use("ggplot")

SACREBLEU = evaluate.load("sacrebleu")
BLEU = evaluate.load("bleu")

class LambdaLRWrapper:
    def __init__(self, optimizer : torch.optim.Optimizer, epoch_period, gamma):
        self.epoch_period = epoch_period
        self.gamma = gamma
        self.step_counter = 0

        def _get_lr(epoch):
            if epoch == 0:
                return 1
            
            return self.gamma ** (self.step_counter + 1) if (epoch + 1) % self.epoch_period == 0 else self.gamma ** (self.step_counter)
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = _get_lr)

    def step(self):
        self.scheduler.step()
        self.step_counter += 1
    
def loss(pred, y, reduction):
    """
    Compute (possibly label smoothed) cross entropy loss given model predictions and true labels.
        pred -> (batch_size, time_steps_max, V)
        y -> (batch_size, time_steps_max)
        padding index of 0 should not contribute to final loss as it comes from padding tokens.
    """
    return torch.nn.functional.cross_entropy(pred.transpose(-1, -2), y, ignore_index = 0, label_smoothing = 0.1, reduction = reduction)

@torch.no_grad()
def eval_loss(model : Transformer, loader : torch.utils.data.DataLoader, device : str) -> float:
    """
    Evaluate the model over a given dataset.
    """
    num_examples = 0
    total_loss = 0
    model.eval()

    for X_s, X_t, y in tqdm.tqdm(loader):
        X_s = X_s.to(device)
        X_t = X_t.to(device)
        y = y.to(device)
        b = X_s.shape[0]
        out = model.forward(X_s, X_t)
        current_loss = loss(out, y, 'mean').item() * b
        total_loss += current_loss
        num_examples += b
        del X_s, X_t, y, out

    model.train()
    return total_loss / num_examples

@torch.no_grad()
def eval_bleu(model : Transformer, device : str, loader : torch.utils.data.DataLoader, source_tokenizer : yttm.BPE, target_tokenizer : yttm.BPE, inference_method : str, 
                   beam_size : int = None, k : int = None, metric : str = "sacrebleu") -> float:
    assert inference_method in ["topk", "beam", "greedy"], f"Invalid inference method selected: {inference_method}, terminating script."

    model.eval()
    candidates, references = [], []

    for X_s, X_t, _ in tqdm.tqdm(loader):
        X_s = X_s.to(device)
        X_t = X_t.to(device)

        for i in range(X_s.shape[0]):
            if inference_method == "greedy":
                candidate = model.inference_greedy("", source_tokenizer, target_tokenizer, source_tokens = X_s[i].unsqueeze(0))
            
            elif inference_method == "beam":
                candidate = model.inference_beam("", source_tokenizer, target_tokenizer, source_tokens = X_s[i].unsqueeze(0), beam_size = beam_size)
            
            else:
                candidate = model.inference_topk("", source_tokenizer, target_tokenizer, source_tokens = X_s[i].unsqueeze(0), k = k)
            
            candidates.append(candidate)
            reference = target_tokenizer.decode(X_t[i].tolist(), ignore_ids = [target_tokenizer.subword_to_id("<BOS>"), target_tokenizer.subword_to_id("<EOS>"), target_tokenizer.subword_to_id("<PAD>")])
            references.append(reference)

    model.train()
    return SACREBLEU.compute(predictions = candidates, references = references)["score"] if metric == "sacrebleu" else BLEU.compute(predictions = candidates, references = references)["bleu"]

def eval_bleu_serialize(model : Transformer, device : str, loader : torch.utils.data.DataLoader, source_tokenizer : yttm.BPE, 
                             target_tokenizer : yttm.BPE, inference_methods : List[str], model_directory : str, beam_size : int = None, k : int = None, metric : str = "sacrebleu"):

    with open(os.path.join(model_directory, ".metadata.json"), "r", encoding = "utf-8") as f:
        d = json.load(f)
    
    scores = {}

    for method in inference_methods:
        print(f"Computing {metric} scores with inference method: {method}")
        
        if method == "topk":
            print(f"k = {k}")

        elif method == "beam":
            print(f"beam_size = {beam_size}")

        score = eval_bleu(model, device, loader, source_tokenizer, target_tokenizer, method, beam_size = beam_size, k = k, metric = metric)

        if method == "greedy":
            scores[method] = {
                "score": score
            }

        elif method == "topk":
            scores[method] = {
                "score": score,
                "k": k
            }
        
        elif method == "beam":
            scores[method] = {
                "score": score,
                "beam_size": beam_size
            }
    
    d[f"{metric}_scores"] = scores

    with open(os.path.join(model_directory, ".metadata.json"), "w+", encoding = "utf-8") as f:
        json.dump(d, f, indent = 4)

def load_state(checkpoint_dir : str, model_label : str, label_suffix : str, device : str, model_class : Type[torch.nn.Module], optimizer_class : Type[torch.optim.Optimizer]) -> dict:
    labels = os.listdir(checkpoint_dir)
    labels = list(filter(lambda x : model_label in x and f"{label_suffix}.pt" in x, labels))
    labels = list(map(lambda x : os.path.join(checkpoint_dir, x), labels))

    if len(labels) == 0:
        # Given checkpoint directory is empty.
        return {
            "model": None,
            "optimizer": None,
            "lr_scheduler": None,
            "last_epoch": None,
            "last_train_step": None,
            "validation_loss": None,
            "best_validation_loss": None,
            "label_smoothing": None,
            "max_grad_norm": None,
            "epoch_evaluation_period": None
        }

    d = torch.load(labels[0])
    model = model_class.from_dict(d["model"], device)

    optimizer = optimizer_class(model.parameters())
    optimizer.load_state_dict(d["optimizer"])

    lr_scheduler = None
    if "lr_scheduler" in d:
        lr_scheduler = LambdaLRWrapper(optimizer, d["lr_scheduler_epoch_period"], d["lr_scheduler_gamma"])
        lr_scheduler.scheduler.load_state_dict(d["lr_scheduler"])

    with open(os.path.join(checkpoint_dir, ".metadata.json"), "r", encoding = "utf-8") as f:
        metadata = json.load(f)
        best_validation_loss = metadata["best_validation_loss"]
        label_smoothing = metadata["label_smoothing"]
        max_grad_norm = metadata["max_grad_norm"]
        epoch_evaluation_period = metadata["epoch_evaluation_period"]

    return {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "epoch": d["epoch"],
        "validation_loss": d["validation_loss"],
        "best_validation_loss": best_validation_loss,
        "label_smoothing": label_smoothing,
        "max_grad_norm": max_grad_norm,
        "epoch_evaluation_period": epoch_evaluation_period
    }

def write_loss(checkpoint_dir : str, 
               model_label : str, 
               epoch_counter : int,
               validation_loss : float) -> None:
    
    loss_output_file = os.path.join(checkpoint_dir, "{}_metrics.csv".format(model_label))
    with open(loss_output_file, "a", newline = "") as f:
        writer = csv.writer(f, delimiter = ";")
        writer.writerow([epoch_counter, validation_loss])

def serialize_metadata(filename : str,
                   epoch_counter : int,
                   epochs : int,
                   total_batches : int,
                   model : Transformer,
                   optimizer : torch.optim.Optimizer,
                   label_smoothing : float,
                   best_validation_loss : float,
                   lr_scheduler : LambdaLRWrapper = None,
                   initial_lr : float = -1,
                   batch_size : int = -1,
                   epoch_evaluation_period : int = None,
                   max_grad_norm : float = None) -> None:
    """
    Serialize metadata about entire training procedure into .metadata.json file of a model's directory.
    """

    if os.path.exists(filename):
        with open(filename, "r") as f:
            d = json.load(f)

    else:
        d = None
    
    model_params = model.save("", dict_export = True)
    del model_params["state_dict"]

    optimizer_params = optimizer.state_dict()["param_groups"][0]
    if "params" in optimizer_params:
        del optimizer_params["params"]

    with open(filename, "w+") as f:
        new_d = {
            "best_validation_loss" : best_validation_loss,
            "initial_lr" : initial_lr if d is None else d["initial_lr"],
            "batch_size": batch_size if d is None else d["batch_size"],
            "epoch_counter": epoch_counter,
            "epochs": epochs,
            "total_batches": total_batches,
            "epoch_evaluation_period": epoch_evaluation_period,
            "max_grad_norm": max_grad_norm,
            "model_class": model.__class__.__mro__[0].__name__,
            "model_params": model_params,
            "optimizer_class": optimizer.__class__.__mro__[0].__name__,
            "optimizer_params": optimizer_params,
            "label_smoothing" : label_smoothing
        }

        if lr_scheduler is not None:
            new_d["lr_scheduler_class"] = "LambdaLRWrapper"
            new_d["lr_scheduler_epoch_period"] = lr_scheduler.epoch_period
            new_d["lr_scheduler_gamma"] = lr_scheduler.gamma
            new_d["lr_scheduler_params"] = lr_scheduler.scheduler.state_dict()
        
        json.dump(new_d, f, indent = 4)

def serialize_state(epoch_counter : int, 
                    checkpoint_dir : str, 
                    model : Transformer, 
                    optimizer : torch.optim.Optimizer, 
                    model_label : str, 
                    validation_loss : float,
                    label_smoothing : float,
                    best_validation_loss : float,
                    lr_scheduler : LambdaLRWrapper = None,
                    initial_lr : float = -1,
                    batch_size : int = -1,
                    epoch_evaluation_period : int = None,
                    epochs : int = -1,
                    total_batches : int = 0,
                    max_grad_norm : float = None) -> None:
    """
    Serialize state of optimization during training. This function operates on two files inside checkpoint_dir:
        .metadata.json - Contains general information about model, optimizer, and latest optimization snapshot. Useful to see what was trained 
        and for how long without needing to load entire model.

        .pt file - Dictionary serialized with torch, it contains following keys:
            epoch - Epoch number corresponding to the snapshot
            train_step - Training step number corresponding to the snapshot
            validation_loss - Loss of the model on validation dataset
            validation_bleu - BLEU of the model on validation dataset
            model - Contains dictionary for the model as returned by save function
            optimizer - Contains state dictionary of the optimizer
            lr_scheduler - State dict of LR scheduler. Assume LambdaLR which decays learning rate by 0.1 every 6-th epoch.
    """

    filename = os.path.join(checkpoint_dir, ".metadata.json")
    serialize_metadata(filename,
                    epoch_counter,
                    epochs,
                    total_batches, 
                    model, 
                    optimizer,
                    label_smoothing,
                    best_validation_loss,
                    lr_scheduler = lr_scheduler,
                    initial_lr = initial_lr,
                    batch_size = batch_size,
                    epoch_evaluation_period = epoch_evaluation_period,
                    max_grad_norm = max_grad_norm)

    model_dict = model.save("", dict_export = True)
    dump_path = os.path.join(checkpoint_dir, f"{model_label}.pt")
    optimizer_dict = optimizer.state_dict()

    # Add model parameters and validation parameters
    d = {
        "epoch": epoch_counter,
        "validation_loss": validation_loss,
        "model": model_dict,
        "optimizer": optimizer_dict
    }

    if lr_scheduler is not None:
        d["lr_scheduler_epoch_period"] = lr_scheduler.epoch_period
        d["lr_scheduler_gamma"] = lr_scheduler.gamma
        d["lr_scheduler"] = lr_scheduler.scheduler.state_dict()

    torch.save(d, dump_path)

def train(model : Transformer,
          epochs : int,
          train_loader : torch.utils.data.DataLoader,
          validation_loader : torch.utils.data.DataLoader,
          optimizer : torch.optim.Optimizer,
          device : str,
          lr_scheduler : torch.optim.lr_scheduler.LRScheduler = None,
          label_smoothing : float = 0,
          max_grad_norm : float = None,
          checkpoint_dir : str = None,
          model_label : str = "model",
          epoch_evaluation_period : int = 1,
          document_loss : bool = True,
          last_epoch : int = 0,
          truncate_loss : bool = False,
          best_validation_loss : float = float("+inf")):
    
    if checkpoint_dir is not None and not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    loss_output_file = os.path.join(checkpoint_dir, "{0}_metrics.csv".format(model_label)) if document_loss else None

    if document_loss and (not os.path.exists(loss_output_file) or truncate_loss):
        with open(loss_output_file, "w+", newline = "") as f:
            writer = csv.writer(f, delimiter = ";")
            writer.writerow(["Epoch", "Validation Loss"])

    total_batches = len(train_loader)

    initial_lr, batch_size = -1, -1
    # Extract initial learning rate from the optimizer 
    if last_epoch == 0:
        initial_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        batch_size = next(iter(train_loader))[0].shape[0]

    for epoch_counter in range(last_epoch, epochs):
        step_counter = 0

        train_step_bar = tqdm.tqdm(total = total_batches, 
                                    unit = "time step", desc = "Epoch {:d}/{:d} Training".format(epoch_counter, epochs), 
                                    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
                                    initial = 1,
                                    dynamic_ncols = True)        

        for X_s, X_t, y in train_loader:
            X_s = X_s.to(device)
            X_t = X_t.to(device)
            y = y.to(device)
            out = model.forward(X_s, X_t)
            current_loss = loss(out, y, "mean")
            current_loss.backward()
            
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            train_step_bar.update()
            step_counter += 1

            del out, current_loss, X_s, X_t, y
            torch.cuda.empty_cache()
        
        pre_elapsed_time, post_elapsed_time = train_step_bar.format_dict["elapsed"], 0
        train_step_bar.close()
        # Empty space between consecutive progress bars.
                    
        if epoch_counter % epoch_evaluation_period == 0:
            start_timestamp = time.time()
            tqdm.tqdm.write("Computing evaluation loss")
            validation_loss = eval_loss(model, validation_loader, device)
            
            post_elapsed_time = time.time() - start_timestamp
            tqdm.tqdm.write(f"Validation loss: {validation_loss}")

            if checkpoint_dir is not None and validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                
                # Save best so far with special suffix to the model label, so it is easily recognizible
                # from the file name.
                serialize_state(epoch_counter + 1,
                checkpoint_dir, 
                model, 
                optimizer,
                model_label + "-best",
                validation_loss, 
                label_smoothing,
                lr_scheduler = lr_scheduler,
                initial_lr = initial_lr,
                batch_size = batch_size,
                best_validation_loss = best_validation_loss,
                epoch_evaluation_period = epoch_evaluation_period,
                epochs = epochs,
                total_batches = total_batches,
                max_grad_norm = max_grad_norm)

            # Regular checkpointing
            serialize_state(epoch_counter + 1, 
            checkpoint_dir, 
            model, 
            optimizer,
            model_label + "-checkpoint",
            validation_loss,
            label_smoothing,
            lr_scheduler = lr_scheduler,
            initial_lr = initial_lr,
            batch_size = batch_size,
            best_validation_loss = best_validation_loss,
            epoch_evaluation_period = epoch_evaluation_period,
            epochs = epochs,
            total_batches = total_batches,
            max_grad_norm = max_grad_norm)

            if document_loss:
                write_loss(checkpoint_dir, model_label, epoch_counter, validation_loss)

        tqdm.tqdm.write("Time elapsed for current epoch: {0} (only training and validation time is taken into account)".format(time.strftime("%H:%M:%S", time.gmtime(pre_elapsed_time + post_elapsed_time))))
        print()

        if lr_scheduler is not None:
            lr_scheduler.step()

def plot_metric(model_directory : str, metric_filename : str, metric_label : str, plot_title : str = ""):
    assert metric_label in ["loss", "BLEU"], f"{metric_label} has to be in {['loss', 'BLEU']}"

    loss_path = os.path.join(model_directory, metric_filename)
    metric = []

    with open(loss_path, "r") as f:
        reader = csv.reader(f, delimiter = ";")

        # Skip header of the loss file
        next(reader)
        for row in reader:
            metric.append(float(row[1]) if metric_label == "loss" else "BLEU")
    
    _, ax = plt.subplots()
    ax.plot(metric, label = metric_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    ax.set_title(plot_title)
    ax.legend()
    plt.show()