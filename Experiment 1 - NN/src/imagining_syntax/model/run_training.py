from dataclasses import dataclass, field
from typing import Any
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from typing import List, Dict, Tuple
import math
import torch.nn as nn
from torch.nn import functional as F
import inspect
import argparse
import json
import os
import yaml

from imagining_syntax.data.dataset import TextDataset, pad_collate_fn, save_idx2token

from imagining_syntax.model.transformer import *
from imagining_syntax.model.training import *
from imagining_syntax.utils.seed import set_global_seed, make_torch_generator

# ==================== ArgParse ====================

def parse_args_and_config():
    """
    Parse command line arguments and config file, with command line arguments
    taking precedence over config file values.
    
    Returns:
        args: Namespace object containing all configuration parameters
    """
    parser = argparse.ArgumentParser(description="Transformer model training and evaluation")
    
    # Config file argument
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    
    # dataset file
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing train.txt, val.txt, and test.txt files")
    parser.add_argument("--model_save_dir", type=str, required=True,
                        help="Directory to save the trained model")
    

    
    # dataset parameters
    parser.add_argument("--train_size", type=int, default=10000,
                        help="Size of the training datastet")
    parser.add_argument("--val_size", type=int, default=2000,
                        help="Size of the validation datastet")
    parser.add_argument("--test_size", type=int, default=2000,
                        help="Size of the test datastet")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for dividing the dataset")
    
    # model parameters
    parser.add_argument("--max_len", type=int, default=50,
                        help="Max length of the sequences to generate")
    parser.add_argument("--n_layer", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_embed", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="General dropout rate")
    
    # training hyperparameters
    parser.add_argument("--max_iter", type=int, default=1200,
                        help="Max number of batches to train on")
    parser.add_argument("--eval_interval", type=int, default=300,
                        help="Number of batch for evaluation result to print out")
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="Learning rate during training")

    # reproducibility
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible training (default: unseeded)")

    # First parse args to check if a config file is provided
    args, unknown = parser.parse_known_args()
    
    # Load config from file if provided
    config_dict = {}
    if args.config is not None and os.path.exists(args.config):  # pragma: no cover - YAML --config branch; operator-only feature, manually verified outside the F.1 audit suite
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    
    # Update default values with config file values
    parser.set_defaults(**config_dict)
    
    # Parse again with updated defaults
    args = parser.parse_args()
    
    return args

# ==================== Main Execution ====================

def run_training(args):
    """Run training given an already-parsed args namespace.

    Mirrors the original main() body. Returns a dict with the trained model's
    save dir, final test loss, train/val loss curves, and the token mappings
    inferred from the data dir.
    """
    if getattr(args, "seed", None) is not None:
        set_global_seed(args.seed)

    special_tokens = {
        'pad': 0,
        'bos': 1,
        'eos': 2,
        'unk': 3
    }

    # Load and tokenize sentences
    train_path = os.path.join(args.data_dir, "train.txt")
    val_path = os.path.join(args.data_dir, "val.txt")
    test_path = os.path.join(args.data_dir, "test.txt") 

    all_tokens = set()

    #for file_name in ["train.txt", "val.txt", "test.txt"]:
        #with open(f"data/{file_name}", 'r') as f:

    for file_path in [train_path, val_path, test_path]:
        with open(file_path, 'r') as f:
            for line in f:
                all_tokens.update(line.strip().split())


    tokens = sorted(all_tokens)
    token2idx = {token: idx + len(special_tokens) for idx, token in enumerate(tokens)}
    #token2idx['unk'] = len(token2idx)  # for unknowns
    for token, idx in special_tokens.items():
        token2idx[token] = idx

    idx2token = {idx: token for token, idx in token2idx.items()}
    
    
    # Create datasets
    # Create three files training, validation, test 12,000 (10,000/1,000/1,000)
    train_dataset = TextDataset(train_path, args.max_len, token2idx, special_tokens)
    val_dataset = TextDataset(val_path, args.max_len, token2idx, special_tokens)
    test_dataset = TextDataset(test_path, args.max_len, token2idx, special_tokens)



    
    # Create data loaders
    batch_size = args.batch_size
    train_generator = (make_torch_generator(args.seed)
                       if getattr(args, "seed", None) is not None else None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            generator=train_generator,
                            collate_fn=lambda b: pad_collate_fn(b, pad_idx=special_tokens['pad']))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          collate_fn=lambda b: pad_collate_fn(b, pad_idx=special_tokens['pad']))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=lambda b: pad_collate_fn(b, pad_idx=special_tokens['pad']))
    print(f'One epoch of training is {len(train_loader)} iterations/batches.')
    
    ###################### Define model and training configurations ######################
    # Model config
    model_config = ModelConfig(
        vocab_size=len(token2idx),
        block_size=args.max_len,  # Should match max_length in dataset
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embed,
        dropout=args.dropout,
        bias=True,
        special_tokens=SpecialTokens(
            pad=special_tokens['pad'],
            bos=special_tokens['bos'],
            eos=special_tokens['eos']
        )
    )
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(model_config).to(device)
    model.token2idx = token2idx
    model.idx2token = idx2token
    
    # Training config
    train_config = {
        'max_iters': args.max_iter,
        'eval_interval': args.eval_interval,
        'learning_rate': args.lr,
        'weight_decay': 0.1,
        'betas': (0.9, 0.95),
        'device': device,
        'grad_clip': 1.0
    }
    
    ###################### Training and Evaluations ######################
    # Train the model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, train_config, args.model_save_dir
    )

    # Test the model
    test_loss = evaluate_model(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")

    losses_path = os.path.join(args.model_save_dir, 'losses.json')
    if os.path.exists(losses_path):
        with open(losses_path, 'r') as f:
            losses_data = json.load(f)
        losses_data['test_loss'] = test_loss
        with open(losses_path, 'w') as f:
            json.dump(losses_data, f, indent=2)

    return {
        "model_save_dir": args.model_save_dir,
        "test_loss": test_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": min(val_losses) if val_losses else float("inf"),
        "token2idx": token2idx,
        "idx2token": idx2token,
    }


def main():
    args = parse_args_and_config()
    run_training(args)


if __name__ == "__main__":
    main()
