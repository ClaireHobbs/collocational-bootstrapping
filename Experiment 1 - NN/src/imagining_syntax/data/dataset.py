import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Dict, Tuple


class TextDataset(Dataset):
    def __init__(self, file_path: str, max_length: int, token2idx: Dict[str, int], special_tokens: Dict[str, int]):
        self.token2idx = token2idx
        self.special_tokens = special_tokens
        self.max_length = max_length
        self.samples = []

        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if 0 < len(tokens) <= max_length:
                    self.samples.append(tokens)

        print(f"Loaded {len(self.samples)} samples from {file_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        token_ids = [self.token2idx.get(token, self.token2idx['unk']) for token in tokens]
        token_ids = [self.special_tokens['bos']] + token_ids + [self.special_tokens['eos']]
        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor(token_ids[1:], dtype=torch.long)
        return x, y

def pad_collate_fn(batch, pad_idx: int):
    """Collate function for DataLoader that handles padding."""
    xs, ys = zip(*batch)
    
    # Find max length in this batch
    max_len = max(x.size(0) for x in xs)
    
    # Pad sequences
    padded_xs = []
    padded_ys = []
    masks = []
    
    for x, y in zip(xs, ys):
        pad_len = max_len - x.size(0)
        
        # Pad input and target
        padded_x = torch.cat([x, torch.full((pad_len,), pad_idx, dtype=torch.long)])
        padded_y = torch.cat([y, torch.full((pad_len,), -1, dtype=torch.long)])  # -1 will be ignored in loss
        
        # Create padding mask (1 for real tokens, 0 for padding)
        mask = torch.cat([
            torch.ones(x.size(0), dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long)
        ])
        
        padded_xs.append(padded_x)
        padded_ys.append(padded_y)
        masks.append(mask)
    
    return torch.stack(padded_xs), torch.stack(padded_ys), torch.stack(masks)


def save_idx2token(idx2token, directory):
    fo = open(os.path.join(directory, "idx2token.tsv"), "w")

    for idx in idx2token:
        fo.write(str(idx) + "\t" + idx2token[idx] + "\n")

    fo.close()

def load_idx2token(directory):
    idx2token = {}
    token2idx = {}

    fi = open(os.path.join(directory, "idx2token.tsv"), "r")

    for line in fi:
        idx, token = line.strip().split("\t")
        idx2token[int(idx)] = token
        token2idx[token] = int(idx)

    return idx2token, token2idx




