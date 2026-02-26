import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Dict, Tuple


# ==================== PCFG Parser and Dataset Generator ====================
class PCFG:
    def __init__(self):
        self.rules = defaultdict(list)
        self.non_terminals = set()
        self.terminals = set()
        self.start_symbol = None
        
    @classmethod
    def from_file(cls, file_path: str):
        '''
        Load a PCFG from a file.
        Args:
            file_path (str): Path to the PCFG file.
        Returns:
            pcfg: An instance of the PCFG class.
        '''
        pcfg = cls()
        all_symbols = set()
        
        # First pass: collect all non-terminals (LHS of rules)
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line[0] == "#":
                    continue
                
                parts = line.split()
                prob = float(parts[0])
                lhs = parts[1] # LHS is the second part of the rule
                rhs = tuple(parts[2:])
                
                pcfg.non_terminals.add(lhs)
                all_symbols.add(lhs)
                all_symbols.update(list(rhs))  # Add all RHS symbols
                
                pcfg.rules[lhs].append((prob, rhs))
                
        pcfg.terminals = all_symbols - pcfg.non_terminals
        
        # Determine start symbol (first non-terminal in first rule)
        if pcfg.rules:
            pcfg.start_symbol = next(iter(pcfg.rules.keys()))
        
        return pcfg
    
    def generate(self, symbol: str = None, max_depth: int = 10) -> List[str]:
        '''
        Generate a sequence of symbols from the PCFG.
        Args:
            symbol (str): The starting symbol to generate from. If None, use the start symbol.
            max_depth (int): Maximum depth of recursion.
        Returns:
            List[str]: A list of generated symbols.
        '''
        if symbol is None: # Use the start symbol if not provided
            symbol = self.start_symbol
        if max_depth <= 0:
            return []
            
        if symbol in self.terminals:
            return [symbol]
            
        # Select a rule randomly according to probabilities
        possible_rules = self.rules.get(symbol, [])
        if not possible_rules:
            return []
            
        probs, rhs_list = zip(*possible_rules)
        probs = np.array(probs)
        probs /= probs.sum()  # (redundant step for a well-defined PCFG) Normalize probabilities
        
        selected_idx = np.random.choice(len(possible_rules), p=probs)
        selected_rhs = rhs_list[selected_idx]
        
        result = []
        for s in selected_rhs:
            result.extend(self.generate(s, max_depth-1))
        return result

# Code to read in dataset instead of pcfg
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

class PCFGDataset(Dataset):
    def __init__(self, pcfg: PCFG, num_samples: int, max_length: int, 
                 token2idx: Dict[str, int], special_tokens: Dict[str, int]):
        self.pcfg = pcfg
        self.max_length = max_length
        self.token2idx = token2idx
        self.special_tokens = special_tokens
        self.samples = []
        
        # Generate samples
        for _ in range(num_samples):
            while True:
                sequence = self.pcfg.generate(max_depth=20)
                if 0 < len(sequence) <= max_length:
                    break
            self.samples.append(sequence)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        '''
        Prepare a sample for the DataLoader, where x contains the input tokens with BOS, 
        and y contains the target tokens without BOS but with EOS.
        Args:
            idx (int): Index of the sample.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
        '''
        sequence = self.samples[idx]
        # Convert to token indices
        token_ids = [self.token2idx[token] for token in sequence]
        
        # Add BOS and EOS tokens
        token_ids = [self.special_tokens['bos']] + token_ids + [self.special_tokens['eos']]
        
        # Create input and target (shifted by one)
        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor(token_ids[1:], dtype=torch.long)
        
        return x, y

def create_token_mappings(pcfg: PCFG, special_tokens: Dict[str, int]) -> Dict[str, int]:
    """Create mappings between tokens and indices."""
    # All terminals become tokens
    tokens = sorted(pcfg.terminals)
    
    # Create vocabulary
    token2idx = {token: idx + len(special_tokens) for idx, token in enumerate(tokens)}
    
    # Add special tokens
    for token, idx in special_tokens.items():
        token2idx[token] = idx
    
    idx2token = {idx: token for token, idx in token2idx.items()}
    
    return token2idx, idx2token

def pcfg_collate_fn(batch, pad_idx: int):
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




