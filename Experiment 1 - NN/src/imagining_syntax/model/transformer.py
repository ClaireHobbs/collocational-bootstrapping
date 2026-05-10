"""
Adapted from nanoGPT by Andrej Karpathy:
https://github.com/karpathy/nanoGPT/blob/master/model.py
Added functions:
- save_model: Saves the model and configuration to a specified directory.
- from_pretrained: Loads a model from a specified directory with device auto-detection.

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from dataclasses import dataclass, field
from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import math
import yaml
import os
from pathlib import Path

from imagining_syntax.data.dataset import save_idx2token, load_idx2token


# ==================== Decoder-only Transformer Implementation (from NanoGPT) ====================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Register causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))
        
    def forward(self, x, padding_mask=None):
        B, T, C = x.size()
        
        # Project to query, key, value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:T, :T] == 0, float('-inf'))
        
        # Apply padding mask if provided
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            att = att.masked_fill(padding_mask == 0, float('-inf'))
            
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        
        # Combine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, padding_mask=None):
        x = x + self.attn(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Weight tying AFTER initialization
        self.lm_head.weight = self.transformer.wte.weight  # Tie weights
        
        # Mappings between indices and tokens
        self.token2idx = None
        self.idx2token = None


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None, padding_mask=None):
        device = idx.device
        b, t = idx.size()
        
        # Position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Forward pass
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, padding_mask)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                 targets.view(-1), ignore_index=-1)
        else:
            #logits = self.lm_head(x[:, [-1], :])
            logits = self.lm_head(x)
            loss = None
            
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def save_model(self, save_dir: str):
        """Save model and config to directory"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))
        
        # Save config as YAML
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'block_size': self.config.block_size,
            'n_layer': self.config.n_layer,
            'n_head': self.config.n_head,
            'n_embd': self.config.n_embd,
            'dropout': self.config.dropout,
            'bias': self.config.bias,
            'special_tokens': {
                'pad': self.config.special_tokens.pad,
                'bos': self.config.special_tokens.bos,
                'eos': self.config.special_tokens.eos
            }
        }
        
        with open(os.path.join(save_dir, "config.yaml"), 'w') as f:
            yaml.dump(config_dict, f)

        save_idx2token(self.idx2token, save_dir)
    
    @classmethod
    def from_pretrained(cls, save_dir: str, device=None):
        """Load model from directory with device auto-detection"""
        # Load config
        with open(os.path.join(save_dir, "config.yaml")) as f:
            config_dict = yaml.safe_load(f)
        
        # Auto-select device if not specified
        if device is None:  # pragma: no cover - device fallback in from_pretrained; callers always pass device explicitly
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Recreate config
        special_tokens = SpecialTokens(
            pad=config_dict['special_tokens']['pad'],
            bos=config_dict['special_tokens']['bos'],
            eos=config_dict['special_tokens']['eos']
        )
        
        config = ModelConfig(
            vocab_size=config_dict['vocab_size'],
            block_size=config_dict['block_size'],
            n_layer=config_dict['n_layer'],
            n_head=config_dict['n_head'],
            n_embd=config_dict['n_embd'],
            dropout=config_dict['dropout'],
            bias=config_dict['bias'],
            special_tokens=special_tokens
        )
        
        # Create model on CPU first
        model = cls(config).to('cpu')
        
        # Load weights with proper device mapping
        state_dict = torch.load(
            os.path.join(save_dir, "model.pt"),
            map_location=torch.device('cpu')
        )
        
        # Handle potential weight tying
        if 'lm_head.weight' not in state_dict:  # pragma: no cover - back-compat for state_dicts predating weight-tying save fix
            state_dict['lm_head.weight'] = state_dict['wte.weight']
        
        model.load_state_dict(state_dict)
        
        # Move to final device
        model = model.to(device)

        # Put into eval mode
        model.eval()

        idx2token, token2idx = load_idx2token(save_dir)
        model.idx2token = idx2token
        model.token2idx = token2idx
        
        return model
        
        
# ==================== Model Configuration ====================

@dataclass
class SpecialTokens:
    pad: int = 0
    bos: int = 1
    eos: int = 2

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
    # special_tokens: SpecialTokens = SpecialTokens()
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)
