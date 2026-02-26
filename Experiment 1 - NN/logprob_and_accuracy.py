import torch
import torch.nn.functional as F
from model import Transformer
import os
import argparse

# === Parse arguments ===
parser = argparse.ArgumentParser(description='Evaluate model accuracy on minimal pairs')
parser.add_argument('--model_dir', type=str, required=True,
                   help='Path to model directory')
parser.add_argument('--input_file', type=str, required=True,
                   help='Path to minimal pairs file')
parser.add_argument('--output_file', type=str, required=True,
                   help='Path to output accuracy file')
parser.add_argument('--device', type=str, default='auto',
                   help='Device to use (cuda/cpu/auto, default: auto)')

args = parser.parse_args()

# === Settings ===
MODEL_DIR = args.model_dir
if args.device == 'auto':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    DEVICE = args.device
INPUT_FILE = args.input_file
OUTPUT_FILE = args.output_file

# === Load model ===
model = Transformer.from_pretrained(MODEL_DIR, device=DEVICE)
model.eval()

token2idx = model.token2idx
idx2token = model.idx2token
bos_idx = token2idx.get('bos', 1)
eos_idx = token2idx.get('eos', 2)
unk_idx = token2idx.get('unk', 3)

# === Helpers ===
def tokenize(sentence):
    tokens = sentence.strip().split()
    print(f"\nTokenizing: {sentence}")
    indices = []
    for token in tokens:
        idx = token2idx.get(token, unk_idx)
        if idx == unk_idx:
            print(f"[UNK] {token}")
        indices.append(idx)
    indices = [bos_idx] + indices + [eos_idx]
    print(f"Indices: {indices}")
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

def sentence_log_prob(sentence):
    input_ids = tokenize(sentence)
    with torch.no_grad():
        output = model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output
        if logits.shape[1] == 0:
            print(" WARNING: Empty logits returned! Skipping this sentence.")
            return float('-inf')  # Penalize this sentence so it won't be chosen
    log_probs = F.log_softmax(logits, dim=-1)

    targets = input_ids[:, 1:]
    log_probs = log_probs[:, :-1]
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()


def evaluate_minimal_pairs(file_path, output_path):
    total = 0
    correct = 0
    results = []

    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
            grammatical, ungrammatical = line.strip().split("\t")

            lp_grammatical = sentence_log_prob(grammatical)
            lp_ungrammatical = sentence_log_prob(ungrammatical)

            is_correct = lp_grammatical > lp_ungrammatical
            correct += int(is_correct)
            total += 1
            result_symbol = "✓" if is_correct else "✗"

            output = (
                f"{result_symbol} G: {lp_grammatical:.2f} | U: {lp_ungrammatical:.2f}\n"
                f"    {grammatical}\n"
                f"    {ungrammatical}\n"
            )
            print(output)
            results.append(output)

    accuracy = correct / total if total > 0 else 0
    summary = f"\nModel Accuracy: {accuracy:.2%} ({correct}/{total})\n"
    print(summary)
    results.append(summary)

    with open(output_path, "w") as out_f:
        out_f.writelines(results)

# === Run ===
if __name__ == "__main__":
    evaluate_minimal_pairs(INPUT_FILE, OUTPUT_FILE)
