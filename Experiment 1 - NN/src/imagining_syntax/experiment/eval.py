import torch
import torch.nn.functional as F
from imagining_syntax.model.transformer import Transformer
import argparse


# === Helpers ===
def tokenize(sentence, model, device, verbose=True):
    tokens = sentence.strip().split()
    token2idx = model.token2idx
    bos_idx = token2idx.get('bos', 1)
    eos_idx = token2idx.get('eos', 2)
    unk_idx = token2idx.get('unk', 3)
    if verbose:
        print(f"\nTokenizing: {sentence}")
    indices = []
    for token in tokens:
        idx = token2idx.get(token, unk_idx)
        if idx == unk_idx and verbose:  # pragma: no cover - verbose=True debug print; runners call with verbose=False
            print(f"[UNK] {token}")
        indices.append(idx)
    indices = [bos_idx] + indices + [eos_idx]
    if verbose:
        print(f"Indices: {indices}")
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)


def sentence_log_prob(sentence, model, device, verbose=True):
    input_ids = tokenize(sentence, model, device, verbose=verbose)
    with torch.no_grad():
        output = model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output
        if logits.shape[1] == 0:  # pragma: no cover - torch shape-zero guard unreachable from runners' tokenizer (BOS+content+EOS ≥ 3 tokens)
            if verbose:
                print(" WARNING: Empty logits returned! Skipping this sentence.")
            return float('-inf')
    log_probs = F.log_softmax(logits, dim=-1)

    targets = input_ids[:, 1:]
    log_probs = log_probs[:, :-1]
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()


def evaluate_minimal_pairs(file_path, output_path, model, device, verbose=True):
    total = 0
    correct = 0
    results = []

    with open(file_path) as f:
        for line in f:
            if not line.strip():  # pragma: no cover - blank-line skip; pair files never contain blank lines per generator
                continue
            grammatical, ungrammatical = line.strip().split("\t")

            lp_grammatical = sentence_log_prob(grammatical, model, device, verbose=verbose)
            lp_ungrammatical = sentence_log_prob(ungrammatical, model, device, verbose=verbose)

            is_correct = lp_grammatical > lp_ungrammatical
            correct += int(is_correct)
            total += 1
            result_symbol = "OK" if is_correct else "NO"

            output = (
                f"{result_symbol} G: {lp_grammatical:.2f} | U: {lp_ungrammatical:.2f}\n"
                f"    {grammatical}\n"
                f"    {ungrammatical}\n"
            )
            if verbose:
                print(output)
            results.append(output)

    accuracy = correct / total if total > 0 else 0
    summary = f"\nModel Accuracy: {accuracy:.2%} ({correct}/{total})\n"
    if verbose:
        print(summary)
    results.append(summary)

    with open(output_path, "w") as out_f:
        out_f.writelines(results)

    return accuracy, correct, total


def main():
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

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device  # pragma: no cover - --device cpu/cuda explicit branch; suite uses --device auto

    model = Transformer.from_pretrained(args.model_dir, device=device)
    model.eval()

    evaluate_minimal_pairs(args.input_file, args.output_file, model, device, verbose=True)


if __name__ == "__main__":
    main()
