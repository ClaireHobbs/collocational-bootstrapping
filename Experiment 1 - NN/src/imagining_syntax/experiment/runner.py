"""Shared training+eval primitives for the experiment runners.

Each function is a building block. Runners (single, sweep) compose them into
experiment-specific flows.
"""
import os
import torch
from argparse import Namespace
from pathlib import Path

from imagining_syntax.data.generate import generate_unique_sentences, split_and_save
from imagining_syntax.data.minimal_pairs import generate_minimal_pairs, save_pairs
from imagining_syntax.experiment.eval import evaluate_minimal_pairs
from imagining_syntax.model.transformer import Transformer
from imagining_syntax.utils.seed import set_global_seed
from imagining_syntax.model.run_training import run_training


CONDITIONS = ("seen_match", "seen_mismatch", "unseen_match", "unseen_mismatch")


def prepare_data_dir(alpha, seed, output_root, *,
                     num_sentences=12000, vocab_size=40, unseen_count=10,
                     distribution_type="zipfian"):
    """Generate train/val/test for a single (alpha, seed) and write to disk.

    Args:
        alpha: Distribution parameter value (alpha for Zipfian).
        seed: Random seed for reproducibility. If None, sentence sampling is
              left unseeded (matches the sub-script CLI behavior).
        output_root: Path object; data is written under output_root/data/.
        num_sentences: Total before train/val/test split.
        vocab_size: Number of seen noun-verb pairs.
        unseen_count: Number of held-out unseen pairs.
        distribution_type: "zipfian" or "oneshot".

    Returns:
        Path to the data directory containing train.txt, val.txt, test.txt.
    """
    data_dir = Path(output_root) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        set_global_seed(seed)
    sentences = generate_unique_sentences(
        count=num_sentences,
        param_value=alpha,
        distribution_type=distribution_type,
        vocab_size=vocab_size,
        unseen_count=unseen_count,
    )
    split_and_save(
        sentences,
        train_path=str(data_dir / "train.txt"),
        val_path=str(data_dir / "val.txt"),
        test_path=str(data_dir / "test.txt"),
    )
    return data_dir


def prepare_eval_sets(alpha, eval_seed, output_root, *,
                      num_pairs=1000, vocab_size=40, unseen_count=10,
                      distribution_type="zipfian"):
    """Generate the four fixed minimal-pair eval sets for a given alpha.

    For unseen conditions, alpha is overridden to a uniform distribution
    (Zipfian Z=0) so the unseen pairs are sampled uniformly.

    Args:
        alpha: Distribution parameter for SEEN conditions.
        eval_seed: Seed for fixed eval-set generation; deriving per-condition
                  seeds as eval_seed + offset keeps the four sets distinct
                  but reproducible. If None, pair sampling is left unseeded.
        output_root: Path object; pair files written under output_root/eval/.
        num_pairs: Pairs per condition.
        vocab_size, unseen_count, distribution_type: As in prepare_data_dir.


    Returns:
        Dict mapping condition name -> Path to the .tsv pair file.
    """
    eval_dir = Path(output_root) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    unseen_alpha = 0.0  # zipfian Z=0 / oneshot ignored param → uniform over seen tokens
    paths = {}
    spec = [
        ("seen_match",      alpha,         False, False),
        ("seen_mismatch",   alpha,         False, True),
        ("unseen_match",    unseen_alpha,  True,  False),
        ("unseen_mismatch", unseen_alpha,  True,  True),
    ]
    for offset, (cond, a, noun_OOD, mismatch) in enumerate(spec):
        if eval_seed is not None:
            set_global_seed(eval_seed + offset)
        pairs = generate_minimal_pairs(
            param_value=a,
            distribution_type=distribution_type,
            num_pairs=num_pairs,
            prep_obj_mismatch=mismatch,
            both_pps_present=True,
            noun_OOD=noun_OOD,
            vocab_size=vocab_size,
            unseen_count=unseen_count,
        )
        out_path = eval_dir / f"{cond}.tsv"
        save_pairs(pairs, str(out_path))
        paths[cond] = out_path
    return paths


def build_training_args(*, data_dir, model_save_dir, seed,
                        max_iter=1200, batch_size=32, max_len=50,
                        n_layer=2, n_head=4, n_embed=256,
                        dropout=0.1, lr=6e-4, eval_interval=300):
    """Construct the Namespace expected by imagining_syntax.model.run_training.run_training()."""
    return Namespace(
        config=None,
        data_dir=str(data_dir),
        model_save_dir=str(model_save_dir),
        train_size=10000, val_size=2000, test_size=2000,  # honored only by PCFG path
        batch_size=batch_size, max_len=max_len,
        n_layer=n_layer, n_head=n_head, n_embed=n_embed,
        dropout=dropout, max_iter=max_iter, eval_interval=eval_interval,
        lr=lr, seed=seed,
    )


def train_one(args, output_root, *, model_subdir="model"):
    """Run training; return (model_dir, train_result) where train_result is
    the dict returned by imagining_syntax.model.run_training.run_training()."""
    model_dir = Path(output_root) / model_subdir
    model_dir.mkdir(parents=True, exist_ok=True)
    args.model_save_dir = str(model_dir)
    train_result = run_training(args)
    return model_dir, train_result


def evaluate_loaded(model, eval_sets, output_root, *, conditions=CONDITIONS, device=None):
    """Evaluate an already-loaded model on every named eval set;
    return condition -> {accuracy, correct, total}.

    Used directly by callers that hold a live model and want to evaluate at
    intermediate training waypoints without a save/reload round-trip.
    """
    if device is None:  # pragma: no cover - cuda fallback; callers always pass device explicitly
        device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path(output_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    accuracies = {}
    for cond in conditions:
        if cond not in eval_sets:  # pragma: no cover - defensive skip for missing condition; callers pass matching eval_sets/conditions
            continue
        pair_path = eval_sets[cond]
        out_path = results_dir / f"accuracy_{cond}.txt"
        accuracy, correct, total = evaluate_minimal_pairs(
            str(pair_path), str(out_path), model, device, verbose=False,
        )
        accuracies[cond] = {"accuracy": accuracy, "correct": correct, "total": total}
    return accuracies


def evaluate_all(model_dir, eval_sets, output_root, *, conditions=CONDITIONS, device=None):
    """Load a saved model from disk and evaluate it; thin wrapper over evaluate_loaded."""
    if device is None:  # pragma: no cover - cuda fallback; callers always pass device explicitly
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer.from_pretrained(str(model_dir), device=device)
    model.eval()
    return evaluate_loaded(model, eval_sets, output_root, conditions=conditions, device=device)
