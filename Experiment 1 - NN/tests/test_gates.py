"""Gates from RESEARCH_PLAN.md Hard Rules: any change to imagining_syntax.model
(transformer / training / run_training) must keep these green."""
import hashlib
import subprocess

import torch

TINY_FLAGS = [
    "--train_size", "50",
    "--val_size", "10",
    "--test_size", "10",
    "--batch_size", "8",
    "--max_len", "20",
    "--n_layer", "1",
    "--n_head", "2",
    "--n_embed", "32",
    "--max_iter", "10",
    "--eval_interval", "5",
    "--lr", "6e-4",
    "--dropout", "0.1",
]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_seed_produces_byte_identical_model_checkpoint(
    tmp_path, tiny_data_dir, tiny_subprocess_env, python_executable
):
    """Training the same data with the same --seed twice produces byte-identical
    model.pt.

    This is the determinism gate from RESEARCH_PLAN.md Hard Rules: any change to
    imagining_syntax.model (transformer/training/run_training) must preserve this property."""
    out_a, out_b = tmp_path / "a", tmp_path / "b"

    for out in (out_a, out_b):
        subprocess.run(
            [python_executable, "-m", "imagining_syntax.model.run_training",
             "--data_dir", str(tiny_data_dir),
             "--model_save_dir", str(out),
             "--seed", "42",
             *TINY_FLAGS],
            env=tiny_subprocess_env, check=True,
        )

    hash_a = _sha256(out_a / "model.pt")
    hash_b = _sha256(out_b / "model.pt")
    assert hash_a == hash_b, (
        f"Same --seed produced different model.pt:\n  {hash_a}\n  {hash_b}"
    )


def test_cli_runs_without_seed_flag_and_produces_valid_checkpoint(
    tmp_path, tiny_data_dir, tiny_subprocess_env, python_executable
):
    """imagining_syntax.model.run_training without --seed still works (preserves pre-seed CLI surface)."""
    out = tmp_path / "out"

    result = subprocess.run(
        [python_executable, "-m", "imagining_syntax.model.run_training",
         "--data_dir", str(tiny_data_dir),
         "--model_save_dir", str(out),
         *TINY_FLAGS],
        env=tiny_subprocess_env, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"run_training failed without --seed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    model_pt = out / "model.pt"
    assert model_pt.exists(), f"model.pt not created at {model_pt}"
    state = torch.load(model_pt, map_location="cpu", weights_only=False)
    assert isinstance(state, dict), f"model.pt did not deserialize as a dict (got {type(state).__name__})"
    assert any("weight" in k for k in state), f"state-dict has no weights: keys={list(state)}"


def test_run_training_writes_losses_json_with_expected_keys(
    tmp_path, tiny_data_dir, tiny_subprocess_env, python_executable
):
    """After training, model_save_dir contains a losses.json with the keys the
    publication-era plot scripts depend on."""
    import json

    out = tmp_path / "out"
    subprocess.run(
        [python_executable, "-m", "imagining_syntax.model.run_training",
         "--data_dir", str(tiny_data_dir),
         "--model_save_dir", str(out),
         "--seed", "42",
         *TINY_FLAGS],
        env=tiny_subprocess_env, check=True,
    )

    losses_path = out / "losses.json"
    assert losses_path.exists(), f"losses.json not created at {losses_path}"

    data = json.loads(losses_path.read_text())
    expected = {"eval_steps", "train_losses", "val_losses", "best_step", "best_val_loss", "test_loss"}
    missing = expected - data.keys()
    assert not missing, f"losses.json missing keys: {missing}"
    # eval_steps must align 1:1 with the recorded loss series.
    assert len(data["eval_steps"]) == len(data["train_losses"]) == len(data["val_losses"])


def test_different_seeds_produce_different_models(
    tmp_path, tiny_data_dir, tiny_subprocess_env, python_executable
):
    """Sanity: --seed 42 and --seed 43 produce different checkpoints.
    Catches a regression where seeding accidentally becomes a silent no-op."""
    out_42, out_43 = tmp_path / "s42", tmp_path / "s43"

    for seed, out in [("42", out_42), ("43", out_43)]:
        subprocess.run(
            [python_executable, "-m", "imagining_syntax.model.run_training",
             "--data_dir", str(tiny_data_dir),
             "--model_save_dir", str(out),
             "--seed", seed,
             *TINY_FLAGS],
            env=tiny_subprocess_env, check=True,
        )

    assert _sha256(out_42 / "model.pt") != _sha256(out_43 / "model.pt"), (
        "Different seeds produced byte-identical models — seeding is a no-op."
    )
