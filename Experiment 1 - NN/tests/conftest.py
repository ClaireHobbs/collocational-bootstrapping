"""Shared pytest fixtures for the system-test suite.

All tests use the CLI boundary or run_training import. CUDA is forced off via
CUDA_VISIBLE_DEVICES="" because (a) CUDA is non-deterministic for byte-equality
checks and (b) for the tiny test configs (1 layer, 32-dim) CPU is faster
than CUDA due to kernel-launch overhead.
"""
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def force_cpu():
    """Force CPU for every test in the session.

    We don't restore the previous value; CUDA_VISIBLE_DEVICES is process-local.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield


@pytest.fixture
def repo_root():
    """Absolute path to the repository root (where the runner scripts live)."""
    return REPO_ROOT


@pytest.fixture
def tiny_data_dir(tmp_path):
    """Copy tiny_*.txt fixtures into a fresh temp dir; yield the dir path."""
    dest = tmp_path / "data"
    dest.mkdir()
    for name in ("tiny_train.txt", "tiny_val.txt", "tiny_test.txt"):
        target_name = name.replace("tiny_", "")  # train.txt, val.txt, test.txt
        shutil.copy(FIXTURES / name, dest / target_name)
    return dest


@pytest.fixture
def tiny_minimal_pairs_file(tmp_path):
    """Copy the minimal-pairs fixture into a temp dir; yield the path."""
    dest = tmp_path / "tiny_minimal_pairs.tsv"
    shutil.copy(FIXTURES / "tiny_minimal_pairs.tsv", dest)
    return dest


@pytest.fixture
def tiny_training_args(tiny_data_dir, tmp_path):
    """argparse.Namespace with the smallest training config that still trains."""
    return Namespace(
        config=None,
        data_dir=str(tiny_data_dir),
        model_save_dir=str(tmp_path / "model"),
        txt_file_name=None,
        saved_model_name=None,
        train_size=50,
        val_size=10,
        test_size=10,
        batch_size=8,
        max_len=20,
        n_layer=1,
        n_head=2,
        n_embed=32,
        dropout=0.1,
        max_iter=10,
        eval_interval=5,
        lr=6e-4,
        seed=42,
    )


@pytest.fixture
def tiny_subprocess_env():
    """Environment dict for subprocess.run: CPU only, deterministic hashing."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONHASHSEED"] = "0"
    return env


@pytest.fixture
def python_executable():
    """Absolute path to the venv's python; falls back to sys.executable."""
    return sys.executable


def pytest_configure(config):
    """Exclude 'science' tests by default unless -m science is explicitly passed."""
    # Only add the exclusion if no explicit marker filter is already in place
    if not config.option.markexpr:
        config.option.markexpr = "not science"
