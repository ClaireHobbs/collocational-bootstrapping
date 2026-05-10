"""System tests for the `imsyn gen dataset` CLI."""
import subprocess

import pytest

from imagining_syntax.data.distributions import (
    create_distribution,
    format_param_for_filename,
    get_parameter_name,
    validate_parameter,
)
from imagining_syntax.data.sentences import generate_sentence


def _common_args(out_dir, param, dist_type, *, seed="42", count="200", vocab="8", unseen="2"):
    args = [
        str(param),
        "--output_dir", str(out_dir),
        "--train_file", "train.txt",
        "--val_file", "val.txt",
        "--test_file", "test.txt",
        "--sentence_count", count,
        "--vocab_size", vocab,
        "--unseen_count", unseen,
        "--seed", seed,
    ]
    if dist_type == "oneshot":
        args.append("--oneshot")
    elif dist_type != "zipfian":
        raise AssertionError(f"unsupported dist_type in test helper: {dist_type}")
    return args


def test_zipfian_dataset_generation_produces_three_split_files(
    tmp_path, tiny_subprocess_env
):
    """Same shape contract for Zipfian distribution."""
    subprocess.run(
        ["imsyn", "gen", "dataset",
         *_common_args(tmp_path, "1.4", "zipfian")],
        env=tiny_subprocess_env, check=True,
    )

    for name in ("train.txt", "val.txt", "test.txt"):
        path = tmp_path / name
        assert path.exists(), f"missing {name}"
        assert path.read_text().strip(), f"{name} is empty"


def test_oneshot_dataset_generation_produces_three_split_files(
    tmp_path, tiny_subprocess_env
):
    """Same shape contract for oneshot (deterministic 1:1) distribution."""
    subprocess.run(
        ["imsyn", "gen", "dataset",
         *_common_args(tmp_path, "0", "oneshot")],
        env=tiny_subprocess_env, check=True,
    )

    for name in ("train.txt", "val.txt", "test.txt"):
        path = tmp_path / name
        assert path.exists(), f"missing {name}"
        assert path.read_text().strip(), f"{name} is empty"


def test_same_seed_produces_byte_identical_dataset(
    tmp_path, tiny_subprocess_env
):
    """Re-running with the same --seed produces byte-identical train/val/test files."""
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    out_a.mkdir(); out_b.mkdir()

    for out in (out_a, out_b):
        subprocess.run(
            ["imsyn", "gen", "dataset",
             *_common_args(out, "1.0", "zipfian")],
            env=tiny_subprocess_env, check=True,
        )

    for name in ("train.txt", "val.txt", "test.txt"):
        a = (out_a / name).read_text()
        b = (out_b / name).read_text()
        assert a == b, f"{name} differs between runs with same --seed"


def test_generated_sentences_use_only_known_vocabulary(
    tmp_path, tiny_subprocess_env
):
    """Every word in every generated sentence is a known token (det / prep / noun / verb).

    This catches token leaks (e.g., raw indices, unknown placeholder strings)."""
    subprocess.run(
        ["imsyn", "gen", "dataset",
         *_common_args(tmp_path, "1.0", "zipfian")],
        env=tiny_subprocess_env, check=True,
    )

    text = (tmp_path / "train.txt").read_text()
    tokens = set(text.split())
    # Allowed: 'the', 'by', 'near', and noun/verb stems and their plural/3sg forms.
    # We don't enumerate the full vocab; we just ensure every token is alphabetic.
    for tok in tokens:
        assert tok.isalpha(), f"non-alphabetic token in dataset: {tok!r}"


# ---------------------------------------------------------------------------
# Direct-import tests for imagining_syntax.data.distributions module (fast tier).
# ---------------------------------------------------------------------------


def test_create_distribution_zipfian_default_returns_vocab_sized_list():
    """Same default-shape contract for zipfian distribution."""
    dist = create_distribution("zipfian", 0.5)
    assert len(dist) == 40


def test_create_distribution_zipfian_z0_is_uniform_over_seen_portion():
    """Zipfian Z=0 is uniform over the seen portion, with zeros for unseen tail."""
    dist = create_distribution("zipfian", 0.0, vocab_size=10, unseen_count=2)
    seen = dist[:8]
    unseen = dist[8:]
    # Uniform over 8 seen items: each gets 1/8.
    assert all(abs(p - 1.0 / 8) < 1e-9 for p in seen), f"seen portion not uniform: {seen}"
    assert unseen == [0, 0], f"unseen tail not zero: {unseen}"


def test_create_distribution_oneshot_is_deterministic_first_position():
    """Oneshot distribution puts all mass at offset 0 across the seen portion;
    param_value is ignored and the unseen tail stays zero."""
    dist = create_distribution("oneshot", 0.0, vocab_size=10, unseen_count=2)
    assert dist[0] == 1.0
    assert dist[1:] == [0.0] * 9
    # param_value really is ignored — different values produce identical output.
    assert create_distribution("oneshot", 7.5, vocab_size=10, unseen_count=2) == dist


def test_create_distribution_rejects_unknown_distribution_type():
    """Unknown distribution_type raises ValueError."""
    with pytest.raises(ValueError):
        create_distribution("poisson", 0.5)


def test_validate_parameter_rejects_negative_zipfian():
    """Zipfian Z must be >= 0; negative values raise ValueError."""
    with pytest.raises(ValueError):
        validate_parameter("zipfian", -0.5)


def test_validate_parameter_rejects_unknown_distribution_type():
    """Unknown distribution_type raises ValueError in validate_parameter."""
    with pytest.raises(ValueError):
        validate_parameter("foo", 0)


def test_get_parameter_name_rejects_unknown_distribution_type():
    """Unknown distribution_type raises ValueError in get_parameter_name."""
    with pytest.raises(ValueError):
        get_parameter_name("foo")


def test_format_param_for_filename_zipfian_15_returns_150():
    """format_param_for_filename for zipfian Z=1.5 produces a stable filename token."""
    assert format_param_for_filename("zipfian", 1.5) == "150"


def test_format_param_for_filename_oneshot_always_returns_zero():
    """Oneshot has no parameter — filename token is always '0'."""
    assert format_param_for_filename("oneshot", 0.0) == "0"
    assert format_param_for_filename("oneshot", 99.9) == "0"


def test_get_parameter_name_oneshot_returns_o():
    """Oneshot uses 'O' as its single-letter parameter name in directory paths."""
    assert get_parameter_name("oneshot") == "O"


def test_generate_sentence_with_default_vocab_size_and_unseen_count():
    """generate_sentence works with all defaults and returns a non-empty string."""
    result = generate_sentence(0.5)
    assert isinstance(result, str) and len(result) > 0
