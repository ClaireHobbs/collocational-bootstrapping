"""System tests for the `imsyn gen pairs` CLI."""
import subprocess

import pytest


def _run_minimal_pairs(out_dir, output_file, *, param="0.5", dist_type="zipfian",
                      noun_OOD=False, prep_obj_mismatch=False,
                      num_pairs="20", vocab="8", unseen="2", seed="42"):
    cmd = [
        "imsyn", "gen", "pairs",
        str(param),
        "--output_dir", str(out_dir),
        "--output_file", output_file,
        "--num_pairs", num_pairs,
        "--vocab_size", vocab,
        "--unseen_count", unseen,
        "--seed", seed,
    ]
    if dist_type == "oneshot":
        cmd.append("--oneshot")
    elif dist_type != "zipfian":
        raise AssertionError(f"unsupported dist_type in test helper: {dist_type}")
    if noun_OOD:
        cmd.append("--noun_OOD")
    if prep_obj_mismatch:
        cmd.append("--prep_obj_mismatch")
    return cmd


def test_minimal_pairs_file_has_one_pair_per_line_with_tab_separator(
    tmp_path, tiny_subprocess_env
):
    """Each line is `grammatical\\tungrammatical`; line count equals --num_pairs."""
    subprocess.run(
        _run_minimal_pairs(tmp_path, "pairs.txt"),
        env=tiny_subprocess_env, check=True,
    )

    lines = (tmp_path / "pairs.txt").read_text().splitlines()
    assert len(lines) == 20, f"expected 20 pairs, got {len(lines)}"
    for line in lines:
        parts = line.split("\t")
        assert len(parts) == 2, f"expected tab-separated grammatical+ungrammatical, got: {line!r}"
        assert parts[0] != parts[1], f"grammatical and ungrammatical are identical: {line!r}"


def test_noun_ood_flag_runs_without_error(
    tmp_path, tiny_subprocess_env
):
    """--noun_OOD switches to unseen noun-verb pairings; CLI accepts the flag."""
    subprocess.run(
        _run_minimal_pairs(tmp_path, "ood_pairs.txt", param="1.0", noun_OOD=True),
        env=tiny_subprocess_env, check=True,
    )
    assert (tmp_path / "ood_pairs.txt").exists()


def test_prep_obj_mismatch_flag_runs_without_error(
    tmp_path, tiny_subprocess_env
):
    """--prep_obj_mismatch generates pairs where preposition objects mismatch number."""
    subprocess.run(
        _run_minimal_pairs(tmp_path, "mismatch_pairs.txt", prep_obj_mismatch=True),
        env=tiny_subprocess_env, check=True,
    )
    assert (tmp_path / "mismatch_pairs.txt").exists()


def test_same_seed_produces_byte_identical_pairs_file(
    tmp_path, tiny_subprocess_env
):
    """Re-running with same --seed produces byte-identical pair files."""
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    out_a.mkdir(); out_b.mkdir()
    for out in (out_a, out_b):
        subprocess.run(
            _run_minimal_pairs(out, "pairs.txt"),
            env=tiny_subprocess_env, check=True,
        )
    assert (out_a / "pairs.txt").read_text() == (out_b / "pairs.txt").read_text()


# ---------------------------------------------------------------------------
# Direct-import tests for flip_verb (fast tier).
# ---------------------------------------------------------------------------


def test_flip_verb_with_default_args_returns_string_for_known_verb():
    """flip_verb on a sentence ending in a known singular verb returns the plural form."""
    from imagining_syntax.data.minimal_pairs import flip_verb

    # 'twirls' is in verbs_singular; flip_verb should return the plural form 'twirl'.
    result = flip_verb("the dog twirls")
    assert result == "the dog twirl", f"unexpected flip_verb output: {result!r}"


def test_flip_verb_returns_none_for_unknown_verb():
    """flip_verb returns None when the final word is not in either verb list."""
    from imagining_syntax.data.minimal_pairs import flip_verb

    result = flip_verb("the dog xyz")
    assert result is None
