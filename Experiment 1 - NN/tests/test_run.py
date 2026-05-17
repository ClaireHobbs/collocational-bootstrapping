"""Characterization + resume tests for `imsyn run`.

Slow-tier tests use --experiment-name to anchor output directories under
runs/ for predictable cleanup.
"""
import csv
import hashlib
import os
import signal
import subprocess
import time

import pytest


@pytest.mark.slow
def test_comprehensive_zipfian_sweep_produces_csv_with_expected_rows(
    tmp_path, tiny_subprocess_env, repo_root
):
    """Sweeping Z in {0, 1.0} with 1 iteration yields a CSV with one row per
    (param, eval_type) — at least 8 rows for 2 params × 4 eval types.

    NOTE: We use the runner's default vocab-size=40 / unseen-count=10. With
    smaller values (e.g., vocab=8/unseen=2) the unseen noun-verb pair space
    becomes smaller than 1000, and `imsyn gen pairs`'s deduplicating loop in
    `while len(pairs) < num_pairs` cannot terminate."""
    exp_name = f"test_basic_{tmp_path.name}"
    exp_dir = repo_root / "runs" / exp_name

    try:
        subprocess.run(
            ["imsyn", "run",
             "--parameter-range", "0.0,1.0", "--step", "1.0", "--n-iterations", "1",
             "--experiment-name", exp_name],
            env=tiny_subprocess_env, check=True, timeout=1800,
        )

        csv_files = list(exp_dir.rglob("comprehensive_results.csv"))
        assert csv_files, f"no CSV under {exp_dir}; tree: {list(exp_dir.rglob('*'))}"
        csv_path = csv_files[0]

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 8, f"expected >=8 rows (2 params x 4 conds), got {len(rows)}"

        required = {"param_value", "eval_type", "mean", "std", "min", "max"}
        assert required.issubset(rows[0].keys()), (
            f"CSV missing columns: required={required}, got={set(rows[0])}"
        )

        param_dirs = list((exp_dir / "experiments").glob("Z_*"))
        assert len(param_dirs) >= 2, f"expected >=2 Z_* param dirs, got {param_dirs}"
    finally:
        import shutil
        shutil.rmtree(exp_dir, ignore_errors=True)


@pytest.mark.slow
def test_comprehensive_resume_picks_up_where_it_left_off(
    tmp_path, tiny_subprocess_env, repo_root
):
    """Killing `imsyn run sweep` mid-sweep and restarting with --resume
    completes the sweep without re-doing finished iterations.

    On CPU we sleep ~180s before SIGTERM to give the runner time to complete at
    least the first param value (~150-180s per train including dataset gen and
    eval), so the resume has real prior work to skip."""
    exp_name = f"resume_target_{tmp_path.name}"
    exp_dir = repo_root / "runs" / exp_name

    try:
        cmd = [
            "imsyn", "run",
            "--parameter-range", "0.0,1.0", "--step", "0.5", "--n-iterations", "1",
            "--experiment-name", exp_name,
        ]
        proc = subprocess.Popen(
            cmd, env=tiny_subprocess_env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        # Give it ~180s (3 min) to complete at least the first param value, then SIGTERM.
        time.sleep(180)
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=60)

        assert exp_dir.exists(), f"experiment dir not created: {exp_dir}"

        # Resume
        subprocess.run(
            cmd + ["--resume", str(exp_dir)],
            env=tiny_subprocess_env, check=True, timeout=2400,
        )

        # 3 params × 4 conditions = 12 rows expected
        csv_candidates = list(exp_dir.rglob("comprehensive_results.csv"))
        assert csv_candidates, f"no CSV under {exp_dir} after resume"
        csv_path = csv_candidates[0]
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 12, f"expected >=12 rows after resume, got {len(rows)}"
    finally:
        import shutil
        shutil.rmtree(exp_dir, ignore_errors=True)


@pytest.mark.slow
def test_run_comprehensive_with_seed_produces_byte_identical_csv(
    tmp_path, tiny_subprocess_env, repo_root
):
    """Comprehensive sweep with the same --seed twice produces byte-identical
    CSV. Uses the runner's default vocab/unseen sizes; smaller values make
    `imsyn gen pairs`'s dedup loop non-terminating."""
    exp_name_a = f"seed_test_a_{tmp_path.name}"
    exp_name_b = f"seed_test_b_{tmp_path.name}"
    exp_dir_a = repo_root / "runs" / exp_name_a
    exp_dir_b = repo_root / "runs" / exp_name_b
    try:
        for exp_name in (exp_name_a, exp_name_b):
            subprocess.run(
                ["imsyn", "run",
                 "--parameter-range", "0.0,0.5", "--step", "0.5", "--n-iterations", "1",
                 "--experiment-name", exp_name,
                 "--seed", "42"],
                env=tiny_subprocess_env, check=True, timeout=1800,
            )

        def csv_hash(exp_dir):
            candidates = list(exp_dir.rglob("comprehensive_results.csv"))
            assert candidates, f"no comprehensive_results.csv under {exp_dir}"
            return hashlib.sha256(candidates[0].read_bytes()).hexdigest()

        h_a, h_b = csv_hash(exp_dir_a), csv_hash(exp_dir_b)
        assert h_a == h_b, (
            f"comprehensive_results.csv differs between runs with --seed 42:\n"
            f"  run_a={h_a}\n"
            f"  run_b={h_b}"
        )
    finally:
        import shutil
        shutil.rmtree(exp_dir_a, ignore_errors=True)
        shutil.rmtree(exp_dir_b, ignore_errors=True)


# ---------------------------------------------------------------------------
# Fast-tier argparse-validation error tests (no training; pure CLI exit).
# ---------------------------------------------------------------------------


def test_run_comprehensive_rejects_inverted_param_range():
    """param_min > param_max produces a non-zero exit and an explanatory error."""
    cmd = ["imsyn", "run", "--parameter-range", "1.0,0.5", "--step", "0.1", "--n-iterations", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 1, f"expected exit 1, got {result.returncode}; stderr={result.stderr!r}"
    combined = result.stdout + result.stderr
    assert "parameter-range MIN must be <= MAX" in combined, (
        f"expected error message about inverted range; got: {combined!r}"
    )


def test_run_comprehensive_rejects_zero_param_step():
    """param_step <= 0 produces a non-zero exit and an explanatory error."""
    cmd = ["imsyn", "run", "--parameter-range", "0.0,1.0", "--step", "0", "--n-iterations", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 1, f"expected exit 1, got {result.returncode}; stderr={result.stderr!r}"
    combined = result.stdout + result.stderr
    assert "--step must be positive" in combined, (
        f"expected error message about param_step; got: {combined!r}"
    )


def test_run_comprehensive_rejects_out_of_range_param_value():
    """validate_parameter raises -> main() prints 'Error:' and exits 1."""
    cmd = [
        "imsyn", "run",
        "--parameter-range=-0.5,-0.5", "--step", "0.1", "--n-iterations", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 1
    assert "Error:" in result.stdout or "Error:" in result.stderr


def test_run_comprehensive_resume_missing_dir_exits_non_zero():
    """`--resume <nonexistent>` prints an error and exits 1."""
    cmd = [
        "imsyn", "run",
        "--parameter-range", "0.0,1.0", "--step", "0.1", "--n-iterations", "1",
        "--resume", "/tmp/this_dir_does_not_exist_for_F1_audit",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 1
    assert "does not exist" in result.stdout or "does not exist" in result.stderr


def test_run_comprehensive_rejects_zero_iterations():
    """n_iterations < 1 produces a non-zero exit and an explanatory error."""
    cmd = ["imsyn", "run", "--parameter-range", "0.0,1.0", "--step", "0.1", "--n-iterations", "0"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 1, f"expected exit 1, got {result.returncode}; stderr={result.stderr!r}"
    combined = result.stdout + result.stderr
    assert "--n-iterations must be at least 1" in combined, (
        f"expected error message about n_iterations; got: {combined!r}"
    )


def test_load_progress_returns_dict_when_progress_file_exists(tmp_path):
    """load_progress reads the progress JSON and returns its dict."""
    from imagining_syntax.experiment.resume import load_progress

    progress_file = tmp_path / "progress.json"
    progress_file.write_text('{"last_completed_param": 0.5}')
    result = load_progress(str(progress_file))
    assert result == {"last_completed_param": 0.5}


def test_load_progress_returns_none_when_progress_file_missing(tmp_path):
    """load_progress returns None when the progress file does not exist.

    imagining_syntax.runners.run's `if progress_data:` guard (in main()) depends on
    this None to fall through to the 'no progress file found' warning path."""
    from imagining_syntax.experiment.resume import load_progress

    missing_path = tmp_path / "nonexistent_progress.json"
    assert load_progress(str(missing_path)) is None


# ---------------------------------------------------------------------------
# System tests for --extend.
#
# Group C (fast, CLI-only): test argparse validation without running the
# training pipeline. Uses `imsyn run` subprocess for realistic boundary.
#
# Group A (slow): test file-state assertions — directory layout, CSV
# aggregation, progress.json — by running a real tiny-config sweep via
# `imsyn run`.
#
# Group B (slow): test seed-sequence continuity by reading each
# iter_NNN/seed.txt artifact written by the runner before sub-scripts launch.
# ---------------------------------------------------------------------------


def _initial_sweep_subprocess(exp_name, n_iter, base_seed, tiny_subprocess_env, repo_root):
    """Run a full initial sweep via `imsyn run` with a tiny Zipfian range.

    Uses Z in {0.0, 0.2, 0.4} (3 param values) and the real runner binary so
    all on-disk artifacts are produced exactly as in production.  Runs from
    repo_root so runs/ lands there for cleanup.
    """
    subprocess.run(
        [
            "imsyn", "run",
            "--parameter-range", "0.0,0.4", "--step", "0.2", "--n-iterations", str(n_iter),
            "--experiment-name", exp_name,
            "--seed", str(base_seed),
        ],
        env=tiny_subprocess_env,
        check=True,
        timeout=3600,
        cwd=str(repo_root),
    )


# ---------------------------------------------------------------------------
# Group C: Fast CLI-validation tests (no training).
# ---------------------------------------------------------------------------


def test_run_comprehensive_extend_without_resume_dir_rejected():
    """--extend requires --resume DIR; passing --extend alone exits non-zero
    with a message saying so explicitly (not just argparse usage)."""
    cmd = [
        "imsyn", "run",
        "--parameter-range", "0.0,0.4", "--step", "0.2", "--n-iterations", "2",
        "--extend", "3",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "requires --resume" in combined, (
        f"expected explicit 'requires --resume' message, "
        f"not just argparse usage; got: {combined!r}"
    )


def test_run_comprehensive_extend_without_seed_anywhere_rejected(tmp_path):
    """--extend with --resume but no --seed AND no base_seed in progress.json
    exits with a clear message, since the seed derivation cannot be recovered."""
    fake_dir = tmp_path / "fake_resume_no_seed"
    fake_dir.mkdir()
    # Legacy-shape progress.json with no base_seed field.
    (fake_dir / "progress.json").write_text(
        '{"last_completed_param": 0.4, "status": "completed"}'
    )

    cmd = [
        "imsyn", "run",
        "--parameter-range", "0.0,0.4", "--step", "0.2", "--n-iterations", "2",
        "--extend", "3",
        "--resume", str(fake_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "needs a base seed" in combined, (
        f"expected explanation about missing base seed; got: {combined!r}"
    )


def test_count_existing_iterations_returns_zero_for_missing_directory(tmp_path):
    """count_existing_iterations safely returns 0 when the iterations/ dir
    does not exist — supports extending a sweep with a newly-added param value
    whose dir was never created in the prior run."""
    from imagining_syntax.experiment.resume import count_existing_iterations

    missing = tmp_path / "no_such_iter_dir"
    assert count_existing_iterations(str(missing)) == 0


def test_read_iter_result_skips_eval_types_whose_accuracy_file_is_missing(tmp_path):
    """read_iter_result returns a dict containing only eval types whose
    accuracy file is present — supports extending sweeps where some prior
    iterations have sparse results."""
    from imagining_syntax.experiment.resume import read_iter_result

    iter_dir = tmp_path / "iter_001"
    results_dir = iter_dir / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "accuracy_seen_match.txt").write_text(
        "Model Accuracy: 87.50% (875/1000)\n"
    )
    # Note: no accuracy file for unseen_match — should be silently skipped.

    out = read_iter_result(str(iter_dir),
                           ["seen_match", "unseen_match"])
    assert out == {"seen_match": 87.5}


def test_run_comprehensive_extend_with_zero_count_rejected(tmp_path):
    """--extend 0 is rejected at the CLI: extend must add at least one iteration."""
    # Create a minimal resume target so the validation we're testing is the one
    # that fires (not the missing-resume-dir one).
    fake_dir = tmp_path / "fake_resume"
    fake_dir.mkdir()
    (fake_dir / "progress.json").write_text('{"last_completed_param": 0.4}')

    cmd = [
        "imsyn", "run",
        "--parameter-range", "0.0,0.4", "--step", "0.2", "--n-iterations", "2",
        "--extend", "0",
        "--resume", str(fake_dir),
        "--seed", "42",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "--extend" in combined and ("must be" in combined or ">=" in combined), (
        f"expected error about --extend value; got: {combined!r}"
    )


# ---------------------------------------------------------------------------
# Group A: Slow system tests asserting on file-state after a real sweep.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_comprehensive_extend_creates_new_iter_directories_without_overwriting_existing(
    tmp_path, tiny_subprocess_env, repo_root
):
    """Running --extend creates iter_{k+1}..iter_{k+N} for each param dir
    while leaving the existing iter_001..iter_k directories untouched.

    Uses Z in {0.0, 0.2, 0.4} (3 param values), 2 initial iterations, then
    extends by 3 so the final layout is iter_001..iter_005 per param dir."""
    exp_name = f"extend_preserve_{tmp_path.name}"
    exp_dir = repo_root / "runs" / exp_name

    try:
        _initial_sweep_subprocess(exp_name, n_iter=2, base_seed=42,
                                  tiny_subprocess_env=tiny_subprocess_env,
                                  repo_root=repo_root)

        # Capture mtime and content of an existing accuracy file before extending.
        existing_acc = (
            exp_dir / "experiments" / "Z_0" / "iterations" / "iter_001"
            / "results" / "accuracy_seen_match.txt"
        )
        existing_mtime = existing_acc.stat().st_mtime
        existing_content = existing_acc.read_text()

        # Extend by 3 more iterations.
        subprocess.run(
            [
                "imsyn", "run",
                "--parameter-range", "0.0,0.4", "--step", "0.2", "--n-iterations", "2",
                "--experiment-name", exp_name,
                "--resume", str(exp_dir),
                "--extend", "3",
                "--seed", "42",
            ],
            env=tiny_subprocess_env,
            check=True,
            timeout=3600,
            cwd=str(repo_root),
        )

        # New iter dirs created at the next sequential indices for every param.
        for z_dir in ("Z_0", "Z_20", "Z_40"):
            iter_root = exp_dir / "experiments" / z_dir / "iterations"
            for new_idx in (3, 4, 5):
                new_iter = iter_root / f"iter_{new_idx:03d}"
                assert new_iter.is_dir(), (
                    f"expected new iter dir: {new_iter}; "
                    f"found: {sorted(iter_root.iterdir())}"
                )

        # Pre-existing iter_001 accuracy file is not modified.
        assert existing_acc.stat().st_mtime == existing_mtime, (
            f"pre-existing accuracy file was rewritten: {existing_acc}"
        )
        assert existing_acc.read_text() == existing_content

    finally:
        import shutil
        shutil.rmtree(exp_dir, ignore_errors=True)


@pytest.mark.slow
def test_run_comprehensive_extend_summary_json_aggregates_existing_and_new_iterations(
    tmp_path, tiny_subprocess_env, repo_root
):
    """After --extend, the per-param summary JSON reports statistics computed
    across all iterations (existing + newly added), not only the new ones.

    Starts with 2 iterations and extends by 3, so the per-param summary JSON
    should record n_iterations=5 and five raw_results entries."""
    exp_name = f"extend_summary_{tmp_path.name}"
    exp_dir = repo_root / "runs" / exp_name

    try:
        _initial_sweep_subprocess(exp_name, n_iter=2, base_seed=42,
                                  tiny_subprocess_env=tiny_subprocess_env,
                                  repo_root=repo_root)

        summary_csv = exp_dir / "summary" / "comprehensive_results.csv"
        assert summary_csv.exists(), (
            f"initial sweep did not produce summary CSV at {summary_csv}"
        )

        subprocess.run(
            [
                "imsyn", "run",
                "--parameter-range", "0.0,0.4", "--step", "0.2", "--n-iterations", "2",
                "--experiment-name", exp_name,
                "--resume", str(exp_dir),
                "--extend", "3",
                "--seed", "42",
            ],
            env=tiny_subprocess_env,
            check=True,
            timeout=3600,
            cwd=str(repo_root),
        )

        # Per-param JSON summary should report n_iterations = 5 (2 existing + 3 new).
        import json
        z00_summary = json.loads(
            (exp_dir / "experiments" / "Z_0" / "summary" / "summary.json").read_text()
        )
        assert z00_summary["n_iterations"] == 5, (
            f"per-param summary should reflect 2 + 3 iters; got {z00_summary['n_iterations']}"
        )
        assert len(z00_summary["raw_results"]) == 5, (
            f"per-param summary should record 5 raw_results entries; "
            f"got {len(z00_summary['raw_results'])}"
        )

    finally:
        import shutil
        shutil.rmtree(exp_dir, ignore_errors=True)


@pytest.mark.slow
def test_run_comprehensive_extend_persists_base_seed_in_progress_json(
    tmp_path, tiny_subprocess_env, repo_root
):
    """An initial run records its --seed as base_seed in progress.json so a
    later --extend invocation can recover the correct seed derivation without
    requiring the user to retype it."""
    exp_name = f"extend_seed_persist_{tmp_path.name}"
    exp_dir = repo_root / "runs" / exp_name

    try:
        _initial_sweep_subprocess(exp_name, n_iter=2, base_seed=137,
                                  tiny_subprocess_env=tiny_subprocess_env,
                                  repo_root=repo_root)

        import json
        progress = json.loads((exp_dir / "progress.json").read_text())
        assert progress.get("base_seed") == 137, (
            f"base_seed not recorded in progress.json: {progress}"
        )

    finally:
        import shutil
        shutil.rmtree(exp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Group B: Slow system tests asserting on seed.txt artifacts.
#
# The runner writes iter_seed to iter_NNN/seed.txt before launching
# sub-scripts.  These tests read that file to verify the seed-derivation
# rule (base_seed + 1000*p_idx + iter_idx) without inspecting subprocess args.
# ---------------------------------------------------------------------------


def _read_iter_seeds(exp_dir, z_dirs, iter_indices):
    """Collect integer seeds from seed.txt in each iter dir, in param order.

    Returns a flat list of ints: all iter_indices for z_dirs[0], then all for
    z_dirs[1], etc. — matching the order that run_experiments_for_param runs."""
    seeds = []
    for z_dir in z_dirs:
        for idx in iter_indices:
            seed_file = (
                exp_dir / "experiments" / z_dir / "iterations"
                / f"iter_{idx:03d}" / "seed.txt"
            )
            assert seed_file.exists(), f"seed.txt missing: {seed_file}"
            seeds.append(int(seed_file.read_text().strip()))
    return seeds


@pytest.mark.slow
def test_run_comprehensive_extend_adds_iterations_with_continuing_seed_sequence(
    tmp_path, tiny_subprocess_env, repo_root
):
    """Extending a finished sweep with --extend N adds N more iterations to
    each parameter value, using seeds that continue the original derivation
    (base_seed + 1000*p_idx + iter_index) so the combined cohort has no seed
    collisions.

    Asserts via the seed.txt artifact written by the runner at iteration start."""
    exp_name = f"extend_seed_seq_{tmp_path.name}"
    exp_dir = repo_root / "runs" / exp_name
    base_seed = 42
    # 3 param values for Z in {0.0, 0.2, 0.4}
    z_dirs = ("Z_0", "Z_20", "Z_40")

    try:
        _initial_sweep_subprocess(exp_name, n_iter=2, base_seed=base_seed,
                                  tiny_subprocess_env=tiny_subprocess_env,
                                  repo_root=repo_root)

        # Verify seed.txt in the initial 2 iterations follows the derivation rule.
        initial_seeds = _read_iter_seeds(exp_dir, z_dirs, iter_indices=(1, 2))
        expected_initial = [
            base_seed + 1000 * p_idx + i
            for p_idx in range(3)
            for i in (0, 1)  # 0-based iter indices for iter_001, iter_002
        ]
        assert initial_seeds == expected_initial, (
            f"sanity: initial seed.txt values wrong: {initial_seeds} vs {expected_initial}"
        )

        # Extend by 3 more iterations.
        subprocess.run(
            [
                "imsyn", "run",
                "--parameter-range", "0.0,0.4", "--step", "0.2", "--n-iterations", "2",
                "--experiment-name", exp_name,
                "--resume", str(exp_dir),
                "--extend", "3",
                "--seed", str(base_seed),
            ],
            env=tiny_subprocess_env,
            check=True,
            timeout=3600,
            cwd=str(repo_root),
        )

        # Verify seed.txt in the 3 new iterations continues the sequence.
        extend_seeds = _read_iter_seeds(exp_dir, z_dirs, iter_indices=(3, 4, 5))
        expected_extend = [
            base_seed + 1000 * p_idx + i
            for p_idx in range(3)
            for i in (2, 3, 4)  # 0-based indices for iter_003, iter_004, iter_005
        ]
        assert extend_seeds == expected_extend, (
            f"extend seed.txt values wrong: {extend_seeds} vs {expected_extend}"
        )

    finally:
        import shutil
        shutil.rmtree(exp_dir, ignore_errors=True)


@pytest.mark.slow
def test_run_comprehensive_extend_uses_recorded_seed_when_seed_arg_not_given(
    tmp_path, tiny_subprocess_env, repo_root
):
    """If progress.json records base_seed (from the initial run), --extend
    can run without the user re-passing --seed and still produces the
    correct continuing seed sequence.

    Asserts via seed.txt: the 3 new iteration directories have seed values
    matching base_seed + 1000*p_idx + iter_idx with the recovered base_seed."""
    exp_name = f"extend_rec_seed_{tmp_path.name}"
    exp_dir = repo_root / "runs" / exp_name
    base_seed = 137
    z_dirs = ("Z_0", "Z_20", "Z_40")

    try:
        _initial_sweep_subprocess(exp_name, n_iter=2, base_seed=base_seed,
                                  tiny_subprocess_env=tiny_subprocess_env,
                                  repo_root=repo_root)

        # Extend WITHOUT passing --seed; the runner must recover it from progress.json.
        subprocess.run(
            [
                "imsyn", "run",
                "--parameter-range", "0.0,0.4", "--step", "0.2", "--n-iterations", "2",
                "--experiment-name", exp_name,
                "--resume", str(exp_dir),
                "--extend", "3",
                # NOTE: no --seed; should be recovered from progress.json
            ],
            env=tiny_subprocess_env,
            check=True,
            timeout=3600,
            cwd=str(repo_root),
        )

        extend_seeds = _read_iter_seeds(exp_dir, z_dirs, iter_indices=(3, 4, 5))
        expected = [
            base_seed + 1000 * p_idx + i
            for p_idx in range(3)
            for i in (2, 3, 4)
        ]
        assert extend_seeds == expected, (
            f"seed.txt values without --seed: {extend_seeds} vs {expected}"
        )

    finally:
        import shutil
        shutil.rmtree(exp_dir, ignore_errors=True)


@pytest.mark.slow
def test_run_comprehensive_extend_falls_back_to_seed_arg_when_progress_lacks_base_seed(
    tmp_path, tiny_subprocess_env, repo_root
):
    """For backward compatibility with progress.json files written before
    base_seed was persisted, --extend accepts the original seed via --seed
    and uses it for the continuing sequence.

    Simulates a legacy progress.json by stripping base_seed after the initial
    run, then verifies seed.txt values in the extended iterations."""
    exp_name = f"extend_legacy_{tmp_path.name}"
    exp_dir = repo_root / "runs" / exp_name
    base_seed = 42
    z_dirs = ("Z_0", "Z_20", "Z_40")

    try:
        _initial_sweep_subprocess(exp_name, n_iter=2, base_seed=base_seed,
                                  tiny_subprocess_env=tiny_subprocess_env,
                                  repo_root=repo_root)

        # Strip base_seed to simulate a legacy progress.json.
        import json
        progress_file = exp_dir / "progress.json"
        progress = json.loads(progress_file.read_text())
        progress.pop("base_seed", None)
        progress_file.write_text(json.dumps(progress))

        # Extend with explicit --seed to compensate for the missing progress field.
        subprocess.run(
            [
                "imsyn", "run",
                "--parameter-range", "0.0,0.4", "--step", "0.2", "--n-iterations", "2",
                "--experiment-name", exp_name,
                "--resume", str(exp_dir),
                "--extend", "3",
                "--seed", str(base_seed),
            ],
            env=tiny_subprocess_env,
            check=True,
            timeout=3600,
            cwd=str(repo_root),
        )

        extend_seeds = _read_iter_seeds(exp_dir, z_dirs, iter_indices=(3, 4, 5))
        expected = [
            base_seed + 1000 * p_idx + i
            for p_idx in range(3)
            for i in (2, 3, 4)
        ]
        assert extend_seeds == expected, (
            f"legacy-progress seed.txt values: {extend_seeds} vs {expected}"
        )

    finally:
        import shutil
        shutil.rmtree(exp_dir, ignore_errors=True)
