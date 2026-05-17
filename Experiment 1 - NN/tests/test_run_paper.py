"""Slow system test: `imsyn` (no args) runs the paper experiment + Figure 2."""
import shutil
import subprocess

import pytest


@pytest.mark.slow
def test_imsyn_no_args_runs_paper_experiment_and_writes_figure_2(
    tmp_path, tiny_subprocess_env, repo_root, monkeypatch
):
    """`imsyn` with no args produces runs/paper_*/figure_2.png plus the two
    sub-run summary CSVs.

    To keep this test under 10 minutes we monkey-patch the paper command's
    bundled defaults via env var IMSYN_PAPER_FAST_TEST=1, which the runner
    reads to substitute a 3-α sweep × 1 iter and oneshot × 1 iter."""
    env = {**tiny_subprocess_env, "IMSYN_PAPER_FAST_TEST": "1"}

    pre_existing = set(
        d.name for d in (repo_root / "runs").glob("paper_*")
    ) if (repo_root / "runs").exists() else set()

    subprocess.run(["imsyn"], env=env, check=True, timeout=2400, cwd=str(repo_root))

    new_dirs = [
        d for d in (repo_root / "runs").glob("paper_*")
        if d.name not in pre_existing
    ]
    assert len(new_dirs) == 1, f"expected exactly one new paper_* dir, got {new_dirs}"
    paper_dir = new_dirs[0]

    try:
        zipfian_csv = paper_dir / "zipfian" / "summary" / "comprehensive_results.csv"
        oneshot_csv = paper_dir / "oneshot" / "summary" / "comprehensive_results.csv"
        figure = paper_dir / "figure_2.png"
        assert zipfian_csv.exists(), f"missing {zipfian_csv}"
        assert oneshot_csv.exists(), f"missing {oneshot_csv}"
        assert figure.exists(), f"missing {figure}"
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)
