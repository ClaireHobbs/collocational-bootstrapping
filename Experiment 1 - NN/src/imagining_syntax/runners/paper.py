"""Default `imsyn` command: run the full paper experiment and write Figure 2.

This module is the entrypoint for `imsyn` invoked with no arguments. It
runs the zipfian sweep and the oneshot (α→∞) limit case as two child
invocations of `imsyn run`, then composes Figure 2 from their two CSVs.

The bundled defaults match the publication: α=0–3 in 0.1 steps, 10
iterations per α, seed 42. To allow a fast smoke test, setting the
environment variable IMSYN_PAPER_FAST_TEST=1 substitutes a 3-α × 1-iter
sweep + 1-iter oneshot."""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from imagining_syntax.plotting import plot_paper_figure


def _bundled_defaults():
    """Return (param_range, step, n_iterations) for the paper experiment.

    Honors IMSYN_PAPER_FAST_TEST=1 to shorten the run for tests."""
    if os.environ.get("IMSYN_PAPER_FAST_TEST") == "1":
        return ("0.0,1.4", 1.4, 1)
    return ("0.0,3.0", 0.1, 10)


def _run_subexperiment(label, args, paper_dir, env):
    """Invoke `imsyn run` as a child process; output goes under
    paper_dir/<label>/. Returns the path to comprehensive_results.csv."""
    name = f"paper_{paper_dir.name}_{label}"
    cmd = ["imsyn", "run", *args, "--experiment-name", name]
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)
    sub_dir = Path("runs") / name
    target = paper_dir / label
    if target.exists():
        # Robust to re-runs in the unlikely paper_dir-already-exists case.
        import shutil
        shutil.rmtree(target)
    sub_dir.rename(target)
    return target / "summary" / "comprehensive_results.csv"


def main(args):
    """Run the paper experiment + write Figure 2."""
    del args  # the default command takes no arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_dir = Path("runs") / f"paper_{timestamp}"
    paper_dir.mkdir(parents=True, exist_ok=True)

    param_range, step, n_iter = _bundled_defaults()

    env = os.environ.copy()
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = ""

    print(f"Paper experiment output: {paper_dir}")
    zip_csv = _run_subexperiment(
        "zipfian",
        ["--parameter-range", param_range, "--step", str(step),
         "--n-iterations", str(n_iter), "--seed", "42"],
        paper_dir, env,
    )
    one_csv = _run_subexperiment(
        "oneshot",
        ["--oneshot", "--n-iterations", str(n_iter), "--seed", "42"],
        paper_dir, env,
    )

    figure_path = paper_dir / "figure_2.png"
    plot_paper_figure(str(zip_csv), str(one_csv), str(figure_path))
    print(f"\n*** Paper experiment complete. Figure 2: {figure_path} ***")
    return 0
