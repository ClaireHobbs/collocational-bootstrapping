"""Helper invoked by test_paper_reproduction.py.

`imsyn run sweep` expects (min, max, step, n_iterations) and sweeps uniformly.
The paper's headline check needs three specific α values: {0.0, 1.4, 3.0}.
This wrapper invokes the runner three times and merges the three resulting
CSVs into a single table.

The runner writes outputs to <cwd>/runs/paper_repro_<alpha>/.
Pytest is expected to run from repo_root, so cleanup at
repo_root/runs/paper_repro_<alpha>/ matches. Each alpha
uses a unique --experiment-name so the three sweeps don't collide. The caller
(test_paper_reproduction.py) is responsible for cleaning up those dirs after
collecting results."""
import csv
import subprocess
from pathlib import Path


def experiment_dir(repo_root, alpha):
    """Path the runner writes to for a given alpha (under repo_root)."""
    return Path(repo_root) / "runs" / f"paper_repro_{alpha}"


def run_three_alphas(out_root, alphas, n_seeds, repo_root, env, *,
                    max_iter=1200, vocab_size=40, unseen_count=10, base_seed=42):
    """Run `imsyn run sweep` once per alpha; return merged rows.

    Each invocation uses a unique --experiment-name. The runner writes to
    repo_root/runs/paper_repro_<alpha>/. The caller is
    responsible for cleaning up those directories.

    base_seed is forwarded to the runner's --seed flag; the runner derives
    per-iteration seeds (base_seed + i for i in range(n_seeds)). Fixing the
    seed is what makes this gate deterministic — without it, run-to-run
    variance can flip the inverted-U gap assertion at its tight threshold.

    out_root is unused by the wrapper itself but is kept in the signature for
    backward-compatible callers and as a place tests may use for unrelated
    side-data.
    """
    del out_root  # unused; output goes to repo_root/runs/
    all_rows = []
    for alpha in alphas:
        subprocess.run(
            ["imsyn", "run",
             "--parameter-range", f"{alpha},{alpha}",
             "--step", "0.1",
             "--n-iterations", str(n_seeds),
             "--vocab-size", str(vocab_size),
             "--unseen-count", str(unseen_count),
             "--seed", str(base_seed),
             "--experiment-name", f"paper_repro_{alpha}"],
            env=env, check=True,
        )
        exp_dir = experiment_dir(repo_root, alpha)
        csv_path = next(exp_dir.rglob("comprehensive_results.csv"))
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                row["alpha"] = float(row.get("param_value") or alpha)
                all_rows.append(row)
    return all_rows


def aggregate(rows):
    """Group rows by (alpha, eval_type); compute mean accuracy.

    The runner emits accuracy values as percentages (e.g., 95.40), so we divide
    by 100 to return fractions in [0, 1]."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        key = (row["alpha"], row["eval_type"])
        # Accuracy column may be 'mean' (percent) — runner CSV stores per-(C, eval_type)
        # mean/std/min/max. Use 'mean' divided by 100 to convert percent → fraction.
        acc = float(row.get("mean", row.get("accuracy", 0.0))) / 100.0
        groups[key].append(acc)
    return {k: sum(v) / len(v) for k, v in groups.items()}
