"""Paper-reproduction regression gate.

The Collocational Bootstrapping paper's headline finding (Fig 2): on a Zipfian-α
sweep, UNSEEN_MISMATCH accuracy traces an inverted U with a peak at α ≈ 1.4.
SEEN_MATCH stays ≈100% across all α.

This test runs a 3-point sweep ({0.0, 1.4, 3.0}, 2 seeds each, full vocab,
1200 iters) and asserts the inverted-U shape. ~15-25 min on a 1660 Ti.

Default-suite usage: SKIPPED (marked science).
Explicit invocation: pytest -m science
"""
import json
from pathlib import Path

import pytest

from tests._paper_repro_wrapper import aggregate, experiment_dir, run_three_alphas

BASELINE_PATH = Path(__file__).parent / "fixtures" / "paper_baseline.json"
ALPHAS = (0.0, 1.4, 3.0)


@pytest.mark.science
def test_zipfian_sweep_reproduces_unseen_mismatch_inverted_u_at_alpha_1p4(
    tmp_path, tiny_subprocess_env, repo_root
):
    """The paper's headline finding (Fig 2): UNSEEN_MISMATCH peaks at α≈1.4
    by ≥15 percentage points over both α=0 and α=3; SEEN_MATCH stays ≥95%
    everywhere. Locks in the science, not just the plumbing."""
    try:
        rows = run_three_alphas(
            out_root=tmp_path,
            alphas=list(ALPHAS),
            n_seeds=2,
            repo_root=repo_root,
            env=tiny_subprocess_env,
        )
        means = aggregate(rows)

        unseen_mm = {a: means[(a, "unseen_mismatch")] for a in ALPHAS}
        seen_m = {a: means[(a, "seen_match")] for a in ALPHAS}

        assert unseen_mm[1.4] - unseen_mm[0.0] >= 0.15, (
            f"sweet spot vs low α: {unseen_mm}"
        )
        assert unseen_mm[1.4] - unseen_mm[3.0] >= 0.15, (
            f"sweet spot vs high α: {unseen_mm}"
        )
        assert all(v >= 0.95 for v in seen_m.values()), (
            f"SEEN_MATCH collapsed: {seen_m}"
        )

        if BASELINE_PATH.exists():
            baseline = json.loads(BASELINE_PATH.read_text())
            for key, expected in baseline.items():
                alpha, cond = key.split("/", 1)
                actual = means[(float(alpha), cond)]
                assert abs(actual - expected) <= 0.05, (
                    f"{key}: drift from baseline {expected:.3f} -> {actual:.3f} "
                    f"(±5pp tolerance)"
                )
        else:
            # First successful run: capture the baseline so subsequent runs can
            # enforce the ±5pp tolerance check above.
            BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            baseline = {f"{alpha}/{cond}": v for (alpha, cond), v in means.items()}
            BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True))
    finally:
        # Clean up the runner-output dirs that live under repo_root/.
        # The wrapper's `imsyn run sweep` writes to
        # repo_root/runs/paper_repro_<alpha>/.
        import shutil
        for alpha in ALPHAS:
            shutil.rmtree(experiment_dir(repo_root, alpha), ignore_errors=True)
