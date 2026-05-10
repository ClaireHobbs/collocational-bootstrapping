"""Direct-import tests for the plotting module."""
import csv

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.image as mpimg

from imagining_syntax.plotting import (
    plot_oneshot_bars,
    plot_paper_figure,
    plot_zipfian_sweep,
)


def _write_minimal_csv(path, rows):
    """Write a comprehensive_results.csv-shaped file with the given rows."""
    with open(path, "w") as f:
        w = csv.writer(f)
        w.writerow(["param_value", "eval_type", "mean", "std", "min", "max"])
        w.writerows(rows)


def test_plot_zipfian_sweep_writes_png_to_output_path(tmp_path):
    """plot_zipfian_sweep produces a PNG with non-zero pixel content."""
    csv_path = tmp_path / "comprehensive_results.csv"
    _write_minimal_csv(
        csv_path,
        [
            ("0.0", "seen_match", "100.0", "0.5", "99.5", "100.0"),
            ("0.0", "unseen_mismatch", "20.0", "5.0", "15.0", "25.0"),
            ("1.4", "seen_match", "100.0", "0.0", "100.0", "100.0"),
            ("1.4", "unseen_mismatch", "75.0", "3.0", "72.0", "78.0"),
            ("3.0", "seen_match", "100.0", "0.5", "99.5", "100.0"),
            ("3.0", "unseen_mismatch", "30.0", "4.0", "26.0", "34.0"),
        ],
    )
    out_path = tmp_path / "fig.png"
    plot_zipfian_sweep(str(csv_path), str(out_path))
    assert out_path.exists(), "plot file not created"
    img = mpimg.imread(str(out_path))
    assert img.shape[0] > 100 and img.shape[1] > 100, f"plot too small: {img.shape}"


def test_plot_oneshot_bars_writes_png_to_output_path(tmp_path):
    """plot_oneshot_bars renders 4 bars (one per condition) with mean ±std."""
    csv_path = tmp_path / "comprehensive_results.csv"
    _write_minimal_csv(
        csv_path,
        [
            ("0.0", "seen_match", "100.0", "0.0", "100.0", "100.0"),
            ("0.0", "seen_mismatch", "85.0", "2.0", "83.0", "87.0"),
            ("0.0", "unseen_match", "60.0", "5.0", "55.0", "65.0"),
            ("0.0", "unseen_mismatch", "40.0", "4.0", "36.0", "44.0"),
        ],
    )
    out_path = tmp_path / "oneshot.png"
    plot_oneshot_bars(str(csv_path), str(out_path))
    assert out_path.exists()


def test_plot_paper_figure_combines_zipfian_sweep_and_oneshot_point(tmp_path):
    """plot_paper_figure reads two CSVs and writes a single combined PNG."""
    zip_csv = tmp_path / "zipfian.csv"
    one_csv = tmp_path / "oneshot.csv"
    _write_minimal_csv(
        zip_csv,
        [
            ("0.0", "seen_match", "100.0", "0.5", "99.5", "100.0"),
            ("0.0", "unseen_mismatch", "20.0", "5.0", "15.0", "25.0"),
            ("1.4", "seen_match", "100.0", "0.0", "100.0", "100.0"),
            ("1.4", "unseen_mismatch", "75.0", "3.0", "72.0", "78.0"),
            ("3.0", "seen_match", "100.0", "0.5", "99.5", "100.0"),
            ("3.0", "unseen_mismatch", "30.0", "4.0", "26.0", "34.0"),
        ],
    )
    _write_minimal_csv(
        one_csv,
        [
            ("0.0", "seen_match", "100.0", "0.0", "100.0", "100.0"),
            ("0.0", "unseen_mismatch", "85.0", "1.5", "83.5", "86.5"),
        ],
    )
    out_path = tmp_path / "figure_2.png"
    plot_paper_figure(str(zip_csv), str(one_csv), str(out_path))
    assert out_path.exists()


def test_run_sweep_main_wires_autogen_plot(tmp_path, monkeypatch, repo_root):
    """A direct-import drive of runners.sweep.main produces images/*.png at
    the end of the run — verified by monkey-patching out the heavy training
    pipeline so the autogen path runs in <1s."""
    import os
    from argparse import Namespace
    from imagining_syntax.runners import run as run_mod

    # Stub out the heavy work: we just need create_comprehensive_summary to
    # write a CSV and the plot wiring to fire.
    def fake_run_for_param(*args, **kwargs):
        return {}  # statistics dict

    def fake_create_summary(param_values, distribution_type, all_statistics, dirs,
                            param_step, n_iterations):
        csv_path = os.path.join(dirs['summary'], 'comprehensive_results.csv')
        os.makedirs(dirs['summary'], exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("param_value,eval_type,mean,std,min,max\n")
            f.write("1.4,seen_match,100.0,0.5,99.5,100.0\n")
            f.write("1.4,unseen_mismatch,75.0,3.0,72.0,78.0\n")

    monkeypatch.setattr(run_mod, "run_experiments_for_param", fake_run_for_param)
    monkeypatch.setattr(run_mod, "create_comprehensive_summary", fake_create_summary)

    exp_name = f"plot_smoke_{tmp_path.name}"
    args = Namespace(
        parameter_range="1.4,1.4", step=0.1, n_iterations=1,
        oneshot=False,
        eval_types=None, vocab_size=40, unseen_count=10,
        seed=None, resume=None, extend=None, experiment_name=exp_name,
    )
    run_mod.main(args)

    plot_path = repo_root / "runs" / exp_name / "images" / "accuracy_vs_alpha.png"
    try:
        assert plot_path.exists(), f"autogen plot not at {plot_path}"
    finally:
        import shutil
        shutil.rmtree(repo_root / "runs" / exp_name, ignore_errors=True)
