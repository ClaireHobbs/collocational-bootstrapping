# Imagining-Syntax Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pare the `Experiment 1 - NN/` codebase down to exactly what reproduces the *Collocational Bootstrapping* paper's main neural-network experiment (Fig. 2) — Zipfian α∈[0,3] sweep + α→∞ oneshot point + Figure 2 plot.

**Architecture:** Eight tasks, each a single committable unit. Tasks 1–5 are pure removals (cloud, plot CLI, --reuse-model, emojis, geometric distribution, autodetect helper); Tasks 6–8 introduce new behavior (plotting module + autogen, flattened CLI, default paper command). Tests are deleted alongside the features they covered, and new system tests are added at the CLI boundary for the new behavior. The README is updated at every task that changes its content; Task 9 is a final README pass.

**Tech Stack:** Python ≥ 3.10, PyTorch 2.7.1, NumPy 2.3.1, pandas 2.3.0, matplotlib 3.10.3, scikit-learn 1.7.0, scipy 1.16.0, PyYAML 6.0.2, tqdm 4.67.1, pytest 8.3.5 + pytest-cov 6.0.0.

**Working directory for all commands:** `/home/jason/coll_boot/Experiment 1 - NN`. Quote it because of the spaces.

**Reference spec:** `docs/superpowers/specs/2026-05-09-imagining-syntax-cleanup-design.md` (in the parent repo, not under Experiment 1 - NN).

---

## File Structure (Post-Cleanup)

```
Experiment 1 - NN/
├── pyproject.toml             # google-cloud-storage dep removed
├── README.md                  # rewritten: no emojis, no cloud/plot/single/geometric content
├── .coveragerc                # cloud_run_job/manage_cloud_jobs/plot_*/etc. omits removed
├── .gitignore                 # unchanged
├── src/imagining_syntax/
│   ├── __init__.py            # unchanged
│   ├── cli.py                 # `run` and `gen` only; default fn = paper command
│   ├── plotting.py            # NEW: plot_zipfian_sweep / plot_oneshot_bars / plot_paper_figure
│   ├── data/
│   │   ├── __init__.py
│   │   ├── distributions.py   # zipfian + oneshot only; geometric & parse_experiment_directory gone
│   │   ├── generate.py        # --oneshot replaces --distribution_type
│   │   ├── minimal_pairs.py   # same
│   │   └── sentences.py       # internal distribution_type kept (zipfian|oneshot)
│   ├── experiment/
│   │   ├── __init__.py
│   │   ├── eval.py            # ✓/✗ stripped
│   │   ├── resume.py          # unchanged
│   │   ├── runner.py          # reuse-model plumbing & geometric branch removed
│   │   └── stats.py           # reuse_model field & "c_value" CSV header → "param_value"
│   ├── model/                 # untouched
│   ├── runners/
│   │   ├── __init__.py
│   │   ├── paper.py           # NEW: full paper experiment + Figure 2
│   │   └── run.py             # renamed from sweep.py; flag-based CLI; --oneshot path
│   └── utils/                 # untouched
└── tests/
    ├── __init__.py
    ├── conftest.py            # unchanged
    ├── _paper_repro_wrapper.py  # uses new flag-based CLI
    ├── fixtures/              # unchanged
    ├── test_dataset_generation.py  # geometric tests removed; --distribution_type → --oneshot
    ├── test_gates.py          # unchanged
    ├── test_minimal_pairs.py  # same as test_dataset_generation
    ├── test_paper_reproduction.py  # uses new flag-based CLI
    ├── test_plotting.py       # NEW: covers plotting.py at the function level
    ├── test_run.py            # renamed from test_runner_comprehensive.py; new CLI shape
    └── test_run_paper.py      # NEW: covers `imsyn` no-args paper command
```

**Removed entirely:**
- `Experiment 1 - NN/cloud/` (top-level Docker dir)
- `Experiment 1 - NN/analysis/` (top-level legacy plot scripts)
- `Experiment 1 - NN/PARALLEL.md`
- `Experiment 1 - NN/test.sh`
- `src/imagining_syntax/cloud/` (whole package)
- `src/imagining_syntax/analysis/` (whole package)
- `src/imagining_syntax/runners/single.py`
- `tests/test_runner_single.py`

---

## Task 1: Remove the Cloud Subsystem

**Goal:** Delete every cloud-specific file/dependency. After this task, no GCS / Cloud Run code remains, the `imsyn cloud` subcommand is gone, and the test suite still passes.

**Files:**
- Delete: `Experiment 1 - NN/cloud/` (whole directory: `Dockerfile`, `Dockerfile.deps`, `Dockerfile.local`, `cloudbuild.app.yaml`, `cloudbuild.deps.yaml`, `build_app_only.sh`, `build_deps_only.sh`)
- Delete: `Experiment 1 - NN/src/imagining_syntax/cloud/` (whole package: `__init__.py`, `download.py`, `manage.py`, `queue.py`, `run_job.py`)
- Delete: `Experiment 1 - NN/PARALLEL.md`
- Delete: `Experiment 1 - NN/test.sh`
- Modify: `Experiment 1 - NN/pyproject.toml` (drop `google-cloud-storage==3.1.1`)
- Modify: `Experiment 1 - NN/src/imagining_syntax/cli.py:28-30` (drop `_add_cloud_group`)
- Modify: `Experiment 1 - NN/.coveragerc` (drop cloud-related `omit` entries)
- Modify: `Experiment 1 - NN/README.md` (drop the "Manual Operation" cloud-related code blocks if any, and remove any cloud links)

- [ ] **Step 1.1: Confirm no live tests cover the cloud subsystem**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
grep -rn "imagining_syntax.cloud\|imsyn cloud" tests/ 2>/dev/null
```

Expected output: empty (the existing test suite does not import or invoke `imagining_syntax.cloud`). If it returns lines, stop and update those tests/notes before proceeding.

- [ ] **Step 1.2: Delete the top-level cloud directory and the cloud package**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
rm -rf cloud/
rm -rf src/imagining_syntax/cloud/
```

- [ ] **Step 1.3: Delete cloud-specific docs/scripts**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
rm PARALLEL.md test.sh
```

- [ ] **Step 1.4: Drop the cloud subparser registration in cli.py**

Open `Experiment 1 - NN/src/imagining_syntax/cli.py`. Replace:

```python
def _add_cloud_group(subparsers: argparse._SubParsersAction) -> None:
    from imagining_syntax.cloud import manage
    manage.add_parser(subparsers)
```

with nothing (delete those four lines). Then in `build_parser()`, delete the call `_add_cloud_group(subparsers)` so the function reads:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imsyn")
    subparsers = parser.add_subparsers(dest="group", metavar="GROUP")
    _add_run_group(subparsers)
    _add_plot_group(subparsers)
    _add_gen_group(subparsers)
    return parser
```

(`_add_plot_group` is still here; it's removed in Task 2.)

- [ ] **Step 1.5: Drop google-cloud-storage from pyproject.toml**

Open `Experiment 1 - NN/pyproject.toml`. In the `dependencies` list, delete the line:

```
    "google-cloud-storage==3.1.1",
```

The remaining list should end with `"matplotlib==3.10.3",` followed by the closing `]`.

- [ ] **Step 1.6: Update .coveragerc**

Open `Experiment 1 - NN/.coveragerc`. Replace the file with:

```ini
[run]
source = .
    src/imagining_syntax
omit =
    venv/*
    tests/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    if __name__ == .__main__.:
    raise NotImplementedError
```

(Drops `analysis/*`, `cloud_run_job.py`, `manage_cloud_jobs.py`, `queue_parallel_jobs.py`, `download_sweep_results.py`, `plot_*.py`, `plot_single_experiment 2.py` — all of these are paths that no longer exist or are deleted in later tasks.)

- [ ] **Step 1.7: Run tests, confirm green**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -x -q
```

Expected: all currently-passing fast tests still pass. (`pytest` defaults to `not slow and not science`, so this is the fast tier only.)

If a test fails because it imports `imagining_syntax.cloud` or invokes `imsyn cloud`, that test was missed in step 1.1 — delete or update it.

- [ ] **Step 1.8: Commit**

```bash
cd "/home/jason/coll_boot"
git add -A "Experiment 1 - NN/"
git commit -m "$(cat <<'EOF'
Remove cloud subsystem from imagining-syntax

Deletes the GCS/Cloud Run integration: top-level cloud/ Docker assets,
src/imagining_syntax/cloud/ package, PARALLEL.md, test.sh, and the
google-cloud-storage dependency. The `imsyn cloud` subcommand is gone.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Remove the Plot CLI and the analysis/ Directories

**Goal:** Delete `imsyn plot`, the top-level `analysis/` legacy scripts, and `src/imagining_syntax/analysis/`. Plotting will be re-introduced as a single new module in Task 6; this task only removes.

**Files:**
- Delete: `Experiment 1 - NN/analysis/` (top-level: 6 .py files)
- Delete: `Experiment 1 - NN/src/imagining_syntax/analysis/` (whole package: `__init__.py`, `single.py`, `sweep.py`, `variance.py`)
- Modify: `Experiment 1 - NN/src/imagining_syntax/cli.py` (drop `_add_plot_group`)
- Modify: `Experiment 1 - NN/README.md` (drop "Visualization" and "imsyn plot ..." sections)

- [ ] **Step 2.1: Confirm no kept code imports from analysis/**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
grep -rn "imagining_syntax.analysis\|imsyn plot" --include="*.py" src/ tests/ 2>/dev/null
```

Expected output: only matches inside `src/imagining_syntax/cli.py` (the `_add_plot_group` import) and inside `src/imagining_syntax/analysis/` itself. No tests should reference `imsyn plot`. If they do, note them — they'll be updated/deleted in step 2.6.

- [ ] **Step 2.2: Delete the top-level analysis/ directory and the analysis package**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
rm -rf analysis/
rm -rf src/imagining_syntax/analysis/
```

- [ ] **Step 2.3: Drop the plot subparser registration in cli.py**

Open `Experiment 1 - NN/src/imagining_syntax/cli.py`. Delete these lines:

```python
def _add_plot_group(subparsers: argparse._SubParsersAction) -> None:
    from imagining_syntax.analysis import single, sweep, variance
    plot = subparsers.add_parser("plot", help="Plot results from a completed run.")
    plot_sub = plot.add_subparsers(dest="plot_subcommand", metavar="SUBCOMMAND")
    single.add_parser(plot_sub)
    sweep.add_parser(plot_sub)
    variance.add_parser(plot_sub)
```

…and the call to `_add_plot_group(subparsers)` inside `build_parser()`. After this step `build_parser()` should look like:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imsyn")
    subparsers = parser.add_subparsers(dest="group", metavar="GROUP")
    _add_run_group(subparsers)
    _add_gen_group(subparsers)
    return parser
```

- [ ] **Step 2.4: Trim README — remove the "Visualization" section**

Open `Experiment 1 - NN/README.md`. Delete this block (currently around lines 60–66):

```markdown
### Visualization

**Auto-detects distribution type:**
```bash
imsyn plot sweep runs/EXPERIMENT_DIR
imsyn plot single runs/EXPERIMENT_DIR
```
```

(The README will be rewritten more thoroughly in Task 9; this step is just to make sure the header doesn't reference a command that no longer exists.)

- [ ] **Step 2.5: Update test_paper_reproduction.py wrapper if it references analysis/**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
grep -n "imagining_syntax.analysis" tests/_paper_repro_wrapper.py tests/test_paper_reproduction.py 2>/dev/null
```

Expected output: empty (the wrapper reads CSV directly via `csv.DictReader`, doesn't go through plotting). If non-empty, remove those imports — the wrapper still works without them.

- [ ] **Step 2.6: Run tests, confirm green**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -x -q
```

Expected: all currently-passing fast tests still pass.

If `imsyn plot ...` is invoked anywhere in tests, the test fails and we delete/update it. Re-run until green.

- [ ] **Step 2.7: Commit**

```bash
cd "/home/jason/coll_boot"
git add -A "Experiment 1 - NN/"
git commit -m "$(cat <<'EOF'
Remove `imsyn plot` and the analysis/ directories

Deletes top-level analysis/ (legacy standalone plot scripts) and the
src/imagining_syntax/analysis/ package, dropping the `imsyn plot`
subcommand entirely. Auto-generated plotting is added back in a later
commit as a single focused module.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Remove `--reuse-model`

**Goal:** Strip the shared-model performance optimization. `runners/sweep.py` no longer takes `--reuse-model`; the `shared_model/` filesystem layout, the `copytree`-into-iter-dirs logic, and the slow test that exercised it all go.

**Files:**
- Modify: `Experiment 1 - NN/src/imagining_syntax/runners/sweep.py` (multiple deletions)
- Modify: `Experiment 1 - NN/src/imagining_syntax/experiment/stats.py:39, 73, 88, 124` (drop the `reuse_model` parameter and field)
- Modify: `Experiment 1 - NN/tests/test_runner_comprehensive.py` (delete the slow `test_comprehensive_reuse_model_creates_shared_model_dir_and_per_iter_models` test)
- Modify: `Experiment 1 - NN/README.md` (remove the "### Model Reuse (Performance Optimization)" subsection, currently around lines 110–113)

- [ ] **Step 3.1: Delete the slow `--reuse-model` test**

In `Experiment 1 - NN/tests/test_runner_comprehensive.py`, delete the entire `test_comprehensive_reuse_model_creates_shared_model_dir_and_per_iter_models` function (currently around lines 108–164, including its `@pytest.mark.slow` decorator).

- [ ] **Step 3.2: Strip reuse_model from runners/sweep.py — argparse flag**

In `Experiment 1 - NN/src/imagining_syntax/runners/sweep.py:307-308`, delete:

```python
    p.add_argument('--reuse-model', action='store_true',
                   help='Train model once per parameter value and reuse for all iterations')
```

- [ ] **Step 3.3: Strip reuse_model from `run_experiments_for_param`**

In `Experiment 1 - NN/src/imagining_syntax/runners/sweep.py`, change the function signature at line 68 from:

```python
def run_experiments_for_param(param_value, distribution_type, n_iterations, experiments_dir, reuse_model=False, eval_types=None, vocab_size=40, unseen_count=10, *, base_seed=None, p_idx=0, start_iter=0, existing_results=None):
```

to:

```python
def run_experiments_for_param(param_value, distribution_type, n_iterations, experiments_dir, eval_types=None, vocab_size=40, unseen_count=10, *, base_seed=None, p_idx=0, start_iter=0, existing_results=None):
```

Then delete:
- The `if reuse_model:` print at lines 93–94
- The entire `Handle shared model if requested.` block at lines 121–159 (everything from `# Handle shared model if requested.` through `shutil.rmtree(temp_data_dir)` plus the surrounding `if reuse_model:` / `else:` and the `shared_model_dir = None` initialization)
- The `shared_model_dir` argument forwarded into `run_single_iteration` at line 169 — change

```python
        result = run_single_iteration(param_value, distribution_type, idx, iteration_dir, shared_model_dir, eval_types, vocab_size, unseen_count, iter_seed=iter_seed)
```

to

```python
        result = run_single_iteration(param_value, distribution_type, idx, iteration_dir, eval_types, vocab_size, unseen_count, iter_seed=iter_seed)
```

- The `reuse_model` argument forwarded into `save_experiment_summary` at line 188 — change

```python
    save_experiment_summary(param_value, distribution_type, total_iters, all_results, statistics,
                           summary_dir, reuse_model)
```

to

```python
    save_experiment_summary(param_value, distribution_type, total_iters, all_results, statistics,
                           summary_dir)
```

- [ ] **Step 3.4: Strip reuse_model from `run_single_iteration`**

In `Experiment 1 - NN/src/imagining_syntax/runners/sweep.py`, change the function signature at line 193 from:

```python
def run_single_iteration(param_value, distribution_type, iteration, iteration_dir, shared_model_dir=None, eval_types=None, vocab_size=40, unseen_count=10, *, iter_seed=None):
```

to:

```python
def run_single_iteration(param_value, distribution_type, iteration, iteration_dir, eval_types=None, vocab_size=40, unseen_count=10, *, iter_seed=None):
```

Update the docstring to drop the `shared_model_dir` line.

Then in the body of the function, delete the `# Create directories, but skip model directory if using shared model` branch and the `if shared_model_dir:` branch (lines 228-231 and 247-250). The two surviving paths simplify to:

```python
    # Create iteration directory structure
    dirs = {
        'base': iteration_dir,
        'data': os.path.join(iteration_dir, 'data'),
        'model': os.path.join(iteration_dir, 'model'),
        'eval': os.path.join(iteration_dir, 'eval'),
        'results': os.path.join(iteration_dir, 'results')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
```

…and the model-train branch becomes unconditional:

```python
    # Step 2: Train new model
    train_args = build_training_args(
        data_dir=dirs['data'],
        model_save_dir=dirs['model'],
        seed=iter_seed,
    )
    train_one(train_args, output_root=iteration_dir, model_subdir='model')
```

(Drop the `# Setup model (train new or copy shared)` heading and the `if shared_model_dir:` branch.)

- [ ] **Step 3.5: Strip reuse_model from `main()`'s callsite**

In `Experiment 1 - NN/src/imagining_syntax/runners/sweep.py:432-435`, drop the print:

```python
    print(f"* Model reuse: {'Yes' if args.reuse_model else 'No'}")
```

In `runners/sweep.py:497-502`, change the call to:

```python
            statistics = run_experiments_for_param(param_value, args.distribution_type, n_iter_to_run,
                                                 dirs['experiments'], args.eval_types,
                                                 args.vocab_size, args.unseen_count,
                                                 base_seed=effective_base_seed, p_idx=i,
                                                 start_iter=start_iter,
                                                 existing_results=existing_results)
```

(removed `args.reuse_model` from the positional list).

In `runners/sweep.py:520-521` and `536-537`, drop `args.reuse_model` from the two calls to `create_comprehensive_summary`. They become:

```python
            create_comprehensive_summary(all_param_values, args.distribution_type, all_statistics, dirs,
                                        args.param_step, comprehensive_n_iterations)
```

and at the bottom of `main()`:

```python
    create_comprehensive_summary(all_param_values, args.distribution_type, all_statistics, dirs,
                                args.param_step, comprehensive_n_iterations)
```

- [ ] **Step 3.6: Strip reuse_model from experiment/stats.py**

In `Experiment 1 - NN/src/imagining_syntax/experiment/stats.py`, change:

- Line 39 — the function signature:

```python
def save_experiment_summary(param_value, distribution_type, n_iterations, all_results, stats, summary_dir, reuse_model):
```

becomes:

```python
def save_experiment_summary(param_value, distribution_type, n_iterations, all_results, stats, summary_dir):
```

Drop the `'reuse_model': reuse_model,` line from the `json_data` dict (currently at line 50).

- Line 73 — the function signature:

```python
def create_comprehensive_summary(param_values, distribution_type, all_statistics, dirs, param_step, n_iterations, reuse_model):
```

becomes:

```python
def create_comprehensive_summary(param_values, distribution_type, all_statistics, dirs, param_step, n_iterations):
```

Drop the `'reuse_model': reuse_model,` line from the `json_data` dict (currently around line 124) and the line:

```python
        f.write(f"Model reuse: {'Yes (shared model per param)' if reuse_model else 'No (separate models)'}\n")
```

(currently at line 88) from the comprehensive_summary.txt writer.

- [ ] **Step 3.7: Trim the README "Model Reuse" subsection**

Open `Experiment 1 - NN/README.md`. Delete this block (currently around lines 110–113):

```markdown
### Model Reuse (Performance Optimization)
```bash
imsyn run sweep 0 1 0.1 5 --distribution_type geometric --reuse-model
```
```

- [ ] **Step 3.8: Run the fast test tier, confirm green**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -x -q
```

Expected: all fast tests pass. The `tests/test_runner_comprehensive.py` argparse-validation tests run quickly and exercise the new signature without `--reuse-model`.

- [ ] **Step 3.9: Commit**

```bash
cd "/home/jason/coll_boot"
git add -A "Experiment 1 - NN/"
git commit -m "$(cat <<'EOF'
Remove --reuse-model from imsyn run sweep

Sharing one trained model across iterations defeats the variance
estimation that the publication's 10 random-seed methodology relies on,
and the paper-reproduction gate doesn't use it. Drops the flag, the
shared_model/ dir, the copytree-into-iter-dirs plumbing, and the slow
test exercising the layout.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Remove Emojis

**Goal:** Strip ✓/✗ from kept Python files and remove emoji headings/icons from README. The cloud emoji prints (Task 1) and analysis emojis (Task 2) are already gone.

**Files:**
- Modify: `Experiment 1 - NN/src/imagining_syntax/experiment/eval.py:62` (✓/✗ in verbose pair output)
- Modify: `Experiment 1 - NN/README.md` (`🔧`, `🔬`, `🚀`, `📊`, `🗂️`, `🧪`, `📈` in section headings)

- [ ] **Step 4.1: Strip ✓/✗ from experiment/eval.py**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
grep -n "✓\|✗" src/imagining_syntax/experiment/eval.py
```

Expected output: one line at `:62`. Open the file and replace:

```python
            result_symbol = "✓" if is_correct else "✗"
```

with:

```python
            result_symbol = "OK" if is_correct else "NO"
```

(The variable is consumed in the `print` block immediately below; both ASCII tokens line up the columns reasonably.)

- [ ] **Step 4.2: Strip emoji headings from README**

In `Experiment 1 - NN/README.md`, perform these exact heading replacements (the headings appear once each):

| Old | New |
| --- | --- |
| `## 🔧 Installation` | `## Installation` |
| `## 🔬 Main Experiment` | `## Main Experiment` |
| `## 🚀 Quick Start` | `## Quick Start` |
| `## 📊 Distribution Types` | `## Distribution Types` |
| `## 🗂️ Directory Structure` | `## Directory Structure` |
| `## 🧪 Evaluation Framework` | `## Evaluation Framework` |
| `## 📈 Advanced Features` | `## Advanced Features` |
| `## 🔧 Manual Operation (Advanced Users)` | `## Manual Operation (Advanced Users)` |

(Other section content is rewritten in later tasks; this step only strips emojis from heading lines.)

- [ ] **Step 4.3: Confirm no other emojis in kept files**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
grep -rn "✓\|✗\|🚀\|🔬\|📊\|🗂️\|🧪\|📈\|🔧" --include="*.py" --include="*.md" src/ tests/ README.md 2>/dev/null
```

Expected output: empty.

- [ ] **Step 4.4: Run tests, confirm green**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -x -q
```

Expected: all fast tests pass. (eval.py's verbose-print path is an internal `verbose=False` default and isn't exercised by the test suite, so this is a defensive check.)

- [ ] **Step 4.5: Commit**

```bash
cd "/home/jason/coll_boot"
git add -A "Experiment 1 - NN/"
git commit -m "$(cat <<'EOF'
Strip emojis from imagining-syntax

Replaces ✓/✗ with OK/NO in the per-pair eval verbose output and
removes emoji icons from README section headings.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Remove the Geometric Distribution and the Autodetect Helper

**Goal:** Drop the geometric distribution everywhere (code, CLI flags, tests, docs) and delete `parse_experiment_directory`. The `--distribution_type {geometric,zipfian,oneshot}` flag is replaced by an `--oneshot` boolean. CSV column `c_value` is renamed to `param_value`.

**Files:**
- Modify: `Experiment 1 - NN/src/imagining_syntax/data/distributions.py` (delete `create_geometric_distribution`, geometric branches, `parse_experiment_directory`)
- Modify: `Experiment 1 - NN/src/imagining_syntax/data/sentences.py` (default distribution_type → "zipfian")
- Modify: `Experiment 1 - NN/src/imagining_syntax/data/generate.py` (replace `--distribution_type` with `--oneshot`)
- Modify: `Experiment 1 - NN/src/imagining_syntax/data/minimal_pairs.py` (same)
- Modify: `Experiment 1 - NN/src/imagining_syntax/runners/sweep.py` (replace `--distribution_type` with `--oneshot`; defaults stay zipfian)
- Modify: `Experiment 1 - NN/src/imagining_syntax/runners/single.py` (replace `--distribution_type` with `--oneshot`)
- Modify: `Experiment 1 - NN/src/imagining_syntax/experiment/runner.py` (drop geometric branch in `prepare_eval_sets`)
- Modify: `Experiment 1 - NN/src/imagining_syntax/experiment/stats.py:137` (CSV header `c_value` → `param_value`)
- Modify: `Experiment 1 - NN/tests/test_dataset_generation.py` (delete geometric tests; switch flags)
- Modify: `Experiment 1 - NN/tests/test_minimal_pairs.py` (same)
- Modify: `Experiment 1 - NN/tests/test_runner_comprehensive.py` (delete geometric tests; switch flags)
- Modify: `Experiment 1 - NN/tests/test_runner_single.py` (delete geometric test; switch flags)
- Modify: `Experiment 1 - NN/tests/_paper_repro_wrapper.py` (drop `--distribution_type zipfian`; defensive `c_value`/`z_value` lookup simplifies)
- Modify: `Experiment 1 - NN/README.md` (rewrite to remove geometric examples and the "Distribution Types" section's geometric paragraph)

> **Note on ordering:** This task is large. We sequence it as: (a) update `data/distributions.py` first, (b) percolate the simpler API through `sentences.py`/`generate.py`/`minimal_pairs.py`, (c) percolate through the runners, (d) update tests last. Each step keeps the import graph compilable so `pytest` runs after each subset.

- [ ] **Step 5.1: Delete `create_geometric_distribution` and rewrite `create_distribution`**

Open `Experiment 1 - NN/src/imagining_syntax/data/distributions.py`. Replace the **entire file** with:

```python
#!/usr/bin/env python3
"""Distribution interface for noun-verb pairing experiments.

Two distributions: zipfian (the publication's main sweep) and oneshot (the
α→∞ limit case where each verb is paired deterministically with one noun)."""

import numpy as np


def create_zipfian_distribution(param_value, vocab_size=None, unseen_count=None):
    """Truncated Zipfian over `vocab_size - unseen_count` seen items, zeros for the
    `unseen_count` unseen tail. param_value is the exponent α (== s in the standard
    Zipfian formula P(k) = 1/k^s); α=0 is uniform over the seen portion."""
    if vocab_size is None:
        vocab_size = 40
    if unseen_count is None:
        unseen_count = 10

    seen_vocab_size = vocab_size - unseen_count
    if param_value == 0:
        prob_per_item = 1.0 / seen_vocab_size
        return [prob_per_item] * seen_vocab_size + [0] * unseen_count

    ranks = np.arange(1, seen_vocab_size + 1)
    unnormalized = 1.0 / (ranks ** param_value)
    seen_probs = unnormalized / np.sum(unnormalized)
    return seen_probs.tolist() + [0] * unseen_count


def create_oneshot_distribution(param_value, vocab_size=None, unseen_count=None):
    """Deterministic 1:1 pairing — all probability mass at offset 0 in the seen
    portion. param_value is ignored (the function takes it for parity with the
    zipfian builder so callers can dispatch uniformly)."""
    del param_value  # explicit: oneshot is parameterless
    if vocab_size is None:
        vocab_size = 40
    if unseen_count is None:
        unseen_count = 10

    seen_vocab_size = vocab_size - unseen_count
    return [1.0] + [0.0] * (seen_vocab_size - 1) + [0.0] * unseen_count


def create_distribution(distribution_type, param_value, vocab_size=None, unseen_count=None):
    """Dispatch to the requested distribution. Two types supported: 'zipfian',
    'oneshot'. Raises ValueError on anything else."""
    if distribution_type.lower() == "zipfian":
        return create_zipfian_distribution(param_value, vocab_size, unseen_count)
    if distribution_type.lower() == "oneshot":
        return create_oneshot_distribution(param_value, vocab_size, unseen_count)
    raise ValueError(
        f"Unsupported distribution type: {distribution_type}. "
        "Supported types: 'zipfian', 'oneshot'"
    )


def validate_parameter(distribution_type, param_value):
    """Raise ValueError if param_value is out of range for the given distribution."""
    if distribution_type.lower() == "zipfian":
        if param_value < 0:
            raise ValueError(
                f"Zipfian distribution parameter must be >= 0, got {param_value}"
            )
        return True
    if distribution_type.lower() == "oneshot":
        return True  # parameterless
    raise ValueError(f"Unsupported distribution type: {distribution_type}")


def get_parameter_name(distribution_type):
    """One-letter token used in directory paths (Z for zipfian, O for oneshot)."""
    if distribution_type.lower() == "zipfian":
        return "Z"
    if distribution_type.lower() == "oneshot":
        return "O"
    raise ValueError(f"Unsupported distribution type: {distribution_type}")


def format_param_for_filename(distribution_type, param_value):
    """Filesystem-safe parameter token. Zipfian: '15' for 1.5, '0' for 0.0,
    '150' for 1.50. Oneshot: always '0'."""
    if distribution_type.lower() == "oneshot":
        return "0"
    if distribution_type.lower() == "zipfian":
        if param_value == int(param_value):
            return str(int(param_value))
        param_str = f"{param_value:.2f}".replace(".", "")
        return param_str.lstrip("0") or "0"
    raise ValueError(f"Unsupported distribution type: {distribution_type}")
```

(That's the full file — `parse_experiment_directory` is gone, `create_geometric_distribution` is gone, and the geometric branches in every other helper are gone.)

- [ ] **Step 5.2: Update `data/sentences.py` default**

In `Experiment 1 - NN/src/imagining_syntax/data/sentences.py:16`, change the function signature default:

```python
def generate_sentence(param_value, distribution_type="geometric", prep_obj_mismatch=False, both_pps_present=False, noun_OOD=False, vocab_size=None, unseen_count=None):
```

to:

```python
def generate_sentence(param_value, distribution_type="zipfian", prep_obj_mismatch=False, both_pps_present=False, noun_OOD=False, vocab_size=None, unseen_count=None):
```

Nothing else in this file changes — `create_distribution` is still imported and called with the explicit `distribution_type` arg.

- [ ] **Step 5.3: Replace `--distribution_type` with `--oneshot` in `data/generate.py`**

In `Experiment 1 - NN/src/imagining_syntax/data/generate.py`:

- In the docstring of `generate_unique_sentences` (line 12+), replace `'geometric' or 'zipfian'` with `'zipfian' or 'oneshot'`. Change the default value of `distribution_type` from `"geometric"` to `"zipfian"`.
- In `_add_args` (lines 59–82), replace the `--distribution_type` argument:

```python
    parser.add_argument('--distribution_type', type=str, default='geometric',
                       choices=['geometric', 'zipfian', 'oneshot'],
                       help='Distribution type: geometric (default), zipfian, or oneshot')
```

with:

```python
    parser.add_argument('--oneshot', action='store_true',
                       help='Use oneshot (deterministic 1:1) distribution instead of zipfian')
```

- In `main()` (line 96+), replace the line:

```python
    all_sentences = generate_unique_sentences(
        args.sentence_count,
        args.param_value,
        args.distribution_type,
        args.vocab_size,
        args.unseen_count
    )
```

with:

```python
    distribution_type = "oneshot" if args.oneshot else "zipfian"
    all_sentences = generate_unique_sentences(
        args.sentence_count,
        args.param_value,
        distribution_type,
        args.vocab_size,
        args.unseen_count
    )
```

…and update the print to use `distribution_type` instead of `args.distribution_type`:

```python
    param_name = get_parameter_name(distribution_type)
    print(f"Generated dataset with {distribution_type} distribution: {param_name}={args.param_value}")
```

- Update the help on the `param_value` positional (line 61) from:

```python
                       help='Distribution parameter value (C for geometric 0-1, Z for Zipfian ≥0)')
```

to:

```python
                       help='Distribution parameter value (Z for zipfian ≥0; ignored when --oneshot is set)')
```

- [ ] **Step 5.4: Same surgery in `data/minimal_pairs.py`**

In `Experiment 1 - NN/src/imagining_syntax/data/minimal_pairs.py`:

- Line 27: change `def generate_minimal_pairs(param_value, distribution_type="geometric"...` → `def generate_minimal_pairs(param_value, distribution_type="zipfian"...`. Update the docstring's `'geometric' or 'zipfian'` → `'zipfian' or 'oneshot'`.

- In `_add_args` (lines 70+), replace the `--distribution_type` argument exactly as in step 5.3.

- In `main()` (line 109+), introduce the `distribution_type` derivation and pass it through:

```python
def main(args):
    """Generate the minimal pairs with parsed args."""
    if args.seed is not None:
        set_global_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    distribution_type = "oneshot" if args.oneshot else "zipfian"
    pairs = generate_minimal_pairs(
        param_value=args.param_value,
        distribution_type=distribution_type,
        num_pairs=args.num_pairs,
        prep_obj_mismatch=args.prep_obj_mismatch,
        both_pps_present=args.both_pps_present,
        noun_OOD=args.noun_OOD,
        vocab_size=args.vocab_size,
        unseen_count=args.unseen_count
    )

    output_path = os.path.join(args.output_dir, args.output_file)
    save_pairs(pairs, output_path)

    param_name = get_parameter_name(distribution_type)
    print(f"Generated {len(pairs)} minimal pairs with {distribution_type} distribution: {param_name}={args.param_value}")
    print(f"Settings: noun_OOD={args.noun_OOD}, prep_obj_mismatch={args.prep_obj_mismatch}")
    print(f"Saved to: {output_path}")
```

- Update the help on the `param_value` positional (line 73) the same way as step 5.3.

- [ ] **Step 5.5: Same surgery in `runners/sweep.py` and `runners/single.py`**

In `Experiment 1 - NN/src/imagining_syntax/runners/sweep.py`:

- Lines 304–306, replace the `--distribution_type` argparse:

```python
    p.add_argument('--distribution_type', type=str, default='geometric',
                   choices=['geometric', 'zipfian', 'oneshot'],
                   help='Distribution type: geometric (default), zipfian, or oneshot')
```

with:

```python
    p.add_argument('--oneshot', action='store_true',
                   help='Run oneshot (deterministic 1:1) instead of zipfian sweep')
```

- In `main()`, add this line near the top (right after the validation block, before the call to `generate_param_values`):

```python
    args.distribution_type = "oneshot" if args.oneshot else "zipfian"
```

(All downstream code in `runners/sweep.py:main` already reads `args.distribution_type`, so this single derivation is enough.)

- Update the help text on `param_min`/`param_max` (lines 296–301) — drop the geometric mentions:

```python
    p.add_argument('param_min', type=float,
                   help='Minimum α value (e.g., 0.0)')
    p.add_argument('param_max', type=float,
                   help='Maximum α value (e.g., 3.0)')
```

In `Experiment 1 - NN/src/imagining_syntax/runners/single.py`:

- Lines 172–174, the same `--distribution_type` → `--oneshot` swap.
- In `main()` (line 181+), add `distribution_type = "oneshot" if args.oneshot else "zipfian"` near the top and use this local everywhere `args.distribution_type` is referenced. Concretely, change:

```python
    distribution_type = args.distribution_type
```

to:

```python
    distribution_type = "oneshot" if args.oneshot else "zipfian"
```

- Lines 226 and 230 in `main()` (the `unseen_param` derivation): the comment `# Test 3: Unseen pairings, objects match subject (use uniform: geometric C=1.0, zipfian Z=0.0)` and the conditional `unseen_param = 1.0 if distribution_type == 'geometric' else 0.0` simplify to:

```python
    # Test 3 & 4: Unseen pairings sample from a uniform distribution
    # (zipfian Z=0.0 gives uniform over the seen vocabulary; oneshot uses Z=0
    # internally for the unseen sample, since oneshot itself doesn't define a
    # range over which to be "uniform").
    unseen_param = 0.0
```

Update the help for the `param_value` positional (line 171) the same way as step 5.3.

- [ ] **Step 5.6: Strip the geometric branch from `experiment/runner.py:prepare_eval_sets`**

In `Experiment 1 - NN/src/imagining_syntax/experiment/runner.py:84`, replace:

```python
    unseen_alpha = 1.0 if distribution_type == "geometric" else 0.0
```

with:

```python
    unseen_alpha = 0.0  # zipfian Z=0 / oneshot ignored param → uniform over seen tokens
```

- [ ] **Step 5.7: Rename `c_value` to `param_value` in stats.py CSV header**

In `Experiment 1 - NN/src/imagining_syntax/experiment/stats.py:137`, replace:

```python
        # Header - keep c_value for backward compatibility with existing plotting scripts
        f.write("c_value,eval_type,mean,std,min,max\n")
```

with:

```python
        f.write("param_value,eval_type,mean,std,min,max\n")
```

- [ ] **Step 5.8: Update `_paper_repro_wrapper.py` to expect `param_value`**

In `Experiment 1 - NN/tests/_paper_repro_wrapper.py:55-66`, simplify the row-extraction block. The current defensive lookup:

```python
                row["alpha"] = float(
                    row.get("z_value")
                    or row.get("c_value")
                    or row.get("param_value")
                    or alpha
                )
```

becomes (we now always emit `param_value`):

```python
                row["alpha"] = float(row.get("param_value") or alpha)
```

Also drop the `--distribution_type zipfian` arg from the subprocess invocation at line 47:

```python
        subprocess.run(
            ["imsyn", "run", "sweep",
             str(alpha), str(alpha), "0.1", str(n_seeds),
             "--vocab-size", str(vocab_size),
             "--unseen-count", str(unseen_count),
             "--seed", str(base_seed),
             "--experiment-name", f"paper_repro_{alpha}"],
            env=env, check=True,
        )
```

(zipfian is now the default since `--distribution_type` is gone).

- [ ] **Step 5.9: Delete geometric-specific tests in test_dataset_generation.py**

Open `Experiment 1 - NN/tests/test_dataset_generation.py`. Delete:

- `test_geometric_dataset_generation_produces_three_split_files` (currently around lines 31–51)
- `test_create_distribution_geometric_default_returns_vocab_sized_list` (currently around lines 131–143)
- `test_validate_parameter_rejects_geometric_out_of_range` (currently around lines 178–181)
- `test_format_param_for_filename` for geometric: there isn't one currently — check the file. There's only `test_format_param_for_filename_zipfian_15_returns_150` and `test_format_param_for_filename_oneshot_always_returns_zero`, both keep.
- `test_parse_experiment_directory_geometric_returns_geometric_and_value`
- `test_parse_experiment_directory_zipfian_returns_zipfian_and_value`
- `test_parse_experiment_directory_oneshot_returns_oneshot_and_zero`
- `test_parse_experiment_directory_malformed_returns_none_tuple`
- `test_get_parameter_name_rejects_unknown_distribution_type` (currently keeps; it's not geometric-specific)

Now switch the flags in the surviving tests:

- `_common_args` helper (line 16): change `"--distribution_type", dist_type,` to:

```python
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
```

- `test_zipfian_dataset_generation_produces_three_split_files` and `test_oneshot_dataset_generation_produces_three_split_files`: keep as-is — they pass `"zipfian"` / `"oneshot"` to the helper which now translates to `--oneshot` or omits the flag.

- `test_same_seed_produces_byte_identical_dataset` and `test_generated_sentences_use_only_known_vocabulary`: change their `_common_args(..., "0.5", "geometric")` calls to `_common_args(..., "1.0", "zipfian")` (zipfian Z=1.0 is a sensible spot in the regime).

- [ ] **Step 5.10: Delete geometric-specific tests in test_minimal_pairs.py**

Open `Experiment 1 - NN/tests/test_minimal_pairs.py`. Update `_run_minimal_pairs` (lines 7–25) the same way as in step 5.9 — drop `--distribution_type` from the cmd list and append `--oneshot` if `dist_type == "oneshot"`. The `dist_type` parameter default `"geometric"` becomes `"zipfian"`.

```python
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
```

The four test functions in this file all use the helper or pass through it; no per-test changes are needed beyond the helper update.

- [ ] **Step 5.11: Delete geometric-specific cases in test_runner_comprehensive.py**

Open `Experiment 1 - NN/tests/test_runner_comprehensive.py`. Update the geometric-typed slow tests to zipfian, and update CLI calls:

- `test_comprehensive_geometric_sweep_produces_csv_with_expected_rows` (line 17+): rename to `test_comprehensive_zipfian_sweep_produces_csv_with_expected_rows`. Change `--distribution_type", "geometric",` → drop. Change the CSV column assertion `required = {"c_value", ...}` → `required = {"param_value", ...}`. Change `param_dirs = list((exp_dir / "experiments").glob("C_*"))` to `glob("Z_*")`. Change `0.0", "0.5", "0.5", "1"` (a 2-α sweep) to `0.0", "1.0", "1.0", "1"` (still 2 α values: 0.0 and 1.0 in zipfian).

- `test_comprehensive_resume_picks_up_where_it_left_off` (line 58+): change `--distribution_type", "geometric"` → drop. Change `0.0", "1.0", "0.5", "1"` (3 α values: 0.0, 0.5, 1.0) to `0.0", "1.0", "0.5", "1"` — same args, but they now mean zipfian. Update the comment about "150-180s per train" if needed.

- `test_run_comprehensive_with_seed_produces_byte_identical_csv` (line 167+): drop `--distribution_type`, use the same numeric args.

- `test_run_comprehensive_rejects_out_of_range_param_value`: currently uses `1.5", "1.5", "0.1", "1", "--distribution_type", "geometric"`. Zipfian doesn't reject 1.5 (Z must be >= 0). Replace with the negative-zipfian case: `"-0.5", "-0.5", "0.1", "1"` (drop the `--distribution_type`).

- All other test-named-`comprehensive` functions that pass `--distribution_type", "zipfian"`: drop that flag pair (zipfian is now default).

Don't rename the file or function names yet — the file rename to `test_run.py` happens in Task 7.

- [ ] **Step 5.12: Update tests/test_runner_single.py**

In `Experiment 1 - NN/tests/test_runner_single.py`:

- `test_run_single_geometric_creates_experiment_dir_with_four_accuracy_files` (line 39+): rename to `test_run_single_zipfian_creates_experiment_dir_with_four_accuracy_files`. Drop `--distribution_type", "geometric"`. Change `*_C_50` glob → `*_Z_15` (Z=1.5 for zipfian) and update the param value from `"0.5"` to `"1.5"`.

- `test_run_single_zipfian_creates_z_prefixed_directory`: drop `--distribution_type", "zipfian"` (now default).

- `test_run_single_with_same_seed_produces_byte_identical_artifacts`: same drop, change C_50 → Z_15 and the param 0.5 → 1.5.

- `test_two_simultaneous_run_single_invocations_create_distinct_experiment_dirs`: same.

- `test_run_single_with_out_of_range_C_exits_non_zero` (line 184+): replace with `test_run_single_with_negative_zipfian_exits_non_zero`. Use `"-0.5"` and drop `--distribution_type`. The validate_parameter rejection for negative zipfian still hits.

This file gets deleted in Task 7 (when `imsyn run single` goes away); the surgery here is minimal — just enough to keep tests green between Task 5 and Task 7.

- [ ] **Step 5.13: Trim the README's "Distribution Types" section**

Open `Experiment 1 - NN/README.md`. Replace the existing "Distribution Types" block (lines 68–81 in the pre-edit file) with:

```markdown
## Distribution Types

### Zipfian Distribution (Parameter α)
- **Range**: α ≥ 0 (corresponds to s in standard Zipf equation: P(k) = 1/k^s)
- **α = 0**: Uniform distribution over the seen vocabulary
- **α = 1**: Classic Zipf's law
- **α > 1**: More concentrated on high-frequency pairs
- **Formula**: P(k) = (1/k^α) / H_N,α

### Oneshot (α → ∞ limit)
- **Parameterless**: every verb is paired with exactly one noun (offset 0)
- **Use**: invoke with `--oneshot` on `imsyn run sweep` or `imsyn gen *`
```

Replace the pre-edit "Quick Start" example (lines 36–58) with:

```markdown
## Quick Start

### Single Experiments

**Zipfian Distribution:**
```bash
imsyn run single 1.5
```

**Oneshot:**
```bash
imsyn run single 0 --oneshot
```

### Comprehensive Experiments

**Zipfian (α from 0 to 3, step 0.1, 5 iterations):**
```bash
imsyn run sweep 0 3 0.1 5
```

**Oneshot (10 iterations):**
```bash
imsyn run sweep 0 0 0.1 10 --oneshot
```
```

(Task 7 will rewrite these once more after the CLI flattens to `imsyn run`. This intermediate edit just removes geometric mentions so the README isn't lying.)

Replace the "Manual Operation (Advanced Users)" section's geometric examples (lines 124–143) with zipfian-only / oneshot variants:

```markdown
## Manual Operation (Advanced Users)

### Dataset Generation
```bash
# Zipfian
imsyn gen dataset 1.0 \
  --output_dir data --train_file train.txt --val_file val.txt --test_file test.txt

# Oneshot
imsyn gen dataset 0 --oneshot \
  --output_dir data --train_file train.txt --val_file val.txt --test_file test.txt
```

### Minimal Pairs Generation
```bash
# Zipfian
imsyn gen pairs 1.0 \
  --output_dir eval --output_file pairs.txt [--noun_OOD] [--prep_obj_mismatch]

# Oneshot
imsyn gen pairs 0 --oneshot \
  --output_dir eval --output_file pairs.txt [--noun_OOD] [--prep_obj_mismatch]
```
```

Also replace the `Main Experiment` example (line 28–32) — drop `--distribution_type zipfian` since it's the default:

```markdown
```bash
imsyn run sweep 0 3 0.1 10 --seed 42
```
```

- [ ] **Step 5.14: Run the fast test tier, confirm green**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -x -q
```

Expected: all fast tests pass. (Slow tests are still skipped by the default marker filter.)

If any test still references `geometric` or `--distribution_type`, that test was missed — track it down and fix.

- [ ] **Step 5.15: Smoke-test the CLI manually**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
imsyn run sweep --help | head -30
imsyn gen dataset --help | head -20
imsyn gen pairs --help | head -20
imsyn run single --help | head -10
```

Expected: all four show `--oneshot` (no `--distribution_type`). No `geometric` anywhere in the help text.

- [ ] **Step 5.16: Commit**

```bash
cd "/home/jason/coll_boot"
git add -A "Experiment 1 - NN/"
git commit -m "$(cat <<'EOF'
Drop geometric distribution and the autodetect helper

The publication's main experiment is zipfian-only, with an α→∞ oneshot
limit case. Removes geometric distribution code, the
parse_experiment_directory autodetect helper, and the user-facing
--distribution_type flag (zipfian becomes default; --oneshot is the
opt-in for the limit case). Renames the comprehensive_results.csv
column from c_value to param_value.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add the Plotting Module + Wire Autogen into the Sweep Runner

**Goal:** Introduce `src/imagining_syntax/plotting.py` with three functions: `plot_zipfian_sweep`, `plot_oneshot_bars`, `plot_paper_figure`. Wire `plot_zipfian_sweep`/`plot_oneshot_bars` into `runners/sweep.py:main` so a plot is always written at the end of `imsyn run sweep`.

**Files:**
- Create: `Experiment 1 - NN/src/imagining_syntax/plotting.py`
- Modify: `Experiment 1 - NN/src/imagining_syntax/runners/sweep.py:main` (call plot at end)
- Create: `Experiment 1 - NN/tests/test_plotting.py`

> **System-test perspective:** the new module is plot-rendering code, which is awkward to system-test through the CLI. We use direct-import tests at the function level to verify file output and basic figure structure (axes, line count, output path); a separate slow integration test in Task 7 will verify the autogen side-effect at the CLI boundary.

- [ ] **Step 6.1: Write failing test — `plot_zipfian_sweep` writes a PNG**

Create `Experiment 1 - NN/tests/test_plotting.py`:

```python
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
```

- [ ] **Step 6.2: Run the new test, verify FAIL**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest tests/test_plotting.py -v
```

Expected: collection-time `ImportError` (`No module named 'imagining_syntax.plotting'`) — the function names don't exist yet.

- [ ] **Step 6.3: Implement `src/imagining_syntax/plotting.py`**

Create `Experiment 1 - NN/src/imagining_syntax/plotting.py`:

```python
"""Plotting for imagining_syntax experiments.

Three entry points:

- plot_zipfian_sweep(csv_path, output_path) — 4-line plot of mean accuracy vs α.
- plot_oneshot_bars(csv_path, output_path) — 4-bar chart of one-α experiment.
- plot_paper_figure(zipfian_csv, oneshot_csv, output_path) — Figure 2 of the
  Collocational Bootstrapping paper: zipfian sweep + α→∞ point with a
  vertical separator and chance-baseline line.

CSVs are the comprehensive_results.csv format produced by
experiment.stats.create_comprehensive_summary: header
`param_value,eval_type,mean,std,min,max`."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITIONS = ("seen_match", "seen_mismatch", "unseen_match", "unseen_mismatch")
COLORS = {
    "seen_match": "#3182bd",
    "seen_mismatch": "#feb24c",
    "unseen_match": "#de2d26",
    "unseen_mismatch": "#756bb1",
}
LABELS = {
    "seen_match": "Seen, Match",
    "seen_mismatch": "Seen, Mismatch",
    "unseen_match": "Unseen, Match",
    "unseen_mismatch": "Unseen, Mismatch",
}
CHANCE_BASELINE = 50.0  # binary-choice minimal-pair baseline


def plot_zipfian_sweep(csv_path, output_path):
    """4-line plot of mean accuracy vs α with ±std bands and chance baseline."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in CONDITIONS:
        subset = df[df["eval_type"] == cond].sort_values("param_value")
        if subset.empty:
            continue
        ax.plot(subset["param_value"], subset["mean"], color=COLORS[cond],
                marker="o", markersize=5, linewidth=2, label=LABELS[cond])
        upper = np.minimum(subset["mean"] + subset["std"], 100)
        lower = np.maximum(subset["mean"] - subset["std"], 0)
        ax.fill_between(subset["param_value"], lower, upper,
                        color=COLORS[cond], alpha=0.2)
    ax.axhline(CHANCE_BASELINE, color="gray", linestyle=":",
               linewidth=1, label="Chance Baseline")
    ax.set_xlabel("α value", fontsize=12)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=12)
    ax.set_title("Model accuracy vs Zipfian α", fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_oneshot_bars(csv_path, output_path):
    """4-bar chart of mean accuracy ±std for a one-parameter (oneshot) run."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    means, stds, colors, labels = [], [], [], []
    for cond in CONDITIONS:
        subset = df[df["eval_type"] == cond]
        if subset.empty:
            continue
        means.append(float(subset["mean"].iloc[0]))
        stds.append(float(subset["std"].iloc[0]))
        colors.append(COLORS[cond])
        labels.append(LABELS[cond])
    xs = np.arange(len(means))
    ax.bar(xs, means, yerr=stds, color=colors, capsize=5, edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Accuracy (%)", fontsize=12)
    ax.set_title("Oneshot accuracy by condition", fontsize=14)
    ax.set_ylim(0, 105)
    ax.axhline(CHANCE_BASELINE, color="gray", linestyle=":", linewidth=1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paper_figure(zipfian_csv, oneshot_csv, output_path):
    """Figure 2 of the Collocational Bootstrapping paper.

    Reads two CSVs and produces a single figure: zipfian sweep on the left
    with the four condition lines + chance baseline, and a separate α→∞
    point on the right (offset past the sweep's α_max) with error bars."""
    z_df = pd.read_csv(zipfian_csv)
    o_df = pd.read_csv(oneshot_csv)

    fig, ax = plt.subplots(figsize=(14, 8))
    z_max = float(z_df["param_value"].max())
    oneshot_x = z_max + 0.4

    for cond in CONDITIONS:
        z_sub = z_df[z_df["eval_type"] == cond].sort_values("param_value")
        if not z_sub.empty:
            ax.plot(z_sub["param_value"], z_sub["mean"], color=COLORS[cond],
                    marker="o", markersize=4, linewidth=2, label=LABELS[cond])
            upper = np.minimum(z_sub["mean"] + z_sub["std"], 100)
            lower = np.maximum(z_sub["mean"] - z_sub["std"], 0)
            ax.fill_between(z_sub["param_value"], lower, upper,
                            color=COLORS[cond], alpha=0.2)
        o_sub = o_df[o_df["eval_type"] == cond]
        if not o_sub.empty:
            mean_v = float(o_sub["mean"].iloc[0])
            std_v = float(o_sub["std"].iloc[0])
            upper_err = min(std_v, 100 - mean_v)
            ax.errorbar(oneshot_x, mean_v,
                        yerr=[[std_v], [upper_err]],
                        marker="D", markersize=10, color=COLORS[cond],
                        capsize=5, capthick=2, linewidth=2, linestyle="none")

    ax.axhline(CHANCE_BASELINE, color="gray", linestyle=":",
               linewidth=1, label="Chance Baseline")
    ax.axvline(z_max + 0.2, color="gray", linestyle="--",
               linewidth=1, alpha=0.5)

    ax.set_xlabel("α value", fontsize=14)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=14)
    ax.set_title("Model accuracy: Zipfian sweep with α→∞ limit", fontsize=15)
    ax.set_ylim(0, 105)
    z_min = float(z_df["param_value"].min())
    ax.set_xlim(z_min - 0.1, oneshot_x + 0.3)

    xticks = list(np.arange(z_min, z_max + 1e-9, 0.5)) + [oneshot_x]
    xticklabels = [f"{v:.1f}" for v in np.arange(z_min, z_max + 1e-9, 0.5)] + ["α→∞"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 6.4: Run the test, verify PASS**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest tests/test_plotting.py -v
```

Expected: 3 passing, 0 failing.

- [ ] **Step 6.5: Wire autogen plot into `runners/sweep.py:main`**

Open `Experiment 1 - NN/src/imagining_syntax/runners/sweep.py`. Add an import near the top, alongside the other `from imagining_syntax...` imports:

```python
from imagining_syntax.plotting import plot_oneshot_bars, plot_zipfian_sweep
```

At the very end of `main()`, just before the closing print of the comprehensive-results banner (currently at lines 539–543: `print(f"\n{'*'*80}")` etc.), insert:

```python
    # Auto-generate plot at the end of the sweep.
    images_dir = os.path.join(dirs['base'], 'images')
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(dirs['summary'], 'comprehensive_results.csv')
    if args.distribution_type == "oneshot":
        plot_path = os.path.join(images_dir, 'oneshot_accuracy.png')
        plot_oneshot_bars(csv_path, plot_path)
    else:
        plot_path = os.path.join(images_dir, 'accuracy_vs_alpha.png')
        plot_zipfian_sweep(csv_path, plot_path)
    print(f"* Plot saved to: {plot_path}")
```

- [ ] **Step 6.6: Smoke-test by mocking the autogen at unit level**

Add this to `tests/test_plotting.py`:

```python
def test_run_sweep_main_wires_autogen_plot(tmp_path, monkeypatch, repo_root):
    """A direct-import drive of runners.sweep.main produces images/*.png at
    the end of the run — verified by monkey-patching out the heavy training
    pipeline so the autogen path runs in <1s."""
    import os
    from argparse import Namespace
    from imagining_syntax.runners import sweep as sweep_mod

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

    monkeypatch.setattr(sweep_mod, "run_experiments_for_param", fake_run_for_param)
    monkeypatch.setattr(sweep_mod, "create_comprehensive_summary", fake_create_summary)

    exp_name = f"plot_smoke_{tmp_path.name}"
    args = Namespace(
        param_min=1.4, param_max=1.4, param_step=0.1, n_iterations=1,
        oneshot=False,
        eval_types=None, vocab_size=40, unseen_count=10,
        seed=None, resume=None, extend=None, experiment_name=exp_name,
    )
    args.distribution_type = "zipfian"
    sweep_mod.main(args)

    plot_path = repo_root / "runs" / exp_name / "images" / "accuracy_vs_alpha.png"
    try:
        assert plot_path.exists(), f"autogen plot not at {plot_path}"
    finally:
        import shutil
        shutil.rmtree(repo_root / "runs" / exp_name, ignore_errors=True)
```

- [ ] **Step 6.7: Run all tests, confirm green**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -x -q
```

Expected: all fast tests including the four in `test_plotting.py` pass.

- [ ] **Step 6.8: Commit**

```bash
cd "/home/jason/coll_boot"
git add -A "Experiment 1 - NN/"
git commit -m "$(cat <<'EOF'
Add plotting module and autogen plot at the end of `imsyn run sweep`

New module src/imagining_syntax/plotting.py provides three entry points:
plot_zipfian_sweep, plot_oneshot_bars, plot_paper_figure. The first two
fire automatically at the end of `imsyn run sweep`, writing the plot
into the experiment dir's images/ subdirectory.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Flatten the CLI — Drop `run single`, Rename `runners/sweep.py` → `runners/run.py`, Switch Positionals to Flags

**Goal:** Collapse the `imsyn run {single,sweep}` two-level split into a single leaf command `imsyn run`. Param range, step, and n_iterations move from positional to optional flags whose defaults match the paper experiment. The single-iteration use-case is `imsyn run --parameter-range 1.4,1.4 --n-iterations 1`.

**Files:**
- Delete: `Experiment 1 - NN/src/imagining_syntax/runners/single.py`
- Delete: `Experiment 1 - NN/tests/test_runner_single.py`
- Rename: `Experiment 1 - NN/src/imagining_syntax/runners/sweep.py` → `Experiment 1 - NN/src/imagining_syntax/runners/run.py`
- Rename: `Experiment 1 - NN/tests/test_runner_comprehensive.py` → `Experiment 1 - NN/tests/test_run.py`
- Modify: `Experiment 1 - NN/src/imagining_syntax/cli.py:_add_run_group` (register `run` as leaf)
- Modify: `Experiment 1 - NN/src/imagining_syntax/runners/run.py:add_parser` and `:main` (no positionals; new flags)
- Modify: `Experiment 1 - NN/tests/test_run.py` (every CLI invocation switches shape)
- Modify: `Experiment 1 - NN/tests/_paper_repro_wrapper.py` (drop `sweep` token; switch to flags)
- Modify: `Experiment 1 - NN/README.md` (rewrite Quick Start, Main Experiment, Manual Operation sections to reflect new CLI)

> **Heads up:** This task touches a lot of test surface. We plan it as: (a) delete `runners/single.py` + tests, (b) rename `runners/sweep.py` → `runners/run.py` + adjust import in `cli.py`, (c) rewrite `runners/run.py:add_parser`/`:main` to drop positionals, (d) port test_runner_comprehensive.py to the new shape and rename. The fast test tier passes only after (d) is done; in the interim we run `pytest --collect-only` to confirm imports resolve.

- [ ] **Step 7.1: Delete `runners/single.py` and `test_runner_single.py`**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
rm src/imagining_syntax/runners/single.py
rm tests/test_runner_single.py
```

- [ ] **Step 7.2: Update `cli.py` to drop the `single` subcommand registration**

Open `Experiment 1 - NN/src/imagining_syntax/cli.py`. The current `_add_run_group` is:

```python
def _add_run_group(subparsers: argparse._SubParsersAction) -> None:
    from imagining_syntax.runners import single, sweep
    run = subparsers.add_parser("run", help="Run an experiment.")
    run_sub = run.add_subparsers(dest="run_subcommand", metavar="SUBCOMMAND")
    single.add_parser(run_sub)
    sweep.add_parser(run_sub)
```

Replace with:

```python
def _add_run_group(subparsers: argparse._SubParsersAction) -> None:
    from imagining_syntax.runners import run as run_module
    run_module.add_parser(subparsers)
```

(Note: now there's a naming collision — the function is `_add_run_group` but the imported module is `run as run_module` to avoid shadowing the builtin and the dispatcher local `run`. The function name stays for parallelism with `_add_gen_group`.)

- [ ] **Step 7.3: Rename `runners/sweep.py` → `runners/run.py`**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
git mv src/imagining_syntax/runners/sweep.py src/imagining_syntax/runners/run.py
```

(Use `git mv` so the rename is recorded as a rename in the diff.)

- [ ] **Step 7.4: Verify imports resolve**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest --collect-only 2>&1 | head -30
```

Expected: collection succeeds. If you see `ModuleNotFoundError: imagining_syntax.runners.sweep`, something else still imports it — search and fix. (The known callers are cli.py, which was updated, and the package `__init__.py` which doesn't import submodules.)

- [ ] **Step 7.5: Restructure `runners/run.py:add_parser` to drop positionals and add flags**

Open `Experiment 1 - NN/src/imagining_syntax/runners/run.py`. Replace the entire `add_parser` function (currently registers a subparser called "sweep" with positionals) with a top-level register:

```python
def add_parser(subparsers):
    """Register the `imsyn run` leaf subparser."""
    p = subparsers.add_parser(
        "run",
        help="Run a Zipfian α sweep (or oneshot) and write the autogen plot.",
        description=(
            "Runs a Zipfian α sweep with paper defaults (α=0–3, step 0.1, "
            "10 iterations per α, seed 42). Override any default with the "
            "flags below. Use --oneshot for the α→∞ limit case."
        ),
    )
    p.add_argument('--parameter-range', type=str, default='0.0,3.0',
                   metavar='MIN,MAX',
                   help='Zipfian α range as MIN,MAX (default: 0.0,3.0). '
                        'Ignored when --oneshot is set.')
    p.add_argument('--step', type=float, default=0.1,
                   help='Step size for the α range (default: 0.1). '
                        'Ignored when --oneshot is set.')
    p.add_argument('--n-iterations', type=int, default=10,
                   help='Number of iterations per α value (default: 10).')
    p.add_argument('--oneshot', action='store_true',
                   help='Use the oneshot (α→∞) limit case instead of zipfian. '
                        '--parameter-range and --step are ignored.')
    p.add_argument('--seed', type=int, default=42,
                   help='Base seed; iter i for param p uses '
                        '(seed + 1000*p_idx + i) (default: 42).')
    p.add_argument('--vocab-size', type=int, default=40,
                   help='Vocabulary size (default: 40).')
    p.add_argument('--unseen-count', type=int, default=10,
                   help='Number of unseen pairs (default: 10).')
    p.add_argument('--eval-types', type=str, nargs='+',
                   choices=['seen_match', 'seen_mismatch',
                            'unseen_match', 'unseen_mismatch'],
                   help='Evaluation types to run (default: all four).')
    p.add_argument('--experiment-name', type=str,
                   help='Override the timestamped runs/ subdirectory name.')
    p.add_argument('--resume', type=str, metavar='DIR',
                   help='Resume from a previous run by experiment directory.')
    p.add_argument('--extend', type=int, metavar='N',
                   help='Extend an existing run with N more iterations.')
    p.set_defaults(func=main)
    return p
```

(Drops the explicit "sweep" subparser name — the parser is registered directly as `run` because cli.py already names the leaf "run".)

Now update `main()`. The current main reads `args.param_min`, `args.param_max`, `args.param_step`, `args.n_iterations`, `args.distribution_type`. Replace the head of `main()` (currently around lines 332–360) with:

```python
def main(args):
    """Run the experiment with parsed args."""
    # Translate --parameter-range and --oneshot into the legacy fields the
    # rest of main() reads.
    args.distribution_type = "oneshot" if args.oneshot else "zipfian"
    if args.oneshot:
        args.param_min = 0.0
        args.param_max = 0.0
        args.param_step = 1.0  # one param value
    else:
        try:
            lo, hi = args.parameter_range.split(",")
            args.param_min = float(lo)
            args.param_max = float(hi)
        except (ValueError, AttributeError):
            print(f"Error: --parameter-range must be MIN,MAX (got {args.parameter_range!r})")
            sys.exit(1)
        args.param_step = args.step

    # Validate arguments based on distribution type
    try:
        validate_parameter(args.distribution_type, args.param_min)
        validate_parameter(args.distribution_type, args.param_max)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.param_min > args.param_max:
        print("Error: parameter-range MIN must be <= MAX")
        sys.exit(1)

    if args.param_step <= 0:
        print("Error: --step must be positive")
        sys.exit(1)

    if args.n_iterations < 1:
        print("Error: --n-iterations must be at least 1")
        sys.exit(1)

    if args.extend is not None:
        if args.extend < 1:
            print("Error: --extend N must be >= 1")
            sys.exit(1)
        if not args.resume:
            print("Error: --extend requires --resume DIR")
            sys.exit(1)

    # ...everything below this line is unchanged from the previous main()
    # (param_values generation, resume handling, the param loop, the autogen
    # plot, the closing banner)
```

Keep the rest of `main()` identical to what was there after Task 6 — only the header (positional → flag translation + validation messages) changes.

- [ ] **Step 7.6: Update `tests/_paper_repro_wrapper.py` to the new CLI**

In `Experiment 1 - NN/tests/_paper_repro_wrapper.py:43-53`, replace:

```python
    for alpha in alphas:
        subprocess.run(
            ["imsyn", "run", "sweep",
             str(alpha), str(alpha), "0.1", str(n_seeds),
             "--vocab-size", str(vocab_size),
             "--unseen-count", str(unseen_count),
             "--seed", str(base_seed),
             "--experiment-name", f"paper_repro_{alpha}"],
            env=env, check=True,
        )
```

with:

```python
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
```

- [ ] **Step 7.7: Rename and rewrite `tests/test_runner_comprehensive.py`**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
git mv tests/test_runner_comprehensive.py tests/test_run.py
```

Now open the renamed file and apply these CLI shape changes throughout:

- All occurrences of `["imsyn", "run", "sweep", ...]` become `["imsyn", "run", ...]` (drop the `"sweep"` token).
- All occurrences of positional `min, max, step, n_iter` become flags:
  - `"0.0", "0.5", "0.5", "1"` → `"--parameter-range", "0.0,0.5", "--step", "0.5", "--n-iterations", "1"`
  - `"0.0", "1.0", "0.5", "1"` → `"--parameter-range", "0.0,1.0", "--step", "0.5", "--n-iterations", "1"`
  - `"0.0", "0.4", "0.2", str(n_iter)` → `"--parameter-range", "0.0,0.4", "--step", "0.2", "--n-iterations", str(n_iter)`
  - …and so on for every other invocation.
- Any `--distribution_type" "zipfian"` (added in step 5.11 if any remain) gets dropped.
- The argparse-validation tests (currently named `test_run_comprehensive_*`):
  - `test_run_comprehensive_rejects_inverted_param_range`: change cmd to `["imsyn", "run", "--parameter-range", "1.0,0.5", "--step", "0.1", "--n-iterations", "1"]`. Update the expected message to `"parameter-range MIN must be <= MAX"`.
  - `test_run_comprehensive_rejects_zero_param_step`: change cmd to `["imsyn", "run", "--parameter-range", "0.0,1.0", "--step", "0", "--n-iterations", "1"]`. Update the expected message to `"--step must be positive"`.
  - `test_run_comprehensive_rejects_out_of_range_param_value`: change cmd to `["imsyn", "run", "--parameter-range", "-0.5,-0.5", "--step", "0.1", "--n-iterations", "1"]` (negative zipfian).
  - `test_run_comprehensive_resume_missing_dir_exits_non_zero`: drop `"sweep"`, switch positional to flags.
  - `test_run_comprehensive_rejects_zero_iterations`: change cmd to `["imsyn", "run", "--parameter-range", "0.0,1.0", "--step", "0.1", "--n-iterations", "0"]`. Update the expected message to `"--n-iterations must be at least 1"`.
  - `test_run_comprehensive_extend_without_resume_dir_rejected`: change cmd to `["imsyn", "run", "--parameter-range", "0.0,0.4", "--step", "0.2", "--n-iterations", "2", "--extend", "3"]`.
  - `test_run_comprehensive_extend_without_seed_anywhere_rejected`: same shape switch. Note that `--seed` now defaults to 42 (not None), so the "needs a base seed" branch in main() requires re-think. The fix is in main(): when `--extend` is set and there's no progress.json base_seed AND the user didn't explicitly pass `--seed`, the runner errors. Since `--seed` now has a default of 42, we need a sentinel: change `--seed` default to `None` for this command (the default 42 was a paper-experiment default that should live in the paper command, not here).
  - **Decision:** revert `--seed` default to `None` in `runners/run.py:add_parser` to match Task 7.5's flag list. The paper-experiment command (Task 8) sets seed=42 explicitly. Update Task 7.5's snippet by changing `--seed` from `default=42` to `default=None`. Re-edit the file accordingly.
  - `test_run_comprehensive_extend_with_zero_count_rejected`: same shape switch + `--seed", "42"` arg.
- The slow tests (`test_comprehensive_*` with `@pytest.mark.slow`): apply the same CLI shape changes. They run only with `pytest -m slow` so they don't gate the fast tier.
- Test function/class **names** can stay as-is (renaming them to `test_run_*` would be cosmetic and adds churn). The file rename is enough.

- [ ] **Step 7.8: Confirm fast tests collect**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest --collect-only -q tests/test_run.py 2>&1 | tail -20
```

Expected: every test function in the renamed file shows up in the collection list with no errors.

- [ ] **Step 7.9: Run fast tests**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -x -q
```

Expected: all fast tests pass. (`test_run.py` has multiple non-slow CLI-validation tests that exercise the new flag shape end-to-end.)

If a test fails because the help text/error message doesn't match, edit the test or the corresponding `print(...)` in `runners/run.py:main` so they line up.

- [ ] **Step 7.10: Smoke-test the new CLI manually**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
imsyn run --help | head -30
imsyn --help | head -10
```

Expected:
- `imsyn run --help` shows flags `--parameter-range`, `--step`, `--n-iterations`, `--oneshot`, `--seed`, etc. NO positional args.
- `imsyn --help` shows the `run` and `gen` groups (no `cloud`, no `plot`).

- [ ] **Step 7.11: Trim README "Main Experiment" / "Quick Start" / "Manual Operation" to use new CLI**

Open `Experiment 1 - NN/README.md`. Replace the existing "Main Experiment" block (line 24+) with:

```markdown
## Main Experiment

Sweep Zipfian α from 0 to 3 (step 0.1) with 10 random-seed runs per value.
Each run trains a 2-layer transformer on 12,000 synthetic sentences and
evaluates on minimal pairs across four conditions (seen/unseen subject-verb
pairs × matching/mismatching prepositional objects), characterizing how
variability in subject-verb pairings shapes the model's ability to learn
agreement.

```bash
imsyn run
```

The `imsyn run` command above is the paper experiment by default — α=0 to
3, step 0.1, 10 iterations. To override:

```bash
imsyn run --parameter-range 0,1 --step 0.5 --n-iterations 5
```

To run the α→∞ (oneshot) limit case:

```bash
imsyn run --oneshot --n-iterations 10
```
```

Replace "Quick Start" (after the Distribution Types section in the post-Task-5 README) with:

```markdown
## Quick Start

The default `imsyn run` invocation is the paper experiment.

**Single point of the α sweep (e.g. α=1.4):**
```bash
imsyn run --parameter-range 1.4,1.4 --step 0.1 --n-iterations 1
```

**Custom α sweep:**
```bash
imsyn run --parameter-range 0,1 --step 0.5 --n-iterations 5
```

**α→∞ (oneshot) limit case:**
```bash
imsyn run --oneshot --n-iterations 10
```
```

(The "Manual Operation" section's `imsyn gen ...` examples stay unchanged from Task 5 — they were already on the new flag shape.)

Drop the "Resume Interrupted Experiments" example block from "Advanced Features" (lines 104–108) — `imsyn run --resume DIR` works the same way (paper defaults plus `--resume`); update if necessary.

- [ ] **Step 7.12: Run fast tests one more time**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -x -q
```

Expected: all fast tests still pass after the README and `runners/run.py` edits.

- [ ] **Step 7.13: Commit**

```bash
cd "/home/jason/coll_boot"
git add -A "Experiment 1 - NN/"
git commit -m "$(cat <<'EOF'
Flatten `imsyn run` CLI: drop run/single, flag-based run

Collapses the run/{sweep,single} two-level split into a single leaf
command `imsyn run`. Param range, step, and n_iterations move from
positional to optional flags whose defaults match the paper experiment
(α=0..3, step 0.1, 10 iterations). The single-iteration use-case is
`imsyn run --parameter-range 1.4,1.4 --n-iterations 1`. Renames
runners/sweep.py → runners/run.py and the corresponding test file.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Add the Default `imsyn` (No Args) = Paper Experiment + Figure 2

**Goal:** When `imsyn` is invoked with no arguments, run the full paper experiment — zipfian sweep + oneshot — and write Figure 2 to disk. Add a system test for the layout.

**Files:**
- Create: `Experiment 1 - NN/src/imagining_syntax/runners/paper.py`
- Modify: `Experiment 1 - NN/src/imagining_syntax/cli.py:main` (set the default fn)
- Create: `Experiment 1 - NN/tests/test_run_paper.py`

> **Implementation note:** the paper command can shell out to `imsyn run` twice (once zipfian, once oneshot) — that's the most reliable path because it goes through the same CLI surface used for testing. Sub-runs land in `runs/paper_<timestamp>/zipfian/` and `runs/paper_<timestamp>/oneshot/`; the combined plot goes to `runs/paper_<timestamp>/figure_2.png`.

- [ ] **Step 8.1: Write the failing system test**

Create `Experiment 1 - NN/tests/test_run_paper.py`:

```python
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
```

- [ ] **Step 8.2: Confirm the test currently fails**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -m slow -x tests/test_run_paper.py -v 2>&1 | tail -20
```

Expected: FAIL — `imsyn` with no args currently prints help and exits 0, so no `paper_*` dir is created.

- [ ] **Step 8.3: Implement `runners/paper.py`**

Create `Experiment 1 - NN/src/imagining_syntax/runners/paper.py`:

```python
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
```

- [ ] **Step 8.4: Wire it into cli.py as the default**

Open `Experiment 1 - NN/src/imagining_syntax/cli.py`. Update `main()` so that when no group is selected, it dispatches to `paper.main(args)` instead of printing help:

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "group", None):
        from imagining_syntax.runners import paper
        return paper.main(args) or 0
    func = getattr(args, "func", None)
    if func is None:
        # Group selected but no subcommand wired up yet (or none provided).
        print(f"imsyn {args.group}: no subcommands available yet")
        return 0
    return func(args) or 0
```

(The change is the `if not getattr(args, "group", None):` branch — was `parser.print_help()`, now imports and calls paper.main.)

- [ ] **Step 8.5: Run the slow paper test, verify PASS**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -m slow -x tests/test_run_paper.py -v
```

Expected: PASS in ~3–8 minutes (one zipfian iter at α=0.0, α=1.4 + one oneshot iter; each train is ~150–180s on CPU). The runs/paper_*/figure_2.png is asserted to exist.

If the test times out at 2400s, increase the timeout or further shrink the bundled fast-test defaults in `_bundled_defaults`.

- [ ] **Step 8.6: Run the fast tier, confirm green**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -x -q
```

Expected: all fast tests still pass. The new `tests/test_run_paper.py` is slow-marked so it's excluded.

- [ ] **Step 8.7: Commit**

```bash
cd "/home/jason/coll_boot"
git add -A "Experiment 1 - NN/"
git commit -m "$(cat <<'EOF'
Add default `imsyn` command: paper experiment + Figure 2

Invoking imsyn with no args now runs the full paper experiment —
zipfian α=0–3 sweep, then α→∞ oneshot, both at 10 iterations seed 42 —
and writes Figure 2 to runs/paper_<timestamp>/figure_2.png. Honors
IMSYN_PAPER_FAST_TEST=1 for a sub-10-minute smoke test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Final README Pass + Paper-Reproduction Test Verification

**Goal:** Single coherent README rewrite covering everything that has shifted, then run the paper-reproduction test (the `science` gate) to confirm the science still locks in.

**Files:**
- Modify: `Experiment 1 - NN/README.md` (full rewrite)
- Modify: `Experiment 1 - NN/Experiment 1 - NN/to_do.md` (mark items done? or delete?) — see below

- [ ] **Step 9.1: Inspect current README state**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
wc -l README.md && head -25 README.md
```

This shows the post-Task-7 README. Tasks 4, 5, 7 have been editing it incrementally; this step does a coherent rewrite.

- [ ] **Step 9.2: Replace README.md wholesale**

Open `Experiment 1 - NN/README.md` and replace its full content with:

```markdown
# Imagining Syntax: Collocational Bootstrapping Reproduction

A computational linguistics research project studying how transformer models
learn subject-verb agreement when subject-verb pairings follow a Zipfian
distribution of varying concentration. This repository reproduces the main
neural-network experiment (Figure 2) of *Collocational Bootstrapping*.

## Installation

Requires Python ≥ 3.10. From a fresh clone:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

`-e` installs the package in editable mode so source edits take effect
without reinstalling. `[dev]` pulls in `pytest` for the test suite. After
install, `imsyn` is on your PATH and `pytest` runs the fast-tier system
tests.

To verify:

```bash
imsyn --help
pytest
```

## Reproducing the Paper

Run the paper experiment — Zipfian α=0–3 sweep (step 0.1, 10 iterations
per α, seed 42), plus the α→∞ oneshot limit case at 10 iterations — and
generate Figure 2:

```bash
imsyn
```

Output lands in `runs/paper_<timestamp>/`:

```
runs/paper_20260509_143052/
├── zipfian/                  # the α=0..3 sweep
│   ├── experiments/Z_*/...
│   ├── summary/comprehensive_results.csv
│   └── images/accuracy_vs_alpha.png
├── oneshot/                  # the α→∞ limit case
│   ├── experiments/O_0/...
│   ├── summary/comprehensive_results.csv
│   └── images/oneshot_accuracy.png
└── figure_2.png              # the combined plot
```

The full run takes hours on CPU. CUDA is supported automatically when
available.

## Custom Experiments

`imsyn run` runs the same machinery without the bundled paper defaults,
exposing every hyperparameter as a flag:

```bash
imsyn run \
  --parameter-range 0.0,1.0 \
  --step 0.5 \
  --n-iterations 5 \
  --seed 42
```

A single-α invocation (the old `run single` use-case) is just a
zero-width range:

```bash
imsyn run --parameter-range 1.4,1.4 --step 0.1 --n-iterations 1
```

The α→∞ (oneshot) limit case:

```bash
imsyn run --oneshot --n-iterations 10
```

Other flags: `--vocab-size`, `--unseen-count`, `--eval-types`,
`--experiment-name`, `--resume DIR`, `--extend N`. See `imsyn run --help`.

## Distribution Types

### Zipfian Distribution (parameter α)

- α ≥ 0; corresponds to s in standard Zipf P(k) = 1/k^s
- α = 0: uniform over the seen vocabulary
- α = 1: classic Zipf's law
- α > 1: more concentrated on high-frequency pairs

### Oneshot (α → ∞ limit)

- Parameterless: every verb is paired with exactly one noun (offset 0)
- Invoke with `--oneshot` on `imsyn run` or `imsyn gen *`

## Directory Structure

```
runs/
├── 20260509_145306_Z_0-3_step0.1_n10/  # zipfian sweep
├── 20260509_145306_O_0/                # oneshot
└── paper_20260509_143052/              # default `imsyn` paper experiment
```

## Evaluation Framework

Each experiment iteration tests four conditions:

| Condition | Description |
| --- | --- |
| `seen_match` | Seen subject-verb pairs, matching prepositional objects |
| `seen_mismatch` | Seen pairs, mismatching prep objects |
| `unseen_match` | Unseen pairs, matching prep objects |
| `unseen_mismatch` | Unseen pairs, mismatching prep objects (the headline finding) |

The publication's main result: `unseen_mismatch` traces an inverted-U with
peak accuracy at α≈1.4. `seen_match` stays ≈100% across all α.

## Manual Operation (Advanced Users)

### Dataset Generation

```bash
# Zipfian
imsyn gen dataset 1.0 \
  --output_dir data --train_file train.txt --val_file val.txt --test_file test.txt

# Oneshot
imsyn gen dataset 0 --oneshot \
  --output_dir data --train_file train.txt --val_file val.txt --test_file test.txt
```

### Minimal Pairs Generation

```bash
imsyn gen pairs 1.0 \
  --output_dir eval --output_file pairs.txt [--noun_OOD] [--prep_obj_mismatch]

imsyn gen pairs 0 --oneshot \
  --output_dir eval --output_file pairs.txt [--noun_OOD] [--prep_obj_mismatch]
```

### Model Training (direct)

```bash
python3 -m imagining_syntax.model.run_training \
  --data_dir data --model_save_dir model
```

### Model Evaluation (direct)

```bash
python3 -m imagining_syntax.experiment.eval \
  --model_dir model --input_file pairs.txt --output_file results.txt
```

## Sentence Generation

The PCFG (`imagining_syntax.data.sentences`) creates sentences with this
structure:
- Basic: `[the] [subject_noun] [verb]`
- With PPs: `[prep] [the] [object] [the] [subject_noun] [prep] [the] [object] [verb]`

When `both_pps_present=False` (the default for training data):
- Each of two possible prepositional phrases has a 50% independent chance
  of being included
- 25% chance of 0 PPs (3-word sentences); 50% chance of 1 PP (6-word);
  25% chance of 2 PPs (9-word)

In practice, sentences without PPs are rare in generated training datasets
because the PCFG enforces uniqueness against a 12,000-sentence target — so
the 40-or-so possible 3-word sentences saturate quickly.
```

- [ ] **Step 9.3: Verify the to_do.md is now obsolete**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
cat to_do.md
```

The to-do items 3, 6, 7, 8, 10, 11, 12 should now be addressed by the
preceding tasks. Decide whether to:
- (a) Delete `to_do.md` entirely (cleanup is done; the file's purpose is gone)
- (b) Leave it alone (the project's working file)

Recommendation: **delete it**, since this plan exists in the parent repo as
the durable record. The user can override.

```bash
rm to_do.md
```

(If the user wants to keep it, skip this command.)

- [ ] **Step 9.4: Run the full fast-tier test suite**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -x -q
```

Expected: all fast tests pass. (Slow tests are skipped by default; the
`science` marker is excluded by the conftest's marker-filter logic.)

- [ ] **Step 9.5: (Optional but recommended) Run the paper-reproduction gate**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -m science -q tests/test_paper_reproduction.py 2>&1 | tail -10
```

Expected: PASS in ~15–25 minutes. This is the science-locking test that
asserts the inverted-U at α≈1.4 still emerges.

If it fails because of a baseline drift (`paper_baseline.json` doesn't
exist yet or values differ ±5pp), inspect — a real regression would also
show up as the inverted-U gap dropping below 15pp. The first successful
run captures the baseline file, which gets committed in the next step.

- [ ] **Step 9.6: Run the slow tier**

```bash
cd "/home/jason/coll_boot/Experiment 1 - NN"
pytest -m slow -q 2>&1 | tail -20
```

Expected: all slow tests pass (this includes `test_run_paper.py` from
Task 8 plus the `test_run.py` resume / extend / determinism tests).

- [ ] **Step 9.7: Commit**

```bash
cd "/home/jason/coll_boot"
git add -A "Experiment 1 - NN/"
git commit -m "$(cat <<'EOF'
Rewrite README and clear to_do.md after cleanup

Single coherent README pass covering: paper-reproduction default,
flag-based custom runs, oneshot, manual generation, and the four-condition
evaluation framework. Removes the now-completed to_do.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist (run after writing the plan)

- [x] **Spec coverage:** Every item in the spec maps to a task:
  - Item 11 (cloud + analysis removal) → Tasks 1 + 2
  - Item 10 (--reuse-model) → Task 3
  - Item 8 (emojis) → Task 4
  - Item 3 (geometric) → Task 5
  - Item 6 (autodetect) → Task 5 (combined; `parse_experiment_directory` deletion in step 5.1)
  - Item 7 (autogen plot) → Task 6
  - Item 12 (default command + paper plot + α→∞) → Tasks 7 + 8
  - README updates → Tasks 2, 3, 4, 5, 7, 9
- [x] **Placeholder scan:** No "TBD", "TODO", "implement later", or "fill in details". Every step has the actual content the engineer needs.
- [x] **Type consistency:** `plot_zipfian_sweep`, `plot_oneshot_bars`, `plot_paper_figure` signatures match between Task 6 (definition) and Tasks 7/8 (callers). `--parameter-range`, `--step`, `--n-iterations`, `--oneshot` are introduced in Task 5/7 and used consistently in Tasks 7/8/9.
- [x] **CSV column rename consistency:** `param_value` is set in Task 5 and read by tests in Task 7 and the plotting module in Task 6. `_paper_repro_wrapper.py` is updated to read `param_value` in Task 5.
