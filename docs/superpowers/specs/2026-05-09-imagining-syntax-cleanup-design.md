# Imagining-Syntax Cleanup: Design

## Goal

Pare the `Experiment 1 - NN/` codebase down to exactly what reproduces the
*Collocational Bootstrapping* paper's main neural-network experiment (Fig. 2):
a Zipfian α sweep from 0.0 to 3.0 plus an α→∞ (oneshot) point, ten random-seed
runs each, four evaluation conditions, with a single auto-generated plot.

Concretely, this knocks out the seven items in `Experiment 1 - NN/to_do.md`.

## Final CLI Surface

```text
imsyn                                        # = run + run --oneshot + Figure 2
imsyn run                                    # zipfian sweep, paper defaults
imsyn run --parameter-range 1.4,1.4 \
          --step 0.1 --n-iterations 1        # single-point equivalent of old `run single 1.4`
imsyn run --parameter-range 0.0,1.0 \
          --step 0.1 --n-iterations 5        # custom zipfian sweep
imsyn run --oneshot                          # oneshot, paper-default n_iterations
imsyn run --oneshot --n-iterations 20        # custom oneshot count
imsyn gen dataset <param> ...                # manual zipfian dataset
imsyn gen dataset --oneshot ...              # manual oneshot dataset
imsyn gen pairs <param> ...                  # manual zipfian pairs
imsyn gen pairs --oneshot ...                # manual oneshot pairs
```

**Removed from the CLI:**
- The `run`/`sweep`/`single` two-level split — `imsyn run` is the sweep
  command; the old `run single PARAM` use-case is `imsyn run
  --parameter-range PARAM,PARAM --n-iterations 1`. The implementation file
  `src/imagining_syntax/runners/sweep.py` is renamed to `runners/run.py`
  and `runners/single.py` is deleted.
- `imsyn plot` group entirely
- `imsyn cloud` group entirely
- `--distribution_type {geometric,zipfian,oneshot}` flag everywhere (replaced
  by `--oneshot` boolean)
- `--reuse-model` flag

## Design Choices

### Default command (`imsyn` no args)

Runs the full paper experiment + writes Figure 2:

1. Zipfian sweep α=0.0..3.0 step 0.1, 10 iterations per α, seed 42.
2. Oneshot, 10 iterations, seed 42.
3. Combined plot to disk.

Layout:
```
runs/paper_<timestamp>/
├── zipfian/              # sweep output: experiments/, summary/, progress.json
├── oneshot/              # oneshot output: experiments/, summary/, progress.json
└── figure_2.png
```

No "skip if results already exist" — each invocation always runs from scratch
under a fresh timestamped path. Cancellation is the user's responsibility.

### `imsyn run` (no positionals, no subcommand)

The `run` parser becomes a leaf command. Param range, step, and n_iterations
move from required positional arguments to optional flags whose defaults
match the paper:

| Flag | Default | Effect |
| --- | --- | --- |
| `--parameter-range MIN,MAX` | `0.0,3.0` | Zipfian α range |
| `--step` | `0.1` | Zipfian α step |
| `--n-iterations` | `10` | Iterations per α |
| `--seed` | `42` | Base seed |
| `--oneshot` | off | Switch to oneshot mode |
| `--vocab-size` | `40` | Vocabulary size |
| `--unseen-count` | `10` | Unseen pair count |
| `--resume DIR` | — | Resume an interrupted run |
| `--extend N` | — | Add N more iterations |
| `--experiment-name NAME` | — | Override timestamp dir |

`--oneshot` ignores `--parameter-range` and `--step`, runs `n_iterations`
oneshot trainings under one synthetic param value (formatted `O_0` in the
filesystem). Internally the sweep machinery handles both — same iteration
loop, same per-iteration seed derivation, same CSV/summary writers — just
swapping the distribution generator.

### Oneshot retention

Kept as the underlying mechanism for the α→∞ point on Figure 2. The
`create_oneshot_distribution` builder in `data/distributions.py` stays;
the user-facing surface is the `--oneshot` flag on `run` and `gen *`.

Oneshot fixture-generation in `data/sentences.py` and `data/minimal_pairs.py`
flows through the same path as zipfian — only the distribution shape differs.

### Geometric removal (item 3)

Drop everything geometric from code and docs:
- `create_geometric_distribution` and the geometric branches in
  `create_distribution`, `validate_parameter`, `get_parameter_name`,
  `format_param_for_filename`, `parse_experiment_directory`.
- The `--distribution_type` flag from every CLI parser.
- The `geometric C ...` examples from README, `PARALLEL.md` (which goes away
  anyway), and module docstrings.
- The "C value" naming convention. The CSV column stays `c_value` for back-compat
  *only* if existing test fixtures need it; otherwise rename to `param_value`.
  (Decision deferred to the implementation plan after grepping CSV consumers.)
- The `unseen_param = 1.0 if distribution_type == 'geometric' else 0.0`
  fallback in `experiment/runner.py:prepare_eval_sets` (now just hardcodes
  the zipfian Z=0 / oneshot value).
- All geometric-specific tests in `tests/test_dataset_generation.py`,
  `tests/test_runner_comprehensive.py`, `tests/test_runner_single.py`.

### Plot autogeneration (item 7) and autodetect removal (item 6)

Plotting moves to a single new module: `src/imagining_syntax/plotting.py`.
It exports two functions:

- `plot_zipfian_sweep(csv_path, output_path)` — 4-line plot, one per
  evaluation condition, with ±std bands. Used by `imsyn run` (no oneshot).
- `plot_paper_figure(zipfian_csv, oneshot_csv, output_path)` — Figure 2:
  zipfian sweep + α→∞ point with a vertical separator, plus the chance
  baseline. Used by the default command.

Both functions take explicit distribution context as parameters — no
directory-name parsing, no JSON metadata sniffing. The autodetect helper
`parse_experiment_directory` in `data/distributions.py` is deleted.

Auto-firing: at the end of `runners/run.py:main` (the `imsyn run`
implementation), after the comprehensive summary is written:
- Zipfian mode: `plot_zipfian_sweep(csv, images/accuracy_vs_alpha.png)`.
- Oneshot mode: a 4-bar chart (one bar per condition, mean ±std across the
  N iterations) → `images/oneshot_accuracy.png`. Implemented as a third
  function `plot_oneshot_bars(csv_path, output_path)` in `plotting.py`.

The default command in `runners/paper.py:main` handles its own combined plot
(`plot_paper_figure`) after both sub-experiments complete; the per-sub-run
autogen plots also fire as a side-effect, which is fine.

### Cloud + analysis removal (item 11)

Delete:
- Top-level `cloud/` directory (Docker, Cloud Build files, build scripts)
- Top-level `analysis/` directory (six .py scripts that aren't on the path
  to Figure 2)
- `src/imagining_syntax/cloud/` package entirely (`download.py`,
  `manage.py`, `queue.py`, `run_job.py`, `__init__.py`)
- `src/imagining_syntax/analysis/` package entirely (`single.py`,
  `sweep.py`, `variance.py`, `__init__.py`)
- `PARALLEL.md` (cloud-specific docs)
- `test.sh` (calls deprecated `manage_cloud_jobs.py`)

Drop `google-cloud-storage` from `pyproject.toml`. Drop the `cloud` and
`plot` groups from `cli.py`.

### Reuse-model removal (item 10)

Strip:
- The `reuse_model` parameter from `runners/run.py:run_experiments_for_param`
  (formerly `runners/sweep.py`) and the surrounding shared-model plumbing
  (`shared_model/` dir, `train_one` call into shared subdir, `shutil.copytree`
  of model files into per-iteration dirs).
- The `--reuse-model` CLI flag.
- The `reuse_model` field from `experiment/stats.py` summary writers (drops
  one column from comprehensive_summary.txt).
- The slow test `test_comprehensive_reuse_model_creates_shared_model_dir_and_per_iter_models`.

### Emoji removal (item 8)

Strip ✓/✗ characters and 🔧/🚀/🔬/etc. from:
- `src/imagining_syntax/experiment/eval.py` (✓/✗ for per-pair correctness
  prints — replace with `"OK"`/`"NO"` or just drop the verbose line).
- `README.md` headings — replace with plain text.
- (`src/imagining_syntax/cloud/manage.py` and `src/imagining_syntax/cloud/queue.py`
  have ✓/✗ but those files are deleted under item 11 anyway.)

## Architecture Map (Post-Cleanup)

```
src/imagining_syntax/
├── __init__.py
├── cli.py                     # adds default fn; only `run` and `gen` groups remain
├── plotting.py                # NEW: Figure 2 + sweep plot
├── data/
│   ├── distributions.py       # zipfian + oneshot; geometric and parse_experiment_directory removed
│   ├── generate.py            # --oneshot flag instead of --distribution_type
│   ├── minimal_pairs.py       # --oneshot flag instead of --distribution_type
│   └── sentences.py           # internal distribution_type kept (zipfian|oneshot)
├── experiment/
│   ├── eval.py                # emojis stripped
│   ├── resume.py              # unchanged
│   ├── runner.py              # distribution_type kept internally; reuse-model plumbing gone
│   └── stats.py               # reuse_model field removed; column rename if needed
├── model/                     # untouched
├── runners/
│   ├── paper.py               # NEW: default command for `imsyn`
│   └── run.py                 # renamed from sweep.py; paper defaults, --oneshot,
│                              # no positional args, no --reuse-model. single.py deleted.
└── utils/                     # untouched
```

Top-level (Experiment 1 - NN/) keeps:
- `pyproject.toml` (no google-cloud-storage)
- `README.md` (rewritten)
- `tests/`, `src/`, `.coveragerc`, `.gitignore`

## Test Strategy

The existing test suite is mostly system tests at the CLI boundary. The
cleanup is mostly *removal* — the system-test-driven-development skill says
"system tests for added code"; we'll add new system tests when we add new
code (the default-command paper run, the new sweep CLI shape, the autogen
plot side-effect). Removed code's tests get deleted alongside.

### Tests to add (new behavior)

1. `test_imsyn_no_args_runs_paper_experiment_and_writes_figure_2` — slow
   marker. Smoke test: `subprocess.run(["imsyn"])` exits 0 and produces
   `runs/paper_*/figure_2.png` plus the two sub-run summaries.
2. `test_run_no_args_uses_paper_defaults` — slow. Run `imsyn run` without
   positional args, verify the produced CSV has rows for every α in
   {0.0, 0.1, ..., 3.0} (use `--n-iterations 1` to keep test runtime down).
3. `test_run_oneshot_writes_oneshot_directory_layout` — slow. Run
   `imsyn run --oneshot --n-iterations 2` and assert the
   `experiments/O_0/iterations/iter_001` and `iter_002` dirs exist with
   accuracy artifacts.
4. `test_autogen_plot_written_to_experiment_dir` — slow. After `imsyn run
   --n-iterations 1 --parameter-range 0.0,0.5 --step 0.5`, assert a PNG is
   present under the experiment dir's `images/`.
5. `test_gen_dataset_oneshot_flag_works` — fast (already exists with
   `oneshot` distribution_type; rewrite to use `--oneshot` flag).

### Tests to delete (removed behavior)

- `tests/test_runner_single.py` (whole file: `imsyn run single` is gone;
  the few shape assertions worth keeping are folded into the new
  `test_run_*` tests above)
- `test_comprehensive_reuse_model_creates_shared_model_dir_and_per_iter_models`
- All `--distribution_type geometric` test cases in
  `test_dataset_generation.py`, `test_minimal_pairs.py`,
  `test_runner_comprehensive.py`
- `test_create_distribution_geometric_default_returns_vocab_sized_list`
- `test_validate_parameter_rejects_geometric_out_of_range`
- `test_format_param_for_filename` for geometric
- `test_parse_experiment_directory_*` (helper deleted)

### Tests to update (CLI shape change)

- `test_run_comprehensive_*` in `test_runner_comprehensive.py`: switch
  positional args to flags, change `imsyn run sweep ...` invocations to
  `imsyn run ...`. Argparse-validation tests for "param_min must be <=
  param_max" etc. need to drive the new `--parameter-range MIN,MAX` parser
  (validation moves into a parser callback or post-parse check). Also
  rename the file to `test_runner.py` since "comprehensive" was the
  contrast with "single".
- `test_run_comprehensive_extend_*` slow tests: same — flags instead of
  positionals, drop `--distribution_type zipfian`.
- `tests/_paper_repro_wrapper.py` and `test_paper_reproduction.py`: drop
  `--distribution_type zipfian`, switch to `--parameter-range α,α
  --step 0.1 --n-iterations <n>` shape (or whatever single-α invocation
  looks like in the new CLI).
- All "imsyn gen dataset" / "imsyn gen pairs" tests: switch from
  `--distribution_type {geometric,zipfian,oneshot}` to just an optional
  `--oneshot` flag (zipfian default).

### Existing tests to preserve as-is

- `test_gates.py` — model-determinism gates, unaffected by the cleanup.
- `test_paper_reproduction.py` — the science gate. Adapts to new CLI but
  the assertions stay (inverted-U at α≈1.4, ±5pp from baseline).

## Out of Scope

- Performance work on training (the science gate uses the existing 1200-iter
  config; we don't optimize it).
- Behavior changes to the model itself (`model/transformer.py` is untouched).
- New analysis features beyond Figure 2 (variance plots, distribution-curve
  plots from the deleted `analysis/` dir are not re-added).
- CHILDES corpus analysis (Experiment 2 in the paper) — not in this repo.

## Implementation Order

The to-do items are roughly ordered by independence; the implementation plan
will sequence them so each commit can be tested:

1. **Item 11** (cloud + analysis removal) — biggest delete, no behavior
   change to remaining code; loosens dependencies.
2. **Item 10** (--reuse-model removal) — independent of others.
3. **Item 8** (emoji removal) — cosmetic, touches only kept files.
4. **Item 3 + 6** (geometric + autodetect removal) — combined because they
   touch the same modules.
5. **Item 7** (autogen plot + plotting module) — depends on 6 (no autodetect).
6. **Item 12** (default command + paper experiment) — depends on 7,
   requires the new CLI shape (flags-not-positionals on `imsyn run`,
   merger of `run`/`single`/`sweep` into a single leaf command).

Each step keeps the test suite green at the system-test level. Slow tests
run on the final step.
