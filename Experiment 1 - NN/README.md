# Collocational Bootstrapping: A hypothesis about the learning of subject-verb agreement in humans and neural networks

# Experiment 1

This experiment explores how the variability of subject-verb co-occurrence in training data affects a language model's ability to generalize subject-verb agreement. We train neural network models on synthetic data with variable distributional properties (Zipfian parameter Z) and evaluate performance.

**Note: In the code, we use 'Z' instead of 'α' for ease of implementation.**

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

Sweep Zipfian α from 0 to 3 (step 0.1) with 10 random-seed runs per value, plus the α→∞ oneshot limit case at 10 iterations. Each run trains a 2-layer transformer on 12,000 synthetic sentences and evaluates on minimal pairs across four conditions (seen/unseen subject-verb pairs × matching/mismatching prepositional objects). These conditions are defined by whether the specific subject-verb pairs were present in the training data (SEEN) or not (UNSEEN) and whether the prepositional objects have the same number as the subject noun (MATCH) or not (MISMATCH).

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

Each experiment tests 4 conditions:

- **seen_match**: Seen noun-verb pairs, matching prepositional objects
- **seen_mismatch**: Seen noun-verb pairs, mismatching prepositional objects
- **unseen_match**: Unseen pairs, matching prepositional objects
- **unseen_mismatch**: Unseen pairs, mismatching prepositional objects

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
