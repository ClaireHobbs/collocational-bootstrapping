# Experiment 2 — CHILDES Zipfian Analysis of Subject-Verb Pairs

## Overview

This experiment estimates the Zipfian parameter α that describes the frequency
distribution of subject-verb pairs in child-directed speech from the CHILDES 
database. The analysis is run twice: once on the full dataset (producing a 
single overall α for 0 to 96 months) and once per age group (producing α for
each of eight 12-month age bins from 0 to 96 months).

## Rank-average denominator correction

The original code used a different averaging denominator from the one
specified in the paper. This correction implements the paper's average and
changes the empirical distribution supplied to the MSE fit.

[Equation 2 of the paper](https://arxiv.org/html/2605.20529v1#S5.SS2) specifies:

$$
f_{\mathrm{paper}}(r) = \frac{1}{100}\sum_{v=1}^{100}
\frac{\mathrm{count}(s_{r,v},v)}{\mathrm{count}(v)}.
$$

Here, $s_{r,v}$ is the subject at rank $r$ for verb $v$. Each term is a
within-verb subject proportion, which we can write as $p_v(r)$. Counting
unobserved pairings as zero gives every selected verb the same weight
in the average at every rank. The denominator is always 100.

The code built rows only for observed subject–verb pairings and aggregated
`average_proportion=('proportion', 'mean')` within each rank. If $K_v$ is the
number of subjects observed for verb $v$ and $n_r$ is the number of selected
verbs with $K_v \ge r$, it calculated:

$$
f_{\mathrm{code}}(r) = \frac{1}{n_r}
\sum_{v:K_v\ge r} p_v(r).
$$

Thus a verb with only one observed subject drops out of the denominator at
rank 2. Higher ranks typically have fewer contributing verbs, so their
averages are inflated by a factor of $100/n_r$ relative to Equation 2.
Before MSE fitting, the code normalizes this curve:

$$
f_{\mathrm{fit}}(r) =
\frac{f_{\mathrm{code}}(r)}{\sum_j f_{\mathrm{code}}(j)}.
$$

That normalization moves relative mass from the head toward the inflated
tail, which can lower the fitted Zipf exponent.

For a concrete example, suppose `run` has two equally frequent subjects and
`jump` has one. Using two verbs in place of 100:

| Subject rank | `run` proportion | `jump` proportion | Original average | Paper's average |
|---|---:|---:|---:|---:|
| 1 | 0.50 | 1.00 | (0.50 + 1.00) / 2 = 0.75 | (0.50 + 1.00) / 2 = 0.75 |
| 2 | 0.50 | Unobserved | 0.50 / 1 = 0.50 | (0.50 + 0) / 2 = 0.25 |

The original curve `(0.75, 0.50)` sums to 1.25 and is normalized to
`(0.60, 0.40)` for fitting. The paper's calculation gives `(0.75, 0.25)`,
which already sums to one. The MSE grid fits these curves with α = 0.58
and α = 1.58, respectively. The denominator choice alone accounts for that
difference in this example.

Both analysis scripts now accumulate `proportion_sum` at each rank, then
explicitly calculate `average_proportion = proportion_sum / len(top_verbs)`.
This implements zero-filled averaging across the full selected verb set.

Refitting all saved pairs from the original **2021.1 dataset** gives the
following values. The overall population contains
4,739,189 utterances and 2,802,071 subject–verb pairs.

| Age, months | Original α | Corrected α |
|---|---:|---:|
| Overall | 1.43 | 1.48 |
| 0–12 | 1.46 | 1.58 |
| 12–24 | 1.40 | 1.49 |
| 24–36 | 1.44 | 1.51 |
| 36–48 | 1.38 | 1.47 |
| 48–60 | 1.37 | 1.47 |
| 60–72 | 1.28 | 1.46 |
| 72–84 | 1.23 | 1.51 |
| 84–96 | 1.25 | 1.52 |

The youngest-to-oldest difference shrinks from −0.21 to −0.06 and the pronounced
oldest-bin downturn disappears.

**Why the distortion becomes stronger in the older bins.** The 24–36-month
bin contains 1,163,974 pairs, compared with 52,385 and 46,807 in the two oldest
bins. Fewer observations of a verb mean fewer distinct subjects are seen.
More verbs therefore drop out of the original average at higher ranks,
reducing $n_r$ and increasing the $100/n_r$ inflation. At rank 30, for example,
95 verbs contribute in the 24–36-month bin, compared with 39 and 33 in the
two oldest bins. The smallest nonzero proportion is also $1/N_v$, where
$N_v$ is the number of observations of that verb: a singleton contributes
a larger proportion when its verb has been observed less often.

**Single observations have more influence for sparsely observed verbs.**
Adding one occurrence of a subject whose current proportion is $p$ changes
that proportion by $(1-p)/(N_v+1)$. For example, adding one occurrence to a
subject seen 5 times in 10 changes its proportion from 0.50 to 6/11 ≈ 0.5455.
For a subject seen 500 times in 1,000, the corresponding change is from 0.50
to 501/1001 ≈ 0.5005. If the added occurrence introduces a new subject, it
also introduces a new rank for that verb and changes that rank's denominator
in the original calculation. Normalizing the resulting curve propagates
the change across all fitted ranks.

Equal verb weighting is part of the paper's intended calculation and applies
to the corrected algorithm too. The corrected average gives every selected
verb weight $1/100$ at every rank, including zero contributions. The original
average gives each verb observed at that rank weight $1/n_r$. In both cases,
the weight is independent of the verb's observation count, so proportions
from sparsely observed verbs can strongly affect the average. This explains
sensitivity to individual observations in both algorithms. The specific
denominator error is excluding missing ranks from the average, which
selectively inflates the tail.

MSE fits the averaged curve by squaring absolute probability errors at each
rank. Changes in the relatively large
head probabilities can therefore have substantial influence on the fit.
A single observation can move the fitted α in either direction; the
systematic downward pressure comes from inflated tails and the resulting
reduction in head mass during normalization.

This sample-size effect is directly observable with age held fixed.
Across 100 random subsamples of 50,000 pairs from the 24–36-month bin,
the original estimator falls from the full-data α of 1.44 to a mean of
1.3262. The zero-filled estimator changes from 1.51 to a mean of 1.4995.
This demonstrates a sample-size contribution to the apparent age pattern.

The regression tests exercise the worked example and equal-support
distributions in both analysis scripts. Run them from this experiment directory:

```sh
python -m unittest discover -s tests -v
```

## Pipeline

Data preparation runs in R (uses the `childesr` package). Analysis runs in Python 
(uses spaCy for dependency parsing). The handoff between the two halves is a large 
CSV file of utterances (~612 MB) saved to `data/childes_utterances.csv`. The file
is too big to open in Excel or Numbers. If you need to inspect it, we suggest using 
a terminal command (head, wc) or loading it in pandas.

## Prerequisites

### R

- R 4.x
- Packages: `childesr`, `dplyr`

Install in R:

```r
install.packages(c("childesr", "dplyr"))
```

### Python

- Python 3.10 or later
- Packages: `pandas`, `numpy`, `spacy`, `matplotlib`
- spaCy English model: `en_core_web_sm`

Install (after creating a virtualenv if desired):

```sh
pip install pandas numpy spacy matplotlib
python -m spacy download en_core_web_sm
```

## How to run

Run the scripts from the `Experiment 2 - CHILDES` folder so that the
relative paths in each script resolve correctly.

### 1. Data preparation (R)

Run the three R scripts in order. Each script creates its own output
directory (`rdata/speakers` or `rdata/utterances`) if it does not already
exist. Each script writes a timestamped CSV; the next script finds the
most recent file automatically.

```sh
Rscript childes_get_adult_speakers.R
Rscript childes_filter_speakers.R
Rscript childes_get_utterances.R
```

What each one does:

- `childes_get_adult_speakers.R` — downloads all English-categorized
  participants from CHILDES who are not target children.
- `childes_filter_speakers.R` — restricts the participant list to 25
  caregiver/adult speaker roles (Adult, Caretaker, Mother, Father,
  Investigator, etc.). Also applies a name-based filter to exclude
  corpora with "Biling" in the corpus name (see "Notes on data scope"
  below for limitations).
- `childes_get_utterances.R` — downloads every utterance for the
  filtered speaker list, aggregates tokens into full utterances, and
  saves a 13-column CSV to `rdata/utterances/`. It also copies the
  same file to `data/childes_utterances.csv` so the Python pipeline
  can find it without an extra step.

Expected runtime: the first two scripts complete in a minute or two;
`childes_get_utterances.R` typically takes 30-45 minutes because it
iterates over every unique corpus-role combination.

### 2. Confirm the data is in place

After step 1 finishes, `data/childes_utterances.csv` should exist
(the third R script copies it there automatically). The Python
analyses read from this path. If for any reason the file is not
there, copy it manually from the most recent file in
`rdata/utterances/`.

### 3. Analysis (Python)

Run the two analysis scripts. Each one creates its own output directory
under `output/`.

```sh
python analyze_complete_dataset_96mos.py
python analyze_age_groups_96mos.py
```

What each one does:

- `analyze_complete_dataset_96mos.py` — filters to utterances with
  target child age ≤ 96 months, extracts subject-verb pairs with spaCy,
  ranks subjects within each of the top 100 verbs, averages across
  verbs, and finds the α that minimizes MSE against a theoretical Zipf
  distribution. Produces a single overall α.
- `analyze_age_groups_96mos.py` — same procedure run separately for
  eight 12-month age bins from 0 to 96 months. Produces one α per bin
  and a publication-style plot of α-by-age.

Expected runtime: each script takes 35-40 minutes (spaCy parsing
dominates).

### 4. Optional supporting scripts

- `generate_sample_table_v2.py` — produces a sample-utterance table
  (CSV and styled PNG) showing five representative utterances from each
  of the eight age groups. Runs in seconds; no spaCy parsing.
- `regenerate_96mos_plot.py` — re-renders the α-by-age plot from the
  existing summary CSV produced by `analyze_age_groups_96mos.py`.
  Useful for tweaking plot styling without re-running spaCy.

Both read from existing output files and run in seconds.

## Outputs

- `rdata/speakers/` — speaker lists (timestamped CSVs from steps 1
  and 2 of the R pipeline).
- `rdata/utterances/` — full utterances CSV from step 3 of the R
  pipeline.
- `data/childes_utterances.csv` — the CSV the Python scripts read
  from. Same content as the most recent file in `rdata/utterances/`.
- `output/complete_dataset_96mos/` — overall analysis results
  (rank averages, MSE search curve, actual vs predicted, summary).
- `output/age_groups_complete_96mos/` — per-age-group results, the
  α-by-age plot, and the sample utterances table.

## Notes on data scope

The R pipeline filters CHILDES participants by `language == "eng"`,
which is the CHILDES code for English-language speakers. The filter
operates at the participant level rather than the utterance level, so
the dataset may include a small number of non-English utterances
produced by English-categorized speakers in bilingual or
non-English-primary studies.

The selection also spans all English-language collections in CHILDES
(North American, UK, and clinical English), not a single region.
