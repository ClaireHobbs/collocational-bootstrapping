# Experiment 2 — CHILDES Zipfian Analysis of Subject-Verb Pairs

## Overview

This experiment estimates the Zipfian parameter α that describes the frequency
distribution of subject-verb pairs in child-directed speech from the CHILDES 
database. The analysis is run twice: once on the full dataset (producing a 
single overall α for 0 to 96 months) and once per age group (producing α for
each of eight 12-month age bins from 0 to 96 months).

Expected Results:

- Overall α = 1.43 (4,739,189 utterances, 2,802,071 subject-verb pairs)
- α decreases with child age: 1.46 (0-12 mo) → 1.25 (84-96 mo)

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
