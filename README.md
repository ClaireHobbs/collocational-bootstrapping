# Collocational Bootstrapping

This repository contains code and analysis for investigating **collocational bootstrapping**: 
a hypothesis about how distributional properties of subject-verb pairs support the learning 
of subject-verb agreement. We present two experiments examining this mechanism, one in neural 
networks and one analyzing its presence in child-directed speech.

## Experiments

This repository contains two experiments. Each has its own README with full setup instructions, 
runnable scripts, and expected outputs.

### [Experiment 1 — Neural Networks](./Experiment%201%20-%20NN/README.md)

Trains 2-layer transformer models synthetic sentences whose subject-verb 
co-occurrence distribution is controlled by a Zipfian parameter α, and
evaluates how α affects the model's ability to generalize agreement to unseen
subject-verb pairs and across prepositional phrase interveners. Sweeps α
from 0 to 3 in 0.1-step increments.

In the code, Z is used instead of α for ease of implementation; they refer to
the same parameter.

### [Experiment 2 — CHILDES Zipfian Analysis of Subject-Verb Pairs](./Experiment%202%20-%20CHILDES/README.md)

Estimates the empirical Zipfian parameter α describing the frequency
distribution of subject-verb pairs in child-directed speech, using data from
the CHILDES database. Runs the analysis once on the full dataset (target
child ages 0–96 months) and once per age group (eight 12-month bins).

Key results: overall α = 1.43; α decreases with child age from 1.46 (0–12 mo)
to 1.25 (84–96 mo).

## How the experiments connect

Experiment 2 measures the Zipfian parameter present in child-directed 
speech. Experiment 1 measures how that parameter affects a neural network's 
ability to learn subject-verb agreement. Together they let us ask 
whether the distributional word co-occurrence in child-directed speech 
is in a range that supports learning subject-verb agreement.
