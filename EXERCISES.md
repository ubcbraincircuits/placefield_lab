# Classroom exercises

[Back to README](README.md) · [Interpret the plots](INTERPRETING_THE_PLOTS.md)

Predict the result before running each comparison. In commands below, use your virtual
environment's Python executable as described in the README. Each experiment saves to
its own directory. Keep default settings except for the parameter being investigated.

## 1. Does avoidance change the data we collect?

```bash
python run.py --out experiments/default
python run.py --avoidance 0 --out experiments/no_avoidance
```

Compare the occupancy map, open-arm time, and blank regions in the estimated place
fields. Does the unbiased session spend exactly half its arm time in open arms?
Repeat with seeds 7 and 123 before deciding that one run captures the rule.

**Interpretation:** equal selection weights imply symmetry in the generating process,
not exactly equal observed counts in a finite session. Avoidance changes which fields
can be estimated well. Changing avoidance preserves the generated cell parameters
when the seed and neuron count are fixed, but alters movement and spike realizations.

## 2. Does open-arm preference require an open-arm gain?

```bash
python run.py --open-gain 1 --out experiments/no_gain
```

Do some neurons still fire more in open arms? Find one with an open-arm place field.
Compare the observed-rate scatter with the counterfactual panel.

**Interpretation:** a spatial field alone can create arm preference. Changing gain
preserves the trajectory and underlying field parameters for the same seed, but the
Poisson draws need not remain paired spike by spike. The expected-rate counterfactual
is the exact isolated comparison in this model.

## 3. How much data is enough?

```bash
python run.py --duration 60 --out experiments/one_minute
python run.py --duration 600 --out experiments/ten_minutes
```

Compare unsupported spatial bins and `test_spatial_coverage` in the summaries. Inspect
the number of held-out windows for each arm type before comparing its decoding error.
A short session may contain very few or no test visits to an arm type.

**Interpretation:** longer sessions usually improve sampling, but changing duration
also changes the chronological train/test split. This is not a controlled learning
curve with the same test set. Do not interpret one pair as a guaranteed performance trend.

## 4. What happens when a neuron is mislabeled?

Use the default decoding figure. Compare the intact decoder with the fixed neuron-ID
shuffle. Why does changing neuron identity matter even though the number of spikes
in each test window is preserved?

**Interpretation:** the decoder learned a different field for each neuron. Shuffling
breaks those associations. This does not reproduce the full range of spike-sorting
errors, and one shuffle does not produce a p-value.

## 5. Change the decoding timescale

Copy `config.json` to `config_long_window.json`, change `decode_window_s` from `0.2`
to `1.0`, then run:

```bash
python run.py --config config_long_window.json --out experiments/long_window
```

More spikes in a window can improve count precision. But the mouse can move farther
during that window, while the decoder assumes one location. Compare errors near turns
and in slow portions of the trajectory. A bigger window also produces fewer predictions.

**Interpretation:** temporal averaging trades sampling noise against motion blur; the
best window is not automatically the largest one.

## Optional coding challenge: a fair neuron-count comparison

The CLI supports `--cells`, but changing it regenerates a different cell population.
A stronger comparison simulates one population, selects nested subsets of its neurons,
fits each decoder on the same time split, and evaluates all subsets on identical test
windows. Repeat subset selection and random seeds. State explicitly whether the units
of replication are cells, synthetic sessions, or independent synthetic animals.

## Using an AI assistant responsibly in this exercise

Ask it to explain a prediction, identify a confound, or propose a falsifying control.
Then check the suggestion against the generator and exported data. Useful prompts:

- “Which part of this plot was imposed by the simulation rather than discovered?”
- “Show me how to verify that test spikes cannot affect training rate maps.”
- “What denominator did we use for open-arm occupancy?”
- “Design a comparison that changes one mechanism while holding the others fixed.”

If you add a new biological mechanism, document its equation, units, and assumptions,
and add a check that could fail if its implementation were wrong.
