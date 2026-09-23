# Model and analysis methods

[Back to README](README.md) | [Plot interpretation](INTERPRETING_THE_PLOTS.md)

The values below describe the default configuration. Command-line overrides and
custom JSON configurations are recorded with each generated session.

**Maze and behavior.** Two horizontal open arms and two vertical closed arms;
each arm extends 30 cm from a 6 × 6 cm center. The top-down footprint is 66 × 66 cm.
Elevation and walls are conceptual; no height or sensory representation is modeled.
Every excursion begins at the center, chooses an arm, runs out, pauses, and returns.
Smooth curves introduce small lateral variation while staying inside the maze.
This is a stylized excursion generator, not a validated ethological model.

Each closed arm has selection weight 1; each open arm has weight `exp(-avoidance)`.
Thus total open-arm choice probability is `1 / (1 + exp(avoidance))`, about 16.8%
at the default 1.6. Open-arm excursions also tend to be shallower and pauses shorter.
When avoidance is zero, all arms have identical choice, reach, and dwell distributions.
Actual entry and occupancy fractions fluctuate with the random trajectory.
Arm IDs in `entries.csv`: 0=east open, 1=west open, 2=north closed, 3=south closed.

**Cells.** One Gaussian field per synthetic CA1 neuron, with a center drawn uniformly
from valid spatial bins, width 5–10 cm, peak contribution 8–18 Hz, baseline 0.05–0.3 Hz.
These are illustrative choices, not a population fit. Euclidean field distance is used;
there is no mechanistic network producing the fields. A fixed, randomly selected 25%
of cells receive 1.8× gain on open arms. The gain multiplies baseline and field activity:

```text
base_rate_i(x) = baseline_i + peak_i * exp(-||x - center_i||² / (2 * sigma_i²))
rate_i(x) = base_rate_i(x) * gain_i ** is_open(x)
count_i(t) ~ Poisson(rate_i(position_t) * dt)
```

Counts are conditionally independent across cells and time. Given each count, spike
times are uniformly sampled within the 20 ms interval and globally sorted. This
represents synthetic, perfectly sorted units of the type one could analyze downstream
of Neuropixels. It does not simulate raw electrode voltages, waveforms, probe geometry,
drift, missed spikes, sorting errors, theta, refractoriness, or interneurons.
No dorsal/ventral CA1 distinction is asserted in this first version.

**Spatial maps.** Square 3 cm bins restricted to the maze footprint. Spike totals and
time occupancy are smoothed separately using a Gaussian of maze-graph distance
(4.5 cm bandwidth, truncated at 3 bandwidths), then divided. Graph neighbors share
an edge, preventing smoothing through empty space across maze corners. A 0.2 s
pseudocount toward the session mean regularizes sparse rate estimates. Bins with
less than 0.3 s raw occupancy are blank. Ground-truth maps evaluate the rate function
at bin centers; estimated maps average sampled locations within bins and are smoothed.

**Decoding.** Train on the first 70% of the session; test on the final 30% in
nonoverlapping windows. Rates, occupancy prior, and supported bins use training data
only. The full-session maps shown in the place-field figure are never used by the decoder.
For candidate bin x and observed population counts n:

```text
log posterior(x | n) = constant + log prior(x)
                      + sum_i[n_i * log(rate_i(x)) - window_s * rate_i(x)]
```

Normalize with log-sum-exp and select the maximum-posterior bin. The target is the
mouse's position at the central observed sample of the window. All test windows
are scored, including positions with insufficient training occupancy; the decoder
cannot select unsupported bins. Coverage is explicitly reported. Errors are Euclidean
centimeters, not maze travel distance. No trajectory smoothing or true test positions
enter the decoder. Posterior probabilities assume independent Poisson observations;
their empirical calibration has not been established.

Two baselines: always predict the most occupied training bin, and apply a single
fixed random permutation to test neuron identities. The latter is a diagnostic,
not a multi-shuffle significance test. `summary.json` also gives errors by zone,
since pooled errors can conceal poor performance in the rarely visited open arms.

**Arm comparison.** Observed arm firing rates are spike totals divided by arm dwell
time. Field position and uneven within-arm sampling can still produce arm differences.
The second arm-effect panel compares expected rates at exactly the same open-arm
positions with neural gain enabled and disabled. This isolates the imposed gain
within this model; it is not an estimate of an anxiety effect from experimental data.
Neurons are not independent animal replicates. No significance tests are presented.

## Files and data

- `model.py`: movement, cell rates, spikes, rate maps, decoding, metrics.
- `plots.py`: Matplotlib/Seaborn figures and Pillow animation.
- `run.py`: command-line entry point and data exports.
- `test_model.py`: geometry, conservation, reproducibility, modulation isolation,
  leakage prevention, decoder arithmetic, and avoidance checks.
- `results/session.npz`: compact arrays; `np.load(path, allow_pickle=False)`.
- `results/trajectory.csv`: tracking at 50 Hz by default; cm and seconds.
- `results/spikes.csv`: sorted spike times in seconds and zero-based neuron IDs.
- `results/cells.csv`: ground-truth cell parameters.
- `results/entries.csv`: arm entry times, IDs, and excursion target depths.
- `results/decoding.csv`: held-out targets, predictions, support flags, errors.
- `results/summary.json`: metrics, parameters, and environment.

In `session.npz`, `spike_counts` is time × neuron; rate arrays are spatial bin × neuron;
`grid_xy_cm` maps spatial bin IDs to coordinates. `posterior` is test window × candidate bin;
its column IDs are `posterior_spatial_bins`. The matching test-window times and targets
are in `decoding.csv`. Firing rates use Hz, time uses seconds, and the posterior
contains dimensionless probabilities that sum to one per test window. Random seeds reproduce
runs within the recorded software environment; numerical results may vary with versions.

## Scientific context

The following motivate the ingredients, not the numerical parameters or a claim of
reproducing an experiment:

- [Jun et al., 2017, Nature: Neuropixels](https://www.nature.com/articles/nature24636)
  describes high-density neural recording technology.
- [Jimenez et al., 2018, Neuron: Anxiety cells in a hippocampal-hypothalamic circuit](https://doi.org/10.1016/j.neuron.2018.01.016)
  provides a biological motivation for separating spatial and arm-related activity.
  That work used calcium imaging; this prototype's spiking model is not a recreation
  of that recording method or its results.
- [Zhang et al., 1998, Journal of Neurophysiology](https://doi.org/10.1152/jn.1998.79.2.1017)
  develops population reconstruction methods for hippocampal place cells.

Natural next steps: multiple simulated mice and sessions, direction/speed/theta terms,
measured spike-sorting noise, and cross-validated comparison of spatial-only versus
spatial-plus-context encoding models.
