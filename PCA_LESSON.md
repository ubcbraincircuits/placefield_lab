# PCA classroom dataset

[Back to README](README.md)

Use the original synthetic recording to ask whether a low-dimensional view of
population activity reveals where the mouse was on the maze. You can use Python,
MATLAB, R, or another numerical environment. No regeneration is needed to use the
included CSVs.

## Student handout

Give students these files from [student_data](teaching/pca/student_data):

- [neural_rates_hz.csv](teaching/pca/student_data/neural_rates_hz.csv): **1,200 rows × 100 neurons**.
  Each row contains firing rates in Hz for a consecutive 0.5-second interval.
- [position_labels.csv](teaching/pca/student_data/position_labels.csv): the same 1,200 rows
  with time, position, physical arm identity, and transition flags.
- The accompanying [data dictionary](teaching/pca/student_data/README.md) and
  [provenance](teaching/pca/student_data/provenance.json).

The numeric matrix has a single header row and no index column. There are no missing
values in this example. The first data row has `bin_id=0` in the metadata, the second
has `bin_id=1`, and so on. Preserve the same order in both files. If filtering, use
the same mask for both. Each value is the number of spikes divided by 0.5 seconds:
zero means no spikes in that bin. There is no temporal smoothing or prior normalization.

**Fit PCA to neuron columns only.** Time, position, arm labels, and transition flags
are metadata for plotting and interpretation. Including them in PCA would answer a
different question. Labels were derived from the central tracking sample of each bin;
`arm_fraction` reports how much of the bin had that label. All 123 transition bins
are retained in the included reference.

The four labels describe **maze arms**, not Cartesian quadrants: east and west are
open, north and south are closed. The center is a fifth category. Arm identity remains
the same when the mouse wanders a little to either side of an arm's centerline.

## Suggested assignment

1. Load the matrix. Identify which axis represents observations and which represents neurons.
2. Center each neuron's activity across time. First run PCA without further scaling.
3. Repeat after dividing each centered neuron by its standard deviation. To match the
   standardized reference exactly, use the population standard deviation (`ddof=0`).
   If a column is constant, centering makes it zero; use a divisor of 1 for that column.
4. Plot individual and cumulative explained variance. State how many components you
   would retain and why.
5. Plot PC1 against PC2 and PC1 against PC3. Only now use the metadata to color points
   by arm. Also inspect a 3D projection of the first three components if available.
6. Compare the two preprocessing choices. Do you see distinct clouds, continuous
   branches, or overlap? What happens near the center and during transitions?
7. Explain why a component is a weighted combination of neurons rather than a neuron
   or a physical coordinate. Inspect some positive and negative component weights.

Use all rows for the reference comparison, without whitening, balancing arms, filtering,
or adding another smoothing step. This is **descriptive PCA of a complete session**,
not a held-out decoding evaluation. For predictive use, fit centering, scaling, and
PCA on training data only and apply that transform to held-out data.

Component signs are arbitrary: a mirrored plot can be completely correct. Compare
explained-variance ratios and geometry before comparing signs. Our display convention
makes the largest-magnitude weight in each component positive. Nearly equal eigenvalues
can also make component orientations unstable. Absolute explained variances depend on
normalization conventions; explained-variance ratios are easier to compare.

## Discussion

- PCA maximizes variance, not arm classification or spatial accuracy. Good-looking
  separation is not a statistical test for a set of discrete neural states.
- Adjacent windows are temporally related. The 1,200 rows are not 1,200 independent mice.
- Unequal occupancy affects the covariance matrix. An equally weighted collection of
  visited time bins is not an equally sampled maze.
- Standardizing gives each nonconstant neuron equal marginal variance; it can also
  increase the influence of noisy, low-variance cells. Centering alone is also a valid
  analysis when differences in firing-rate variance are part of the question.
- Location, field overlap, movement, spike noise, and the imposed open-arm gain can
  all shape the point cloud. An apparent branch does not by itself isolate any one cause.

## Instructor reference

The [reference directory](teaching/pca/reference/README.md) contains two figure sheets,
explained-variance tables, component weights, and scores for both preprocessing choices.
Share the student folder first if students should implement PCA before seeing answers.
The reference implementation is [pca_reference.py](pca_reference.py); it uses NumPy's
SVD and adds no new project dependencies.

To regenerate from a saved session (use your virtual environment's Python):

```bash
python pca_reference.py --session results/session.npz
```

To export another class dataset without calculating reference answers:

```bash
python pca_reference.py --session results/session.npz --bin-seconds 0.2 --export-only --out experiments/pca_200ms
```

Only complete bins are used; any trailing incomplete bin is discarded and its sample
count is recorded in provenance. Default output files are replaced on rerun. Use `--out`
for new versions and keep their matrices, labels, and reference answers together.

For library conventions, see the [scikit-learn PCA documentation](https://sklearn.org/stable/modules/decomposition.html#pca):
its PCA centers features but does not standardize them automatically. The same distinction
matters when comparing results across languages and packages.
