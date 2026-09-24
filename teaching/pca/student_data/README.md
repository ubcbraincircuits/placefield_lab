# PCA student data

1200 consecutive, nonoverlapping 0.5-second bins from one synthetic session; 100 neurons. Source: the Placefield Lab synthetic CA1 generator.

- `neural_rates_hz.csv`: the numeric matrix for PCA. Rows are time bins; columns are neurons. Rates are spike counts divided by bin duration, with no smoothing or normalization. There is no index column.
- `position_labels.csv`: matching rows of metadata; `bin_id` is zero-based row order. Use these labels only to color or interpret the fitted result, not as PCA features. Do not independently sort either file.
- `provenance.json`: sample counts, binning, source checksum, and matrix checksum.

`start_time_s` is inclusive and `end_time_s` is exclusive. `label_time_s`, `x_cm`, `y_cm`, and `arm` refer to the central observed tracking sample. `arm_fraction` is the fraction of tracking samples in that bin with the same arm label. `is_transition` marks bins containing more than one arm/center category. The labels are center, east_open, west_open, north_closed, and south_closed—not Cartesian quadrants.

For the instructor reference: center each neuron and divide by its population standard deviation (ddof=0); leave constant columns as zero after centering. Fit PCA to all rows. Also try centering without scaling and compare. Do not whiten. No PCA scores or solutions are included in this student folder.

Questions: how much variance do the first two or three PCs capture? Do the arms appear as separate groups, continuous branches, or overlapping regions? How does preprocessing change your interpretation?
