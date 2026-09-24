"""Export a student PCA dataset and optional instructor reference figures.

Only NumPy is needed for PCA itself: centered/scaled matrix -> SVD.
Run after run.py, or point --session at an existing session.npz.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from plots import plt, sns, style, title, maze, finish, INK, GRAY, TEAL, ORANGE

ARMS = ['center', 'east_open', 'west_open', 'north_closed', 'south_closed']
PALETTE = dict(zip(ARMS, [GRAY, ORANGE, '#b64d72', TEAL, '#4c68bd']))
LABELS = dict(zip(ARMS, ['Center', 'East · open', 'West · open', 'North · closed', 'South · closed']))


def arm_labels(xy):
    """Physical arm identity, with a separate central square; not Cartesian quadrants."""
    x, y = np.asarray(xy).T
    return np.select([x > 3, x < -3, y > 3, y < -3], ARMS[1:], default='center')


def bin_session(time_s, position_cm, counts, bin_s=.5):
    time_s, position_cm, counts = map(np.asarray, (time_s, position_cm, counts))
    if len(time_s) < 2 or position_cm.shape != (len(time_s), 2) or counts.ndim != 2 or len(counts) != len(time_s):
        raise ValueError('Expected aligned time, time × 2 position, and time × neuron counts.')
    dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(bin_s) or not np.isfinite(dt) or dt <= 0 or bin_s < dt:
        raise ValueError('Bin width must be finite and at least one simulation step.')
    if not np.allclose(np.diff(time_s), dt, rtol=1e-7, atol=1e-9):
        raise ValueError('Time samples must be uniformly spaced.')
    size = round(bin_s / dt)
    if not np.isclose(size * dt, bin_s, rtol=1e-8, atol=1e-9):
        raise ValueError('Bin width must be an integer multiple of the simulation time step.')
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError('Spike counts must be finite nonnegative integers.')
    if not np.isfinite(position_cm).all():
        raise ValueError('Positions must be finite.')
    n = len(time_s) // size
    if n < 4:
        raise ValueError('Need at least four complete time bins.')
    end = n * size
    total = counts[:end].reshape(n, size, counts.shape[1]).sum(axis=1)
    rates = total / bin_s
    mid = np.arange(n) * size + size // 2
    labels = arm_labels(position_cm)
    block_labels = labels[:end].reshape(n, size)
    metadata = pd.DataFrame(dict(
        bin_id=np.arange(n), start_time_s=time_s[np.arange(n)*size],
        end_time_s=time_s[np.arange(n)*size]+bin_s, label_time_s=time_s[mid],
        x_cm=position_cm[mid, 0], y_cm=position_cm[mid, 1], arm=labels[mid],
        arm_fraction=(block_labels == labels[mid, None]).mean(axis=1),
        is_transition=np.any(block_labels != block_labels[:, :1], axis=1)))
    return rates, metadata, dict(dt_s=dt, bin_s=bin_s, samples_per_bin=size,
        used_samples=end, discarded_samples=len(time_s)-end,
        included_spikes=int(total.sum()), n_bins=n, n_neurons=counts.shape[1])


def fit_pca(x, standardize=True):
    """Full descriptive PCA, retaining all components. No labels enter the fit."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError('PCA needs a finite matrix with at least two rows and columns.')
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    scale = np.where(std > 0, std, 1.) if standardize else np.ones(x.shape[1])
    centered = (x - mean) / scale
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2 / (len(x)-1)
    if variance.sum() <= 0:
        raise ValueError('PCA is undefined when every neuron is constant over time.')
    # Deterministic display convention only; PCA signs carry no intrinsic meaning.
    largest = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(len(components)), largest])
    components *= np.where(signs == 0, 1, signs)[:, None]
    return dict(mean=mean, scale=scale, components=components,
        scores=centered @ components.T, variance=variance, ratio=variance/variance.sum(),
        constant_neurons=np.flatnonzero(std == 0).tolist(), standardize=standardize)


def scatter_arms(ax, scores, metadata, i=0, j=1, pca=None, legend=False):
    for arm in ARMS:
        select = metadata['arm'].eq(arm).to_numpy()
        ax.scatter(scores[select, i], scores[select, j], s=12, alpha=.45,
                   color=PALETTE[arm], edgecolors='none', label=LABELS[arm], rasterized=True)
    ax.set_xlabel(f'PC{i+1} ({pca["ratio"][i]:.1%} variance)' if pca else f'PC{i+1}')
    ax.set_ylabel(f'PC{j+1} ({pca["ratio"][j]:.1%} variance)' if pca else f'PC{j+1}')
    if legend:
        ax.legend(frameon=False, fontsize=9, markerscale=1.8)


def reference_figures(rates, metadata, scaled, raw, out, bin_s):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(left=.055, right=.97, bottom=.22, top=.77, wspace=.32)
    title(fig, 'A spatial code in population activity',
          f'{len(rates):,} time bins × {rates.shape[1]} neurons · {bin_s:g} s bins · PCA of z-scored firing rates')
    maze(axes[0], fill=True)
    for arm in ARMS:
        select = metadata.arm.eq(arm)
        axes[0].scatter(metadata.loc[select, 'x_cm'], metadata.loc[select, 'y_cm'],
                        s=10, color=PALETTE[arm], alpha=.45, edgecolors='none')
    axes[0].set_title('Known position', loc='left', fontsize=12)
    scatter_arms(axes[1], scaled['scores'], metadata, pca=scaled)
    axes[1].set_title('First two principal components', loc='left', fontsize=12)
    scatter_arms(axes[2], scaled['scores'], metadata, 0, 2, scaled, legend=True)
    axes[2].set_title('A third view of the same data', loc='left', fontsize=12)
    fig.text(.055, .075, 'Colors use physical arm identity at the central tracking sample of each bin. Labels and positions never enter PCA.\n'
             'All time bins are included, including center visits and transitions. This is a descriptive fit to the full session.',
             fontsize=10, color=GRAY)
    finish(fig, out, '01_pca_spatial_structure')

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, left=.07, right=.94, top=.85, bottom=.12, hspace=.42, wspace=.35)
    title(fig, 'What PCA captures—and what preprocessing changes',
          'Scree, three-dimensional projection, covariance PCA, and component weights')
    ax = fig.add_subplot(gs[0, 0])
    n_show = min(15, len(scaled['ratio']))
    indices = np.arange(1, n_show+1)
    ax.bar(indices, scaled['ratio'][:n_show]*100, color=TEAL, alpha=.8)
    ax.set(xlabel='Principal component', ylabel='Individual variance (%)', xticks=indices[::2])
    twin = ax.twinx()
    twin.plot(indices, np.cumsum(scaled['ratio'])[:n_show]*100, 'o-', c=ORANGE, ms=4)
    twin.set(ylabel='Cumulative variance (%)', ylim=(0, 105))
    ax.set_title('Standardized PCA: explained variance', loc='left', fontsize=12)
    ax = fig.add_subplot(gs[0, 1], projection='3d')
    for arm in ARMS:
        select = metadata.arm.eq(arm)
        s = scaled['scores'][select]
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=6, color=PALETTE[arm], alpha=.4, depthshade=False)
    ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')
    ax.view_init(elev=22, azim=38)
    ax.set_title(f'First three PCs: {scaled["ratio"][:3].sum():.1%} of variance', fontsize=12)
    ax = fig.add_subplot(gs[1, 0])
    scatter_arms(ax, raw['scores'], metadata, pca=raw)
    ax.set_title('Centered rates without standardization', loc='left', fontsize=12)
    ax = fig.add_subplot(gs[1, 1])
    weights = scaled['components'][:3]
    sns.heatmap(weights, ax=ax, cmap='vlag', center=0,
                vmin=-abs(weights).max(), vmax=abs(weights).max(),
                xticklabels=max(1, rates.shape[1]//10), yticklabels=['PC1', 'PC2', 'PC3'],
                cbar_kws={'label': 'Component weight', 'shrink': .75})
    ax.set(xlabel='Neuron ID', ylabel='')
    ax.set_title('Standardized PCA: signed neuron weights', loc='left', fontsize=12)
    fig.text(.07, .045, 'Standardization gives each nonconstant neuron unit variance. Centering alone lets high-variance neurons contribute more.\n'
             'Component signs may flip between implementations. Time bins are temporally related; these plots do not establish four discrete clusters.',
             fontsize=10, color=GRAY)
    finish(fig, out, '02_pca_diagnostics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=Path(__file__).with_name('results') / 'session.npz')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'teaching' / 'pca')
    parser.add_argument('--bin-seconds', type=float, default=.5)
    parser.add_argument('--export-only', action='store_true', help='Write student data without computing PCA.')
    args = parser.parse_args()
    with np.load(args.session, allow_pickle=False) as data:
        rates, metadata, provenance = bin_session(data['time_s'], data['position_cm'], data['spike_counts'], args.bin_seconds)
    student = args.out / 'student_data'
    student.mkdir(parents=True, exist_ok=True)
    columns = [f'neuron_{i:03d}_hz' for i in range(rates.shape[1])]
    feature_file = student / 'neural_rates_hz.csv'
    pd.DataFrame(rates, columns=columns).to_csv(feature_file, index=False, float_format='%.12g')
    metadata.to_csv(student / 'position_labels.csv', index=False, float_format='%.12g')
    # Use the exact serialized matrix for the instructor answer, too.
    rates = pd.read_csv(feature_file).to_numpy()
    provenance.update(source_file=args.session.name,
        source_sha256=hashlib.sha256(args.session.read_bytes()).hexdigest(),
        neural_csv_sha256=hashlib.sha256(feature_file.read_bytes()).hexdigest(),
        numpy_version=np.__version__, arm_bin_counts=metadata.arm.value_counts().to_dict(),
        transition_bins=int(metadata.is_transition.sum()),
        matrix_orientation='rows=time bins, columns=neurons', rate_unit='Hz',
        smoothing='none', fit_scope='all complete bins; no balancing or filtering')
    (student / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    (student / 'README.md').write_text(
        '# PCA student data\n\n'
        f'{len(rates)} consecutive, nonoverlapping {args.bin_seconds:g}-second bins from one synthetic session; '
        f'{rates.shape[1]} neurons. Source: the Placefield Lab synthetic CA1 generator.\n\n'
        '- `neural_rates_hz.csv`: the numeric matrix for PCA. Rows are time bins; columns are neurons. '
        'Rates are spike counts divided by bin duration, with no smoothing or normalization. There is no index column.\n'
        '- `position_labels.csv`: matching rows of metadata; `bin_id` is zero-based row order. '
        'Use these labels only to color or interpret the fitted result, not as PCA features. '
        'Do not independently sort either file.\n'
        '- `provenance.json`: sample counts, binning, source checksum, and matrix checksum.\n\n'
        '`start_time_s` is inclusive and `end_time_s` is exclusive. '
        '`label_time_s`, `x_cm`, `y_cm`, and `arm` refer to the central observed tracking sample. '
        '`arm_fraction` is the fraction of tracking samples in that bin with the same arm label. '
        '`is_transition` marks bins containing more than one arm/center category. '
        'The labels are center, east_open, west_open, north_closed, and south_closed—not Cartesian quadrants.\n\n'
        'For the instructor reference: center each neuron and divide by its population standard deviation '
        '(ddof=0); leave constant columns as zero after centering. Fit PCA to all rows. '
        'Also try centering without scaling and compare. Do not whiten. '
        'No PCA scores or solutions are included in this student folder.\n\n'
        'Questions: how much variance do the first two or three PCs capture? '
        'Do the arms appear as separate groups, continuous branches, or overlapping regions? '
        'How does preprocessing change your interpretation?\n', encoding='utf-8')
    if args.export_only:
        print(f'Student CSVs: {student.resolve()}')
        return
    reference = args.out / 'reference'
    reference.mkdir(parents=True, exist_ok=True)
    scaled, raw = fit_pca(rates), fit_pca(rates, standardize=False)
    summary = dict(n_bins=len(rates), n_neurons=rates.shape[1],
        bin_seconds=args.bin_seconds, constant_neurons=scaled['constant_neurons'],
        standardized_first_three=scaled['ratio'][:3].tolist(),
        standardized_first_two_total=float(scaled['ratio'][:2].sum()),
        standardized_first_three_total=float(scaled['ratio'][:3].sum()),
        centered_only_first_three=raw['ratio'][:3].tolist(),
        components_for_80_percent=int(np.searchsorted(np.cumsum(scaled['ratio']), .8)+1),
        fit_scope='descriptive full-session PCA; labels excluded',
        neural_csv_sha256=provenance['neural_csv_sha256'])
    for name, fit in [('standardized', scaled), ('centered_only', raw)]:
        pd.DataFrame(fit['scores'], columns=[f'PC{i+1}' for i in range(fit['scores'].shape[1])]).to_csv(reference / f'{name}_scores.csv', index=False)
        pd.DataFrame(fit['components'].T, index=columns,
                     columns=[f'PC{i+1}' for i in range(fit['components'].shape[0])]).to_csv(reference / f'{name}_weights.csv', index_label='neuron')
        pd.DataFrame(dict(component=np.arange(1, len(fit['ratio'])+1), variance=fit['variance'],
            explained_variance_ratio=fit['ratio'], cumulative_ratio=np.cumsum(fit['ratio']))).to_csv(reference / f'{name}_variance.csv', index=False)
    (reference / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    reference_figures(rates, metadata, scaled, raw, reference, args.bin_seconds)
    print(json.dumps(summary, indent=2))
    print(f'Wrote student data and instructor reference: {args.out.resolve()}')


if __name__ == '__main__':
    main()
