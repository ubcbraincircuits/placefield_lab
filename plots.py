"""Publication-style Matplotlib/Seaborn figures and a short mouse replay."""
import numpy as np
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from model import zone, firing_rate

INK = '#23374d'
TEAL = '#138a86'
ORANGE = '#df8556'
PAPER = '#faf9f5'
GRAY = '#8795a5'
COLORS = [GRAY, ORANGE, TEAL]
OUTLINE = np.array([[-3, -33], [3, -33], [3, -3], [33, -3], [33, 3], [3, 3],
                    [3, 33], [-3, 33], [-3, 3], [-33, 3], [-33, -3], [-3, -3]])


def style():
    sns.set_theme(context='notebook', style='ticks', font='DejaVu Sans',
                  rc={'figure.facecolor': PAPER, 'axes.facecolor': PAPER,
                      'text.color': INK, 'axes.labelcolor': INK,
                      'xtick.color': INK, 'ytick.color': INK,
                      'axes.spines.top': False, 'axes.spines.right': False,
                      'axes.titleweight': 'bold', 'axes.titlepad': 12,
                      'savefig.facecolor': PAPER, 'font.size': 10})


def title(fig, main, subtitle):
    fig.suptitle(main, x=.055, y=.978, ha='left', fontsize=24, weight='bold', color=INK)
    fig.text(.055, .978 - .60 / fig.get_figheight(), subtitle, fontsize=11, color=GRAY)


def maze(ax, labels=False, fill=True):
    if fill:
        ax.add_patch(Rectangle((-33, -3), 66, 6, color=ORANGE, alpha=.13, lw=0))
        ax.add_patch(Rectangle((-3, -33), 6, 66, color=TEAL, alpha=.13, lw=0))
    ax.add_patch(Polygon(OUTLINE, closed=True, fill=False, edgecolor=INK, lw=1.1, zorder=5))
    ax.set(xlim=(-37, 37), ylim=(-37, 37), aspect='equal', xlabel='x (cm)', ylabel='y (cm)')
    ax.set_xticks([-30, 0, 30]); ax.set_yticks([-30, 0, 30])
    if labels:
        ax.text(17, 9, 'OPEN', color=ORANGE, ha='center', fontsize=9, weight='bold')
        ax.text(-17, 9, 'OPEN', color=ORANGE, ha='center', fontsize=9, weight='bold')
        ax.text(7, 24, 'CLOSED', color=TEAL, fontsize=9, rotation=90, va='center', weight='bold')


def map_image(ax, values, grid, cmap='mako', vmin=0, vmax=None):
    matrix = np.full(grid['mask'].shape, np.nan)
    matrix[grid['mask']] = values
    im = ax.imshow(matrix, extent=(-33, 33, -33, 33), origin='lower', interpolation='nearest',
                   cmap=cmap, vmin=vmin, vmax=vmax)
    maze(ax, fill=False)
    return im


def finish(fig, out, name):
    # Render in memory before writing the completed figure.
    for extension in ['png', 'pdf']:
        buffer = BytesIO()
        fig.savefig(buffer, format=extension, dpi=160)
        (out / f'{name}.{extension}').write_bytes(buffer.getvalue())
    plt.close(fig)


def overview(data, maps, result, summary, cfg, out):
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, left=.065, right=.96, top=.86, bottom=.11, wspace=.34, hspace=.48)
    title(fig, 'A mouse, a maze, a spatial code',
          f'SYNTHETIC CA1  /  {cfg.n_cells} neurons  /  {cfg.duration_s / 60:g} min  /  seed {cfg.seed}')
    ax = fig.add_subplot(gs[0, 0]); maze(ax, labels=True)
    ax.plot(*data['xy'][::3].T, color=INK, lw=.55, alpha=.17)
    ax.set_title('01  Exploration with open-arm avoidance', loc='left', fontsize=11)
    ax = fig.add_subplot(gs[0, 1])
    im = map_image(ax, maps['occupancy'], data['grid'], 'mako')
    fig.colorbar(im, ax=ax, shrink=.8, label='Time in spatial bin (s)')
    ax.set_title('02  Where the mouse spends time', loc='left', fontsize=11)
    ax = fig.add_subplot(gs[0, 2])
    vals = [summary['occupancy_seconds'][k] for k in ['open', 'closed', 'center']]
    bars = ax.bar(['Open', 'Closed', 'Center'], vals, color=[ORANGE, TEAL, GRAY], width=.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + max(vals)*.025,
                f'{v / (len(data["xy"]) * cfg.dt_s):.0%}', ha='center', fontsize=12, weight='bold')
    ax.set(ylim=(0, max(vals)*1.2), ylabel='Time (s)')
    ax.set_title('03  Open versus closed arms', loc='left', fontsize=11)
    ax.text(.02, .92, f'{summary["open_entries"]} / {summary["total_entries"]} entries into open arms',
            transform=ax.transAxes, fontsize=10)
    ax = fig.add_subplot(gs[1, :2])
    show = data['spike_times'] < 30
    unit_colors = np.where(data['cells']['gain'] != 1, ORANGE, TEAL)
    ax.scatter(data['spike_times'][show], data['spike_units'][show], s=2,
               c=unit_colors[data['spike_units'][show]], alpha=.7, linewidths=0, rasterized=True)
    ax.set(xlim=(0, 30), ylim=(-1, cfg.n_cells), xlabel='Time (s)', ylabel='Neuron ID')
    ax.set_title('04  Sorted spike raster · first 30 seconds', loc='left', fontsize=11)
    ax = fig.add_subplot(gs[1, 2])
    z = zone(data['xy'])
    means = [data['counts'][z == k].sum(axis=0) / ((z == k).sum() * cfg.dt_s)
             if (z == k).any() else np.full(cfg.n_cells, np.nan) for k in [1, 2]]
    ax.scatter(means[1], means[0], c=unit_colors, s=24, alpha=.8, edgecolors='none')
    finite = np.asarray(means)[np.isfinite(means)]
    lim = max(finite.max() if len(finite) else 1, 1) * 1.08
    if not ((z == 1).any() and (z == 2).any()):
        ax.text(.5, .5, 'An arm type was not visited', ha='center', transform=ax.transAxes)
    ax.plot([0, lim], [0, lim], '--', color=GRAY, lw=1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel='Closed-arm mean rate (Hz)', ylabel='Open-arm mean rate (Hz)')
    ax.set_title('05  Each dot is one neuron', loc='left', fontsize=11)
    fig.text(.065, .035, 'Orange neurons have imposed open-arm gain. Rate differences also reflect field location and sampling.\n'
             'A synthetic sorted-unit dataset: no raw probe voltages, spike sorting, or empirical anxiety inference.', fontsize=10, color=GRAY)
    finish(fig, out, '01_overview')


def place_fields(data, maps, cfg, out):
    truth = firing_rate(data['grid']['xy'], data['cells'])
    # Four modulated and four other neurons, spread along the population index.
    selected = []
    groups = [np.flatnonzero(data['cells']['gain'] != 1), np.flatnonzero(data['cells']['gain'] == 1)]
    for group in groups:
        if len(group):
            selected.extend(group[np.linspace(0, len(group)-1, min(4, len(group))).astype(int)].tolist())
    selected += [i for i in range(cfg.n_cells) if i not in selected][:8-len(selected)]
    selected = selected[:8]
    rows = int(np.ceil(len(selected) / 2))
    fig, axes = plt.subplots(rows, 4, figsize=(13, 3*rows + 1.7), squeeze=False)
    fig.subplots_adjust(left=.06, right=.92, bottom=.075, top=.86, wspace=.65, hspace=.65)
    title(fig, 'Place fields: ground truth meets spikes',
          'Each pair shares a color scale. Estimated maps use spikes / occupancy with maze-constrained smoothing.')
    for idx, unit in enumerate(selected):
        row, col = idx // 2, (idx % 2) * 2
        observed = maps['rates'][:, unit].copy()
        observed[~maps['supported']] = np.nan
        vmax = max(truth[:, unit].max(), np.nanmax(observed))
        ax1, ax2 = axes[row, col:col+2]
        map_image(ax1, truth[:, unit], data['grid'], 'rocket', vmax=vmax)
        im = map_image(ax2, observed, data['grid'], 'rocket', vmax=vmax)
        gain = data['cells']['gain'][unit]
        ax1.set_title(f'Cell {unit:02d} · true ({gain:g}× open)', fontsize=10, loc='left')
        ax2.set_title('Estimated · full session', fontsize=10, loc='left')
        fig.colorbar(im, ax=ax2, fraction=.046, pad=.04, label='Hz')
        for ax in [ax1, ax2]:
            ax.set_xlabel(''); ax.set_ylabel(''); ax.tick_params(labelsize=8)
    for idx in range(len(selected), rows * 2):
        for ax in axes[idx//2, (idx%2)*2:(idx%2)*2+2]: ax.set_visible(False)
    fig.text(.06, .025, f'Blank estimated bins have < {cfg.min_occupancy_s:g} s occupancy. '
             'Fields are prescribed Gaussian rate profiles; their emergence is not modeled.', fontsize=10, color=GRAY)
    finish(fig, out, '02_place_fields')


def arm_effects(data, cfg, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.subplots_adjust(left=.065, right=.96, bottom=.23, top=.78, wspace=.38)
    title(fig, 'Separate behavior from neural modulation',
          'Open-arm avoidance changes sampling. Open-arm gain changes a selected subset of neurons.')
    z = zone(data['xy']); modulated = data['cells']['gain'] != 1
    observed = np.stack([data['counts'][z == k].sum(axis=0) / ((z == k).sum()*cfg.dt_s)
                        if (z == k).any() else np.full(cfg.n_cells, np.nan) for k in [1, 2]])
    mi = (observed[0] - observed[1]) / np.maximum(observed.sum(axis=0), 1e-9)
    groups = ['No imposed gain', 'Imposed gain']
    colors = [TEAL, ORANGE]
    rng = np.random.default_rng(20)
    for g, group in enumerate([~modulated, modulated]):
        axes[0].scatter(g + rng.uniform(-.12, .12, group.sum()), mi[group], color=colors[g], s=20, alpha=.7)
        if group.any() and np.isfinite(mi[group]).any():
            axes[0].plot([g-.2, g+.2], [np.median(mi[group])]*2, color=INK, lw=3)
    axes[0].axhline(0, color=GRAY, ls='--', lw=1)
    axes[0].set(xticks=[0, 1], xticklabels=groups, ylabel='Observed arm modulation index', ylim=(-1.05, 1.05))
    if not np.isfinite(mi).any():
        axes[0].text(.5, .5, 'An arm type was not visited', ha='center', transform=axes[0].transAxes)
    axes[0].set_title('Observed arm preference', loc='left', fontsize=12)
    open_xy = data['xy'][z == 1]
    if len(open_xy):
        with_gain = firing_rate(open_xy, data['cells']).mean(axis=0)
        no_gain = firing_rate(open_xy, data['cells'], modulation=False).mean(axis=0)
        axes[1].scatter(no_gain, with_gain, c=np.where(modulated, ORANGE, TEAL), s=24, alpha=.8)
        limit = max(with_gain.max(), no_gain.max(), 1) * 1.05
        axes[1].plot([0, limit], [0, limit], '--', c=GRAY, lw=1)
        axes[1].set(xlim=(0, limit), ylim=(0, limit))
    else:
        axes[1].text(.5, .5, 'No open-arm samples', ha='center', transform=axes[1].transAxes)
    axes[1].set(xlabel='Expected Hz with gain disabled', ylabel='Expected Hz with configured gain')
    axes[1].set_title('Same-position counterfactual', loc='left', fontsize=12)
    bias = np.linspace(0, 5, 100)
    axes[2].plot(bias, 1/(1+np.exp(bias)), c=INK, lw=2.5)
    axes[2].scatter([cfg.avoidance], [1/(1+np.exp(cfg.avoidance))], c=ORANGE, s=80, zorder=4)
    axes[2].set(xlabel='Avoidance parameter', ylabel='Probability of choosing an open arm', ylim=(0, .55))
    axes[2].set_title('Behavioral choice rule', loc='left', fontsize=12)
    fig.text(.065, .06, 'Left: dots are neurons, dark lines are group medians; no animal-level inference.\n'
             'Middle: expected rates at identical visited open-arm positions, with only gain changed (model ground truth).\n'
             'Right: the configured choice probability is distinct from the fraction of time spent in open arms.', color=GRAY, fontsize=10)
    finish(fig, out, '03_arm_effects')


def decoding(data, result, summary, cfg, out):
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.8, 1, 1], left=.075, right=.96, bottom=.1, top=.86, hspace=.55, wspace=.28)
    title(fig, 'Can spikes tell us where the mouse is?',
          f'Train: first {cfg.train_fraction:.0%} of session  /  Test: final {1-cfg.train_fraction:.0%}  /  {cfg.decode_window_s*1000:g} ms windows')
    ax = fig.add_subplot(gs[0, 0]); maze(ax)
    clip = result['t'] < result['t'][0] + 30
    ax.plot(*result['truth'][clip].T, c=INK, lw=2, label='True path')
    ax.scatter(*result['pred'][clip].T, c=ORANGE, s=12, alpha=.65, label='Decoded position')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.set_title('First 30 seconds of held-out data', loc='left', fontsize=12)
    ax = fig.add_subplot(gs[0, 1])
    for (name, values), color in zip(result['errors'].items(), [TEAL, ORANGE, GRAY]):
        sns.ecdfplot(x=values, ax=ax, color=color, lw=2, label=f'{name} · median {np.median(values):.1f} cm')
    ax.set(xlabel='Euclidean position error (cm)', ylabel='Fraction of test windows', xlim=(0, 66))
    ax.legend(frameon=False, loc='lower right', fontsize=9)
    ax.set_title('Accuracy against two baselines', loc='left', fontsize=12)
    for coord in range(2):
        ax = fig.add_subplot(gs[coord+1, :])
        ax.fill_between(result['t'], -35, 35, where=result['zone'] == 1, color=ORANGE, alpha=.14, step='mid')
        ax.plot(result['t'], result['truth'][:, coord], c=INK, lw=1.5, label='True')
        ax.plot(result['t'], result['pred'][:, coord], c=TEAL, lw=.8, alpha=.7, label='Decoded')
        ax.set(xlim=(result['t'][0], result['t'][-1]), ylim=(-35, 35), ylabel=f'{"xy"[coord]} position (cm)', xlabel='Session time (s)' if coord else '')
        if coord == 0: ax.legend(loc='upper right', frameon=False, ncol=2)
    fig.text(.075, .025, f'Orange shading: true open-arm visits. Test-bin training coverage: {summary["test_spatial_coverage"]:.1%}. '
             f'Within 10 cm: {summary["within_10_cm"]:.1%}.\n'
             'All test windows are scored, including positions without sufficient training occupancy. Baselines use the same windows.',
             fontsize=10, color=GRAY)
    finish(fig, out, '04_decoding')


def replay(data, cfg, out):
    """60-second clip at 5x speed; frames show samples every .4 simulated seconds."""
    opens = data['entries'][data['entries'][:, 1] < 2]
    start = max(0, opens[0, 0] - 20) if len(opens) else 0
    start = min(start, max(0, cfg.duration_s - 60))
    end = min(start + 60, cfg.duration_s - cfg.dt_s)
    frames = np.arange(start, end, .4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1.25]})
    fig.subplots_adjust(left=.08, right=.97, top=.8, bottom=.16, wspace=.4)
    fig.suptitle('Mouse replay  /  synthetic CA1', x=.07, ha='left', fontsize=19, weight='bold')
    maze(axes[0], labels=True)
    trail, = axes[0].plot([], [], color=INK, alpha=.35, lw=1.3)
    mouse, = axes[0].plot([], [], 'o', color=ORANGE, markersize=9, markeredgecolor=INK)
    dots = axes[1].scatter([], [], s=4, color=TEAL, linewidths=0)
    axes[1].set(xlim=(-5, 0), ylim=(-1, cfg.n_cells), xlabel='Seconds before current time', ylabel='Neuron ID', title='Spikes in the last 5 seconds')
    clock = fig.text(.07, .86, '', color=GRAY, fontsize=11)
    fig.text(.07, .04, '5× playback  ·  horizontal arms open / vertical arms closed  ·  trail = last 5 s', color=GRAY, fontsize=10)

    def update(t):
        idx = min(int(t / cfg.dt_s), len(data['xy']) - 1)
        segment = data['xy'][max(0, idx-round(5/cfg.dt_s)):idx+1]
        trail.set_data(segment[:, 0], segment[:, 1])
        mouse.set_data([data['xy'][idx, 0]], [data['xy'][idx, 1]])
        lo, hi = np.searchsorted(data['spike_times'], [max(0, t-5), t])
        dots.set_offsets(np.column_stack([data['spike_times'][lo:hi]-t, data['spike_units'][lo:hi]]))
        clock.set_text(f'Session time {t:6.1f} s  /  { ["center", "open arm", "closed arm"][zone(data["xy"][idx])] }')
        return trail, mouse, dots, clock

    anim = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    anim.save(out / '05_mouse_replay.gif', writer=PillowWriter(fps=12.5), dpi=85)
    plt.close(fig)


def all_figures(data, maps, result, summary, cfg, out, animate=False):
    style()
    overview(data, maps, result, summary, cfg, out)
    place_fields(data, maps, cfg, out)
    arm_effects(data, cfg, out)
    decoding(data, result, summary, cfg, out)
    if animate:
        replay(data, cfg, out)
