"""Synthetic mouse CA1, elevated plus maze, and held-out Poisson decoding.

Lengths are cm, time seconds, and rates Hz. No recorded animal data are used.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.special import logsumexp


@dataclass
class Config:
    seed: int = 42
    duration_s: float = 600.
    dt_s: float = .02
    n_cells: int = 100
    avoidance: float = 1.6
    open_modulated_fraction: float = .25
    open_gain: float = 1.8
    decode_window_s: float = .2
    train_fraction: float = .7
    min_occupancy_s: float = .3
    smoothing_cm: float = 4.5

    def validate(self):
        if not 30 <= self.duration_s <= 3600:
            raise ValueError('duration_s must be between 30 and 3600.')
        if not .005 <= self.dt_s <= .05:
            raise ValueError('dt_s must be between .005 and .05.')
        if not 4 <= self.n_cells <= 500:
            raise ValueError('n_cells must be between 4 and 500.')
        if not 0 <= self.avoidance <= 5:
            raise ValueError('avoidance must be between 0 and 5.')
        if not 0 <= self.open_modulated_fraction <= 1 or not 0 < self.open_gain <= 5:
            raise ValueError('Invalid neural open-arm modulation parameters.')
        if not .4 <= self.train_fraction <= .85:
            raise ValueError('train_fraction must be between .4 and .85.')
        if not self.dt_s <= self.decode_window_s <= 2:
            raise ValueError('decode_window_s must be between dt_s and 2.')
        if not np.isclose(self.decode_window_s / self.dt_s, round(self.decode_window_s / self.dt_s)):
            raise ValueError('decode_window_s must be a multiple of dt_s.')
        if self.min_occupancy_s <= 0 or self.smoothing_cm <= 0:
            raise ValueError('Occupancy threshold and smoothing must be positive.')
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Parameters must be finite.')


def zone(xy):
    """0=center, 1=open (east/west), 2=closed (north/south)."""
    xy = np.asarray(xy)
    return np.where(np.abs(xy[..., 0]) > 3, 1,
                    np.where(np.abs(xy[..., 1]) > 3, 2, 0))


def inside(xy):
    xy = np.abs(xy)
    return (xy.max(axis=-1) <= 33) & (xy.min(axis=-1) <= 3)


def maze_grid():
    edges = np.arange(-33., 34., 3.)
    centers = (edges[:-1] + edges[1:]) / 2
    xx, yy = np.meshgrid(centers, centers)
    mask = (np.abs(xx) < 3) | (np.abs(yy) < 3)
    xy = np.column_stack([xx[mask], yy[mask]])
    # Four-neighbor graph prevents smoothing across off-maze corners.
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    graph = csr_matrix(np.where(np.isclose(dist, 3), 3., 0.))
    path_dist = shortest_path(graph, directed=False)
    lookup = np.full(mask.shape, -1, dtype=int)
    lookup[mask] = np.arange(mask.sum())
    return dict(edges=edges, xy=xy, mask=mask, lookup=lookup, path_dist=path_dist)


def bin_position(xy, grid):
    ij = np.floor((xy + 33) / 3).astype(int).clip(0, 21)
    bins = grid['lookup'][ij[:, 1], ij[:, 0]]
    if np.any(bins < 0):
        raise ValueError('Trajectory left the maze.')
    return bins


def behavior(cfg, rng):
    """Smooth center-to-arm excursions; selection bias is explicitly imposed."""
    n = int(round(cfg.duration_s / cfg.dt_s))
    pieces, entries, elapsed = [], [], 0
    vectors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    weights = np.exp([-cfg.avoidance, -cfg.avoidance, 0, 0])
    weights /= weights.sum()

    def append(segment):
        nonlocal elapsed
        pieces.append(segment)
        elapsed += len(segment)

    while elapsed < n:
        append(np.zeros((max(1, round(rng.uniform(.3, 1.5) / cfg.dt_s)), 2)))
        arm = int(rng.choice(4, p=weights))
        is_open = arm < 2
        # Same reach and dwell distributions when avoidance == 0.
        depth = 8 + 24 * rng.beta(3, 1 + (cfg.avoidance if is_open else 0))
        axis = vectors[arm]
        lateral = np.array([-axis[1], axis[0]])
        speed = rng.uniform(8, 15)
        steps = max(2, round(depth / speed * 1.5 / cfg.dt_s))
        u = np.linspace(0, 1, steps + 1)[1:]
        progress = 3 * u**2 - 2 * u**3
        bend = rng.uniform(-2.3, 2.3)
        outbound = depth * progress[:, None] * axis + bend * np.sin(np.pi * progress)[:, None] * lateral
        crossings = np.flatnonzero(zone(outbound) != 0)
        entry_sample = elapsed + int(crossings[0])
        if entry_sample < n:
            entries.append((entry_sample * cfg.dt_s, arm, depth))
        append(outbound)
        dwell = rng.uniform(.4, 2.4) / (1 + .5 * cfg.avoidance * is_open)
        append(np.tile(depth * axis, (max(1, round(dwell / cfg.dt_s)), 1)))
        inbound = depth * (1 - progress[:, None]) * axis - bend * np.sin(np.pi * progress)[:, None] * lateral
        append(inbound)
    xy = np.concatenate(pieces)[:n]
    return xy, np.asarray(entries).reshape(-1, 3)


def make_cells(cfg, grid, rng):
    centers = grid['xy'][rng.integers(len(grid['xy']), size=cfg.n_cells)]
    gain = np.ones(cfg.n_cells)
    selected = rng.choice(cfg.n_cells, round(cfg.n_cells * cfg.open_modulated_fraction), replace=False)
    gain[selected] = cfg.open_gain
    return dict(centers=centers, sigma=rng.uniform(5, 10, cfg.n_cells),
                peak=rng.uniform(8, 18, cfg.n_cells),
                baseline=rng.uniform(.05, .3, cfg.n_cells), gain=gain)


def firing_rate(xy, cells, modulation=True):
    d2 = ((xy[:, None] - cells['centers'][None])**2).sum(axis=2)
    rate = cells['baseline'] + cells['peak'] * np.exp(-d2 / (2 * cells['sigma']**2))
    if modulation:
        rate *= np.where(zone(xy)[:, None] == 1, cells['gain'], 1.)
    return rate.astype(np.float32)


def simulate(cfg):
    cfg.validate()
    behavior_seed, cell_seed, spike_seed = np.random.SeedSequence(cfg.seed).spawn(3)
    grid = maze_grid()
    xy, entries = behavior(cfg, np.random.default_rng(behavior_seed))
    cells = make_cells(cfg, grid, np.random.default_rng(cell_seed))
    rate = firing_rate(xy, cells)
    rng = np.random.default_rng(spike_seed)
    counts = rng.poisson(rate * cfg.dt_s).astype(np.uint16)
    # Conditional on counts, uniform spike times are exact for piecewise-constant Poisson rates.
    samples, units = np.nonzero(counts)
    repeats = counts[samples, units]
    samples, units = np.repeat(samples, repeats), np.repeat(units, repeats)
    times = (samples + rng.random(len(samples))) * cfg.dt_s
    order = np.argsort(times)
    return dict(grid=grid, xy=xy, entries=entries, cells=cells, counts=counts,
                spike_times=times[order], spike_units=units[order],
                t=np.arange(len(xy)) * cfg.dt_s, bins=bin_position(xy, grid))


def estimate_maps(bins, counts, grid, cfg):
    """Smooth spike totals and occupancy separately, then divide."""
    occupancy = np.bincount(bins, minlength=len(grid['xy'])) * cfg.dt_s
    totals = np.zeros((len(occupancy), counts.shape[1]))
    np.add.at(totals, bins, counts)
    kernel = np.exp(-.5 * (grid['path_dist'] / cfg.smoothing_cm)**2)
    kernel[grid['path_dist'] > 3 * cfg.smoothing_cm] = 0
    kernel /= kernel.sum(axis=1, keepdims=True)
    smooth_occ = kernel @ occupancy
    global_rate = counts.sum(axis=0) / (len(counts) * cfg.dt_s)
    # Small 0.2-second shrinkage toward training mean prevents log(0).
    rates = (kernel @ totals + .2 * global_rate) / (smooth_occ[:, None] + .2)
    return dict(occupancy=occupancy, totals=totals, rates=np.maximum(rates, 1e-6),
                supported=occupancy >= cfg.min_occupancy_s)


def posterior(counts, rates, prior, window_s):
    ll = counts @ np.log(rates).T - window_s * rates.sum(axis=1) + np.log(prior)
    return np.exp(ll - logsumexp(ll, axis=1, keepdims=True))


def decode(data, cfg):
    size = round(cfg.decode_window_s / cfg.dt_s)
    split = (int(len(data['xy']) * cfg.train_fraction) // size) * size
    train = estimate_maps(data['bins'][:split], data['counts'][:split], data['grid'], cfg)
    valid = np.flatnonzero(train['supported'])
    if not len(valid):
        raise ValueError('No adequately sampled training bins; increase duration or lower occupancy threshold.')
    n_windows = (len(data['xy']) - split) // size
    end = split + n_windows * size
    counts = data['counts'][split:end].reshape(n_windows, size, cfg.n_cells).sum(axis=1)
    # Central observed sample is always on the maze (a coordinate mean might not be).
    mid = np.arange(split + size // 2, end, size)
    truth = data['xy'][mid]
    rates = train['rates'][valid]
    prior = train['occupancy'][valid] / train['occupancy'][valid].sum()
    post = posterior(counts, rates, prior, cfg.decode_window_s)
    pred = data['grid']['xy'][valid[post.argmax(axis=1)]]
    perm = np.random.default_rng(cfg.seed + 1000).permutation(cfg.n_cells)
    shuffled_post = posterior(counts[:, perm], rates, prior, cfg.decode_window_s)
    shuffled = data['grid']['xy'][valid[shuffled_post.argmax(axis=1)]]
    baseline = np.tile(data['grid']['xy'][valid[prior.argmax()]], (n_windows, 1))
    errors = {name: np.linalg.norm(p - truth, axis=1)
              for name, p in [('Decoded', pred), ('Unit shuffle', shuffled), ('Occupancy only', baseline)]}
    return dict(train=train, split=split, t=data['t'][mid], truth=truth, pred=pred,
                posterior=post, valid=valid, errors=errors, zone=zone(truth),
                supported=train['supported'][data['bins'][mid]])


def summarize(data, result, cfg):
    zones = zone(data['xy'])
    secs = np.bincount(zones, minlength=3) * cfg.dt_s
    entries = data['entries']
    out = dict(config=asdict(cfg), total_spikes=int(data['counts'].sum()),
               total_entries=len(entries), open_entries=int((entries[:, 1] < 2).sum()),
               occupancy_seconds=dict(zip(['center', 'open', 'closed'], secs.tolist())),
               open_fraction_of_arm_time=float(secs[1] / (secs[1] + secs[2])),
               configured_open_choice_probability=float(1 / (1 + np.exp(cfg.avoidance))),
               train_seconds=result['split'] * cfg.dt_s,
               test_windows=len(result['t']),
               test_spatial_coverage=float(result['supported'].mean()),
               median_error_cm={k: float(np.median(v)) for k, v in result['errors'].items()},
               within_10_cm=float(np.mean(result['errors']['Decoded'] <= 10)))
    out['decoding_by_zone'] = {}
    for z, name in enumerate(['center', 'open', 'closed']):
        use = result['zone'] == z
        out['decoding_by_zone'][name] = dict(n_windows=int(use.sum()),
            median_error_cm=float(np.median(result['errors']['Decoded'][use])) if use.any() else None)
    return out
