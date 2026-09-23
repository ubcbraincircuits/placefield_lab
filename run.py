"""Run from any directory: python path/to/run.py --animate"""
import argparse
import json
from pathlib import Path
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
from model import Config, simulate, estimate_maps, decode, summarize, zone
from plots import all_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('results'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--avoidance', type=float)
    parser.add_argument('--open-gain', type=float)
    parser.add_argument('--duration', dest='duration_s', type=float, help='Session duration in seconds (30–3600).')
    parser.add_argument('--cells', dest='n_cells', type=int, help='Number of synthetic neurons (4–500).')
    parser.add_argument('--animate', action='store_true', help='Also render a 60-second GIF clip.')
    args = parser.parse_args()
    params = json.loads(args.config.read_text())
    for key in ['seed', 'avoidance', 'open_gain', 'duration_s', 'n_cells']:
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    cfg = Config(**params)
    cfg.validate()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print('Simulating mouse trajectory and sorted spike trains...', flush=True)
    data = simulate(cfg)
    maps = estimate_maps(data['bins'], data['counts'], data['grid'], cfg)
    result = decode(data, cfg)
    summary = summarize(data, result, cfg)
    summary['environment'] = dict(python=platform.python_version(),
        packages={p: version(p) for p in ['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'pillow']})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'config_used.json').write_text(json.dumps(summary['config'], indent=2), encoding='utf-8')
    # Complete numeric output without pickle objects; load using np.load(..., allow_pickle=False).
    np.savez_compressed(out / 'session.npz', time_s=data['t'], position_cm=data['xy'],
        spike_counts=data['counts'], spike_times_s=data['spike_times'], spike_unit_ids=data['spike_units'],
        spatial_bin=data['bins'], grid_xy_cm=data['grid']['xy'], occupancy_s=maps['occupancy'],
        estimated_rates_hz=maps['rates'], supported_bins=maps['supported'],
        cell_centers_cm=data['cells']['centers'], cell_sigma_cm=data['cells']['sigma'],
        cell_peak_hz=data['cells']['peak'], cell_baseline_hz=data['cells']['baseline'],
        cell_open_gain=data['cells']['gain'], posterior=result['posterior'],
        posterior_spatial_bins=result['valid'], train_rates_hz=result['train']['rates'],
        train_occupancy_s=result['train']['occupancy'], train_split_sample=result['split'])
    pd.DataFrame({'time_s': data['t'], 'x_cm': data['xy'][:, 0], 'y_cm': data['xy'][:, 1],
                  'zone': np.array(['center', 'open', 'closed'])[zone(data['xy'])]}).to_csv(out / 'trajectory.csv', index=False)
    pd.DataFrame({'time_s': data['spike_times'], 'unit_id': data['spike_units']}).to_csv(out / 'spikes.csv', index=False)
    pd.DataFrame(data['entries'], columns=['entry_time_s', 'arm_id', 'target_depth_cm']).to_csv(out / 'entries.csv', index=False)
    pd.DataFrame({'unit_id': np.arange(cfg.n_cells), 'field_x_cm': data['cells']['centers'][:, 0],
                  'field_y_cm': data['cells']['centers'][:, 1], 'sigma_cm': data['cells']['sigma'],
                  'peak_hz': data['cells']['peak'], 'baseline_hz': data['cells']['baseline'],
                  'open_gain': data['cells']['gain']}).to_csv(out / 'cells.csv', index=False)
    table = pd.DataFrame({'time_s': result['t'], 'true_x_cm': result['truth'][:, 0],
        'true_y_cm': result['truth'][:, 1], 'decoded_x_cm': result['pred'][:, 0],
        'decoded_y_cm': result['pred'][:, 1], 'true_zone': np.array(['center', 'open', 'closed'])[result['zone']],
        'training_supported': result['supported']})
    for key, value in result['errors'].items():
        table[key.lower().replace(' ', '_') + '_error_cm'] = value
    table.to_csv(out / 'decoding.csv', index=False)
    print('Rendering figures' + (' and replay...' if args.animate else '...'), flush=True)
    all_figures(data, maps, result, summary, cfg, out, animate=args.animate)
    print(json.dumps({k: summary[k] for k in ['open_fraction_of_arm_time', 'median_error_cm', 'test_spatial_coverage']}, indent=2))
    print(f'Done: {out}', flush=True)


if __name__ == '__main__':
    main()
