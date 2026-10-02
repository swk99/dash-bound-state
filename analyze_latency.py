"""E6 deadline sweeps and IID BCa quantile CIs. IID inference is explicitly conditional."""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def summarize(values, deadlines, *, resamples=10000, seed=20260928):
    x = np.asarray(values, dtype=float)
    if len(x) < 3 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('need >=3 finite nonnegative latency observations')
    rows = []
    for q in (.5, .9, .95, .99):
        if np.ptp(x) == 0:
            lo = hi = float(x[0])
            status = 'constant_sample'
        else:
            with warnings.catch_warnings(record=True) as caught:
                result = bootstrap((x,), lambda a, axis: np.quantile(a, q, axis=axis),
                    vectorized=True, axis=0, n_resamples=resamples, batch=32,
                    method='BCa', random_state=np.random.default_rng(seed))
            lo, hi = result.confidence_interval
            status = 'ok' if np.isfinite([lo, hi]).all() and not caught else 'inspect_degenerate_ci'
        rows.append(dict(metric=f'p{q*100:g}_ms', value=np.quantile(x, q),
            ci_low=lo, ci_high=hi, n=len(x), method='IID_BCa', status=status))
    for deadline in deadlines:
        if deadline < 0 or not np.isfinite(deadline):
            raise ValueError('finite nonnegative deadlines required')
        rows.append(dict(metric='deadline_coverage', deadline_ms=deadline,
            value=np.mean(x <= deadline), n=len(x), method='empirical'))
    return rows

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--latency-column', default='total_ms')
    p.add_argument('--group-by', nargs='*', default=['engine'])
    p.add_argument('--deadlines', nargs='+', type=float, default=[1, 5, 10, 20, 50, 100, 1000])
    p.add_argument('--resamples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20260928)
    p.add_argument('--output', default='results/latency_statistics.csv')
    a = p.parse_args()
    df = pd.read_csv(a.csv)
    rows = []
    groups = df.groupby(a.group_by, dropna=False, sort=False) if a.group_by else [((), df)]
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        metadata = dict(zip(a.group_by, key))
        rows.extend(dict(metadata, **r) for r in summarize(group[a.latency_column], a.deadlines,
            resamples=a.resamples, seed=a.seed))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output, index=False)
