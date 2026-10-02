"""E0 label agreement on identical float32 inputs; requires real Redis and artifacts."""
import argparse
import uuid
import numpy as np
import redis
import config as cfg
from models import load_dash_harness
from engine import ProposedStatefulEngine, RedisFetchBaselineEngine, InMemoryRecomputeEngine


def check_paths(data, wrapper, client):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != len(cfg.FEATURE_COLS) or not len(data):
        raise ValueError("expected nonempty canonical feature matrix")
    symbols = [f"e0:{uuid.uuid4().hex}:{x}" for x in ('A', 'B1')]
    keys = [f"dash:state:{s}" for s in symbols]
    engines = [ProposedStatefulEngine(wrapper, client), RedisFetchBaselineEngine(wrapper, client)]
    memory = InMemoryRecomputeEngine(wrapper, data)
    mismatches = []
    try:
        for i, feat in enumerate(data):
            labels = [e.process_tick(s, feat)['yhat'] for e, s in zip(engines, symbols)]
            labels.append(memory.process_tick(i)['yhat'])
            if len(set(labels)) != 1:
                mismatches.append((i, labels))
    finally:
        client.delete(*keys)
    if mismatches:
        raise AssertionError(f"label mismatches: {mismatches[:10]}; count={len(mismatches)}")
    return dict(n=len(data), label_agreement=1.0)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('features', help='N x d .npy, canonical feature order')
    p.add_argument('--model', default='Logistic', choices=['Logistic', 'RandomForest', 'XGBoost'])
    a = p.parse_args()
    print(check_paths(np.load(a.features, allow_pickle=False), load_dash_harness(a.model),
        redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)))
