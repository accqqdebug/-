"""Macro-averaged paired ESOT500 partition baseline with sequence bootstrap intervals."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(rows, seed=20260927, replicates=10000, policies=('uniform', 'jitter', 'fixed_count')):
    auc = np.asarray([[row['scores'][policy]['success_auc_21'] for policy in policies] for row in rows])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(rows), size=(replicates, len(rows)))
    means = auc[samples].mean(axis=1)
    result = {'sequences': len(rows), 'bootstrap_seed': seed,
              'bootstrap_replicates': replicates, 'macro_auc': {}, 'paired_auc_delta': {},
              'mean_prediction_iou': {}}
    for i, policy in enumerate(policies):
        result['macro_auc'][policy] = {
            'mean': float(auc[:, i].mean()),
            'ci95': np.quantile(means[:, i], [0.025, 0.975]).tolist(),
        }
    for i, policy in enumerate(policies[1:], 1):
        delta = auc[:, i] - auc[:, 0]
        boot = means[:, i] - means[:, 0]
        result['paired_auc_delta'][policy] = {
            'mean': float(delta.mean()),
            'ci95': np.quantile(boot, [0.025, 0.975]).tolist(),
            'positive_sequences': int((delta > 0).sum()),
            'negative_sequences': int((delta < 0).sum()),
        }
        result['mean_prediction_iou'][policy] = float(np.mean(
            [row['paired'][policy]['mean_prediction_iou'] for row in rows]))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policies', nargs='+', default=['uniform', 'jitter', 'fixed_count'])
    args = p.parse_args()
    rows = json.loads(args.input.read_text())
    result = summarize(rows, policies=tuple(args.policies))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
