"""Label-free probability operations used by frozen routing."""
import numpy as np


def margin(p):
    top = np.partition(p, -2, axis=1)[:, -2:]
    return top.max(1) - top.min(1)


def js(p, q):
    mid = .5 * (p + q)
    logmid = np.log(mid.clip(1e-12))
    return .5 * ((p * (np.log(p.clip(1e-12)) - logmid)).sum(1) +
                 (q * (np.log(q.clip(1e-12)) - logmid)).sum(1))
