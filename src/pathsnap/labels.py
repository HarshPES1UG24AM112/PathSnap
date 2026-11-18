"""Derive an observation-level reference arc from the supplied route (training/evaluation only).

The ``.route`` file is a chronological arc sequence, not one arc per GPS row. We align the
two sequences with a monotone dynamic programme: observation i is assigned route position
j(i), with j non-decreasing in i, minimising sum_i min(d(i, route[j(i)]), CAP) where d is the
point-to-segment distance in metres. Monotonicity uses the route's order, so loops and
U-turns that revisit an arc are handled correctly, unlike a naive nearest-route-arc rule.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment

CAP = 100.0


def align_route(xy: np.ndarray, route: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Return (route_pos[i], label_arc[i], label_dist[i])."""
    n, m = len(xy), len(route)
    ra, rb = A[route], B[route]
    back = np.empty((n, m), dtype=np.int32)
    prev = None
    dists = np.empty((n, m), dtype=np.float32) if n * m < 4e7 else None
    for i in range(n):
        d, _ = point_segment(xy[i][None, :], ra, rb)
        c = np.minimum(d, CAP)
        if dists is not None:
            dists[i] = d
        if prev is None:
            cur = c
            back[i] = np.arange(m)
        else:
            # running argmin of previous row: best k <= j
            run_min = np.minimum.accumulate(prev)
            is_new = np.r_[True, prev[1:] < run_min[:-1]]
            arg = np.maximum.accumulate(np.where(is_new, np.arange(m), 0))
            back[i] = arg
            cur = c + run_min
        prev = cur
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    label = route[pos]
    if dists is not None:
        ldist = dists[np.arange(n), pos].astype(float)
    else:
        ldist, _ = point_segment(xy, A[label], B[label])
    return pos, label, ldist
