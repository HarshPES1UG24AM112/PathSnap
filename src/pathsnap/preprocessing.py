"""GPS track cleaning and kinematic descriptors (label-free, no learned parameters).

Decisions (see report §2):
* drop rows with NaN / out-of-range / (0,0) coordinates;
* sort by timestamp and drop exact duplicate timestamps (keep first);
* keep every remaining observation (each must receive a predicted arc) but flag
  - ``is_jump``: implied speed > MAX_SPEED to BOTH neighbours (isolated GPS spike),
  - ``gap_before``: dt > GAP_S seconds or step > GAP_M metres (segment break for the HMM).
* heading from a centred displacement (i-k -> i+k) growing k until >= MIN_MOVE metres,
  so stationary jitter does not produce random headings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import LocalProjection, bearing_xy

MAX_SPEED = 60.0  # m/s (216 km/h) - physically implausible for road vehicles in this data
GAP_S = 30.0
GAP_M = 500.0
MIN_MOVE = 4.0


@dataclass
class CleanTrack:
    df: pd.DataFrame  # columns: obs_idx (row in raw file), lon, lat, t, x, y, dt, step, speed, heading, heading_valid, is_jump, gap_before
    proj: LocalProjection
    n_raw: int
    n_dropped: int


def clean_track(raw: pd.DataFrame, proj: LocalProjection | None = None) -> CleanTrack:
    n_raw = len(raw)
    df = raw.copy()
    df["obs_idx"] = np.arange(n_raw)
    ok = df[["lon", "lat", "t"]].notna().all(axis=1)
    ok &= df.lat.between(-90, 90) & df.lon.between(-180, 180) & ~((df.lat == 0) & (df.lon == 0))
    df = df[ok].sort_values("t", kind="stable").drop_duplicates("t", keep="first").reset_index(drop=True)
    if proj is None:
        proj = LocalProjection.for_points(df.lon.to_numpy(), df.lat.to_numpy()) if len(df) else LocalProjection(0, 0)
    xy = proj.to_xy(df.lon.to_numpy(), df.lat.to_numpy())
    df["x"], df["y"] = xy[:, 0], xy[:, 1]
    n = len(df)
    t = df.t.to_numpy()
    dt = np.r_[np.nan, np.diff(t)]
    step = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    speed = step / dt
    df["dt"], df["step"], df["speed"] = dt, step, speed
    sp_prev = speed
    sp_next = np.r_[speed[1:], np.nan]
    df["is_jump"] = (np.nan_to_num(sp_prev, nan=0) > MAX_SPEED) & (np.nan_to_num(sp_next, nan=0) > MAX_SPEED)
    df["gap_before"] = (np.nan_to_num(dt, nan=0) > GAP_S) | (np.nan_to_num(step, nan=0) > GAP_M)
    # heading with adaptive centred window
    heading = np.full(n, np.nan)
    for k in (1, 2, 3, 5, 8):
        need = np.isnan(heading)
        if not need.any():
            break
        i = np.arange(n)
        lo, hi = np.clip(i - k, 0, n - 1), np.clip(i + k, 0, n - 1)
        d = xy[hi] - xy[lo]
        mv = np.hypot(d[:, 0], d[:, 1])
        good = need & (mv >= MIN_MOVE)
        heading[good] = bearing_xy(d[good, 0], d[good, 1])
    df["heading_valid"] = ~np.isnan(heading)
    df["heading"] = heading
    df["speed"] = df["speed"].fillna(0).clip(upper=MAX_SPEED)
    df["dt"] = df["dt"].fillna(1.0)
    df["step"] = df["step"].fillna(0.0)
    return CleanTrack(df, proj, n_raw, n_raw - n)
