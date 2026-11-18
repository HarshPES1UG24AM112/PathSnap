"""Build candidate/feature tables for train and student_test partitions.

Outputs (data/processed/, git-ignored, fully reproducible):
  {partition}_cands.parquet : one row per (trip, obs, candidate arc) with features + label
  {partition}_obs.parquet   : one row per cleaned observation with reference arc + kinematics
and results/eda/candidate_stats.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import io  # noqa: E402
from pathsnap.pipeline import prepare_labelled  # noqa: E402

OUT = io.ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def one(tid, part):
    t0 = time.time()
    pt = prepare_labelled(tid, part)
    o = pt.ct.df.copy()
    o.insert(0, "trip_id", tid)
    o["obs"] = np.arange(len(o))
    hit = pt.X.groupby("obs").label.max().reindex(o.obs, fill_value=0).to_numpy()
    o["ref_in_cands"] = hit.astype(np.int8)
    o["n_cands"] = pt.X.groupby("obs").size().reindex(o.obs, fill_value=0).to_numpy()
    stats = dict(partition=part, trip_id=tid, n_obs=len(o), n_dropped=pt.ct.n_dropped, n_rows=len(pt.X),
                 cands_mean=o.n_cands.mean(), empty_cands=int((o.n_cands == 0).sum()),
                 miss_rate=1 - hit.mean(), ref_dist_p95=float(np.percentile(o.ref_dist, 95)),
                 nearest_acc=float(pt.X[pt.X.is_nearest == 1].label.sum() / len(o)), secs=time.time() - t0)
    return pt.X, o, stats


if __name__ == "__main__":
    allstats = []
    for part in io.PARTITIONS:
        res = Parallel(n_jobs=12)(delayed(one)(t, part) for t in io.list_trips(part))
        X = pd.concat([r[0] for r in res], ignore_index=True)
        O = pd.concat([r[1] for r in res], ignore_index=True)
        X.to_parquet(OUT / f"{part}_cands.parquet", index=False)
        O.to_parquet(OUT / f"{part}_obs.parquet", index=False)
        allstats += [r[2] for r in res]
        print(part, "rows", len(X), "obs", len(O), "pos rate", round(X.label.mean(), 4),
              "miss", round(1 - O.ref_in_cands.mean(), 4), "nearest-acc", round(X[X.is_nearest == 1].label.sum() / len(O), 4))
    s = pd.DataFrame(allstats)
    s.to_csv(io.ROOT / "results" / "eda" / "candidate_stats.csv", index=False)
    print(s.groupby("partition")[["cands_mean", "miss_rate", "nearest_acc", "empty_cands", "secs"]].agg(["mean", "max"]))
