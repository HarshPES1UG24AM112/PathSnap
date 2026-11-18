"""End-to-end check against a RUNNING API over HTTP (as Laptop 2 would call it): every trip.
Usage: python scripts/11_e2e_api.py http://localhost:8000
Output: results/e2e_api_results.csv
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")


def call(path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Origin": "http://laptop2.local:5173"},
                                 method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status, dict(r.headers), json.loads(r.read())


st, hdr, h = call("/api/health")
print("health", st, h)
print("CORS header:", hdr.get("access-control-allow-origin"))
_, _, trips = call("/api/trips")
rows = []
for t in trips["trips"]:
    t0 = time.time()
    st, _, p = call("/api/predict", {"trip_id": t["trip_id"]})
    m = p["metrics"] or {}
    rows.append(dict(trip_id=t["trip_id"], status=st, n_obs=p["n_observations"], route_arcs=len(p["route_arc_ids"]),
                     breaks=p["unresolved_breaks"], wall_s=time.time() - t0, server_ms=sum(p["timings_ms"].values()), **m))
    print(rows[-1]["trip_id"], st, f"hmm_top1={m.get('hmm_top1_accuracy', float('nan')):.4f}", f"{rows[-1]['wall_s']:.1f}s")
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "e2e_api_results.csv", index=False)
print(f"{(df.status == 200).sum()}/{len(df)} trips OK; obs-weighted HMM top1 = "
      f"{(df.hmm_top1_accuracy * df.n_obs).sum() / df.n_obs.sum():.4f}")
