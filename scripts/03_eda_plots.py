"""EDA step 2: figures + Folium maps (partitions: train + student_test; never evaluator data)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import io  # noqa: E402
from pathsnap.network import ProjectedNetwork  # noqa: E402
from pathsnap.preprocessing import clean_track  # noqa: E402
from pathsnap.visualization import trip_map  # noqa: E402

FIG = io.ROOT / "results" / "figures" / "eda"
MAPS = io.ROOT / "results" / "maps"
FIG.mkdir(parents=True, exist_ok=True)
MAPS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
C = {"train": "#2f6db5", "student_test": "#e08a1e"}

inv = pd.read_csv(io.ROOT / "results/eda/trip_inventory.csv", dtype={"trip_id": str})
cs = pd.read_csv(io.ROOT / "results/eda/candidate_stats.csv", dtype={"trip_id": str})
O = pd.concat([pd.read_parquet(io.ROOT / f"data/processed/{p}_obs.parquet").assign(partition=p) for p in io.PARTITIONS])

# 1 observations per trip
fig, ax = plt.subplots(figsize=(6, 3.4))
for p in io.PARTITIONS:
    ax.hist(inv[inv.partition == p].n_obs, bins=np.logspace(2.3, 4.4, 25), alpha=0.75, label=p, color=C[p])
ax.set_xscale("log"); ax.set_xlabel("GPS observations per trip"); ax.set_ylabel("trips"); ax.legend(); ax.set_title("Trip length distribution")
fig.tight_layout(); fig.savefig(FIG / "obs_per_trip.png"); plt.close(fig)

# 2 sampling interval & speed
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
dt = O.dt[O.obs > 0]
vals, cnt = np.unique(np.clip(dt, 0, 31).round(), return_counts=True)
axs[0].bar(vals, cnt, color="#2f6db5"); axs[0].set_yscale("log"); axs[0].set_xlabel("Δt between observations (s, clipped at 31)"); axs[0].set_ylabel("count (log)"); axs[0].set_title("Sampling intervals (train+student_test)")
axs[1].hist(O.speed * 3.6, bins=80, color="#2f6db5"); axs[1].set_yscale("log"); axs[1].set_xlabel("implied speed (km/h, clipped 216)"); axs[1].set_title("Speed between fixes")
fig.tight_layout(); fig.savefig(FIG / "dt_speed.png"); plt.close(fig)

# 3 GPS -> reference-arc distance (label noise)
fig, ax = plt.subplots(figsize=(6, 3.4))
for p in io.PARTITIONS:
    ax.hist(np.clip(O[O.partition == p].ref_dist, 0, 100), bins=100, density=True, alpha=0.6, label=p, color=C[p])
ax.axvline(50, color="k", ls="--", lw=1); ax.text(51, ax.get_ylim()[1] * 0.8, "candidate radius 50 m")
ax.set_xlabel("distance GPS → aligned reference arc (m, clipped 100)"); ax.set_ylabel("density"); ax.legend(); ax.set_title("GPS noise relative to reference route")
fig.tight_layout(); fig.savefig(FIG / "gps_to_reference.png"); plt.close(fig)

# 4 network size / connectivity
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for p in io.PARTITIONS:
    d = inv[inv.partition == p]
    axs[0].scatter(d.n_nodes, d.n_arcs, s=14, label=p, color=C[p])
    axs[1].scatter(d.n_nodes, d.weak_components, s=14, label=p, color=C[p])
axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("nodes"); axs[0].set_ylabel("arcs"); axs[0].set_title("Road-network size per trip")
axs[1].set_xscale("log"); axs[1].set_xlabel("nodes"); axs[1].set_ylabel("weakly connected components"); axs[1].set_title("Network fragmentation"); axs[1].legend()
fig.tight_layout(); fig.savefig(FIG / "network_size.png"); plt.close(fig)

# 5 world location of trips
fig, ax = plt.subplots(figsize=(7, 3.4))
for p in io.PARTITIONS:
    d = inv[inv.partition == p]
    ax.scatter((d.lon_min + d.lon_max) / 2, (d.lat_min + d.lat_max) / 2, s=18, color=C[p], label=p)
ax.set_xlim(-180, 180); ax.set_ylim(-60, 75); ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("Trip centroids (worldwide coverage)"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "trip_locations.png"); plt.close(fig)

# 6 candidate statistics
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
axs[0].hist(O.n_cands, bins=np.arange(0, 14) - 0.5, color="#2f6db5"); axs[0].set_xlabel("candidates per observation"); axs[0].set_title("Candidate-set size (r=50 m, K≤12)")
cs2 = cs.sort_values("miss_rate")
axs[1].bar(range(len(cs2)), cs2.miss_rate * 100, color=[C[p] for p in cs2.partition]); axs[1].set_xlabel("trips (sorted)"); axs[1].set_ylabel("candidate miss rate (%)"); axs[1].set_title("Reference arc absent from candidates")
fig.tight_layout(); fig.savefig(FIG / "candidates.png"); plt.close(fig)

# 7 trajectory + network plots (projected metres)
def traj_plot(tid, part, ax):
    tr = clean_track(io.load_track(tid, part))
    net = io.load_network(tid, part)
    xy = tr.df[["x", "y"]].to_numpy()
    pn = ProjectedNetwork(net, tr.proj, bbox_xy=(xy.min(0), xy.max(0)))
    lo, hi = xy.min(0) - 200, xy.max(0) + 200
    mid = (pn.A + pn.B) / 2
    k = np.nonzero((mid >= lo).all(1) & (mid <= hi).all(1))[0]
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(np.stack([pn.A[k], pn.B[k]], 1), colors="#bbbbbb", linewidths=0.4))
    r = io.load_route(tid, part)
    ax.add_collection(LineCollection(np.stack([pn.A[r], pn.B[r]], 1), colors="#1a9850", linewidths=2.2))
    ax.plot(xy[:, 0], xy[:, 1], ".", ms=1.5, color="#1f6feb")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_title(f"{part} {tid}", fontsize=9); ax.tick_params(labelsize=6)

fig, axs = plt.subplots(2, 3, figsize=(11, 7))
for ax, (tid, part) in zip(axs.flat, [("00000001", "train"), ("00000090", "train"), ("00000054", "train"),
                                      ("00000000", "student_test"), ("00000052", "student_test"), ("00000091", "student_test")]):
    traj_plot(tid, part, ax)
fig.suptitle("GPS (blue dots), reference route (green), road network (grey) — local metres")
fig.tight_layout(); fig.savefig(FIG / "trajectories.png"); plt.close(fig)

# 8 Folium maps
for tid, part in [("00000001", "train"), ("00000054", "train"), ("00000052", "student_test")]:
    tr = clean_track(io.load_track(tid, part))
    xy = tr.df[["x", "y"]].to_numpy()
    pn = ProjectedNetwork(io.load_network(tid, part), tr.proj, bbox_xy=(xy.min(0), xy.max(0)))
    m = trip_map(f"EDA – {part} trip {tid}", pn, tr.df, reference=io.load_route(tid, part).tolist())
    m.save(str(MAPS / f"eda_{part}_{tid}.html"))
print("EDA figures + maps written")
