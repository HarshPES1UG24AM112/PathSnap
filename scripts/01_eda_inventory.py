"""EDA step 1: dataset inventory, data-quality and network statistics for every trip.

Partitions used: train (80) and student_test (10). Outputs -> results/eda/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import io  # noqa: E402
from pathsnap.geo import LocalProjection, point_segment  # noqa: E402

OUT = io.ROOT / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for part in io.PARTITIONS:
    for tid in io.list_trips(part):
        d = io.trip_dir(tid, part)
        tr = io.load_track(tid, part)
        net = io.load_network(tid, part)
        route = io.load_route(tid, part)
        n_nodes, n_arcs = len(net.node_xy_ll), len(net.arcs)
        lon, lat, t = tr.lon.to_numpy(), tr.lat.to_numpy(), tr.t.to_numpy()
        nan_rows = int(tr.isna().any(axis=1).sum())
        invalid = int(((np.abs(lat) > 90) | (np.abs(lon) > 180) | ((lat == 0) & (lon == 0))).sum())
        dt = np.diff(t)
        proj = LocalProjection.for_points(lon, lat)
        xy = proj.to_xy(lon, lat)
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        speed = step / np.where(dt > 0, dt, np.nan)
        # network
        a, b = net.arcs[:, 0], net.arcs[:, 1]
        bad_ref = int(((a < 0) | (a >= n_nodes) | (b < 0) | (b >= n_nodes)).sum())
        g = coo_matrix((np.ones(n_arcs), (a, b)), shape=(n_nodes, n_nodes))
        n_weak, lab_w = connected_components(g, directed=True, connection="weak")
        n_strong, lab_s = connected_components(g, directed=True, connection="strong")
        deg = np.bincount(a, minlength=n_nodes) + np.bincount(b, minlength=n_nodes)
        isolated = int((deg == 0).sum())
        largest_scc = np.bincount(lab_s).max() / n_nodes
        nxy = proj.to_xy(net.node_xy_ll[:, 0], net.node_xy_ll[:, 1])
        arc_len = np.linalg.norm(nxy[b] - nxy[a], axis=1)
        self_loops = int((a == b).sum())
        dup_arcs = int(pd.DataFrame(net.arcs).duplicated().sum())
        # route
        r_valid = bool(((route >= 0) & (route < n_arcs)).all())
        rr = route[(route >= 0) & (route < n_arcs)]
        contiguous = float(np.mean(net.arcs[rr[:-1], 1] == net.arcs[rr[1:], 0])) if len(rr) > 1 else np.nan
        repeats = int(len(rr) - len(np.unique(rr)))
        route_len = float(arc_len[rr].sum())
        # GPS -> nearest reference-route arc distance (label plausibility)
        ra, rb = nxy[net.arcs[rr, 0]], nxy[net.arcs[rr, 1]]
        dmin = np.full(len(xy), np.inf)
        for k in range(0, len(rr), 256):
            dd, _ = point_segment(xy[:, None, :], ra[None, k:k + 256], rb[None, k:k + 256])
            dmin = np.minimum(dmin, dd.min(axis=1))
        rows.append(dict(
            partition=part, trip_id=tid,
            track_kb=round((d / f"{tid}.track").stat().st_size / 1024, 1),
            nodes_kb=round((d / f"{tid}.nodes").stat().st_size / 1024, 1),
            n_obs=len(tr), nan_rows=nan_rows, invalid_coords=invalid,
            t_start=t[0], t_end=t[-1], duration_s=t[-1] - t[0],
            non_increasing_t=int((dt <= 0).sum()), frac_dt_1s=float(np.mean(dt == 1)),
            max_dt=float(dt.max()), n_gaps_gt5s=int((dt > 5).sum()),
            track_len_km=float(np.nansum(step) / 1000), median_speed=float(np.nanmedian(speed)),
            max_speed=float(np.nanmax(speed)), n_speed_gt50=int((speed > 50).sum()),
            lon_min=lon.min(), lon_max=lon.max(), lat_min=lat.min(), lat_max=lat.max(),
            n_nodes=n_nodes, n_arcs=n_arcs, bad_node_refs=bad_ref, self_loops=self_loops,
            dup_arcs=dup_arcs, isolated_nodes=isolated, weak_components=n_weak,
            strong_components=n_strong, largest_scc_frac=float(largest_scc),
            median_arc_len=float(np.median(arc_len)), max_arc_len=float(arc_len.max()),
            route_arcs=len(route), route_ids_valid=r_valid, route_contiguity=contiguous,
            route_repeats=repeats, route_len_km=route_len / 1000,
            gps_to_route_med=float(np.median(dmin)), gps_to_route_p95=float(np.percentile(dmin, 95)),
            gps_to_route_max=float(dmin.max()),
        ))
        print(part, tid, len(tr), n_nodes, n_arcs, len(route), f"{np.median(dmin):.1f}/{np.percentile(dmin,95):.1f}/{dmin.max():.0f}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "trip_inventory.csv", index=False)
summ = df.groupby("partition").agg(
    trips=("trip_id", "count"), obs_total=("n_obs", "sum"), obs_median=("n_obs", "median"),
    obs_min=("n_obs", "min"), obs_max=("n_obs", "max"), nan_rows=("nan_rows", "sum"),
    invalid_coords=("invalid_coords", "sum"), non_increasing_t=("non_increasing_t", "sum"),
    frac_dt_1s=("frac_dt_1s", "mean"), gaps_gt5s=("n_gaps_gt5s", "sum"), max_dt=("max_dt", "max"),
    nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"),
    isolated_nodes=("isolated_nodes", "sum"), weak_comp_median=("weak_components", "median"),
    route_contiguity=("route_contiguity", "mean"), gps_to_route_med=("gps_to_route_med", "median"),
)
summ.T.to_csv(OUT / "partition_summary.csv")
print(summ.T.to_string())
