"""Unit tests for geometry, labelling, candidates, HMM and leakage guarantees."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import io  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.geo import LocalProjection, angle_diff, haversine, point_segment  # noqa: E402
from pathsnap.hmm import HMMParams, build_lattice, reconstruct_route, viterbi  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.pipeline import prepare_inputs  # noqa: E402


def test_point_segment_and_angles():
    d, t = point_segment(np.array([[0.0, 5.0]]), np.array([[-10.0, 0.0]]), np.array([[10.0, 0.0]]))
    assert d[0] == pytest.approx(5.0) and t[0] == pytest.approx(0.5)
    d, t = point_segment(np.array([[20.0, 0.0]]), np.array([[-10.0, 0.0]]), np.array([[10.0, 0.0]]))
    assert d[0] == pytest.approx(10.0) and t[0] == 1.0
    assert angle_diff(350, 10) == pytest.approx(20)
    assert angle_diff(0, 180) == pytest.approx(180)


def test_projection_matches_haversine():
    p = LocalProjection(8.6, 49.85)
    xy = p.to_xy([8.6, 8.61], [49.85, 49.86])
    d_proj = np.linalg.norm(xy[1] - xy[0])
    d_hav = haversine(8.6, 49.85, 8.61, 49.86)
    assert abs(d_proj - d_hav) / d_hav < 1e-3
    assert np.allclose(p.to_ll(xy), [[8.6, 49.85], [8.61, 49.86]])


def _toy_network():
    """Two parallel east-west roads 20 m apart (disconnected) plus reverse arcs; lon/lat near (8, 50)."""
    proj = LocalProjection(8.0, 50.0)
    xs = np.arange(0, 501, 100.0)
    main = np.stack([xs, np.zeros_like(xs)], 1)
    side = np.stack([xs, np.full_like(xs, 20.0)], 1)
    nodes_xy = np.vstack([main, side])
    ll = proj.to_ll(nodes_xy)
    n = len(xs)
    arcs = [(i, i + 1) for i in range(n - 1)] + [(n + i, n + i + 1) for i in range(n - 1)]
    arcs += [(b, a) for a, b in list(arcs)]
    return io.Network(ll, np.array(arcs)), proj


def _toy_track(proj):
    xs = np.arange(5, 495, 10.0)
    ys = np.full_like(xs, 4.0)
    ys[20] = 14.0  # one noisy fix pulled towards the side road
    ll = proj.to_ll(np.stack([xs, ys], 1))
    return pd.DataFrame({"lon": ll[:, 0], "lat": ll[:, 1], "t": np.arange(len(xs), dtype=float)})


def test_candidates_preserve_arc_correspondence_and_no_route_needed(tmp_path):
    net, proj = _toy_network()
    pt = prepare_inputs(_toy_track(proj), net, "toy")
    X = pt.X
    assert set(FEATURES) <= set(X.columns)
    assert X.arc_id.between(0, len(net.arcs) - 1).all()
    # recompute the distance to each candidate's own arc -> must match the stored feature
    xy = pt.ct.df[["x", "y"]].to_numpy()[X.obs.to_numpy()]
    d, _ = point_segment(xy, pt.pn.A[X.arc_id.to_numpy()], pt.pn.B[X.arc_id.to_numpy()])
    assert np.allclose(d, X.dist.to_numpy())
    assert "label" not in X.columns  # label-free preparation


def test_route_alignment_is_monotone():
    net, proj = _toy_network()
    tr = _toy_track(proj)
    pt = prepare_inputs(tr, net, "toy")
    route = np.arange(5)  # main road eastbound arcs 0..4
    pos, lab, ld = align_route(pt.ct.df[["x", "y"]].to_numpy(), route, pt.pn.A, pt.pn.B)
    assert (np.diff(pos) >= 0).all()
    assert lab[0] == 0 and lab[-1] == 4
    assert ld.max() < 15


def test_viterbi_prefers_connected_route_over_noisy_argmax():
    net, proj = _toy_network()
    pt = prepare_inputs(_toy_track(proj), net, "toy")
    X = pt.X.copy()
    # emulate a classifier that trusts distance only: closer arc gets higher score
    X["score"] = np.exp(-X.dist / 5.0)
    clf = X.loc[X.groupby("obs").score.idxmax()].set_index("obs").arc_id
    side_arcs = set(range(5, 10)) | set(range(15, 20))
    assert clf.loc[20] in side_arcs  # noisy fix -> classifier picks the disconnected side road
    lat = build_lattice(pt.ct.df, pt.pn, X)
    pred = viterbi(lat, HMMParams(beta=10, lam=1.0, unreachable_logp=-12))
    assert pred[20] not in side_arcs and all(p < 5 for p in pred)  # HMM stays on the eastbound main road
    route, breaks = reconstruct_route(pt.pn, pred)
    assert breaks == 0 and route == [0, 1, 2, 3, 4]


def test_viterbi_handles_gap_and_empty_candidates():
    net, proj = _toy_network()
    tr = _toy_track(proj)
    tr.loc[10, "t"] = tr.loc[9, "t"] + 0.5
    tr.loc[11:, "t"] += 120  # a 2-minute GPS gap
    far = proj.to_ll(np.array([[250.0, 5000.0]]))[0]  # fix 5 km away -> no candidates
    tr = pd.concat([tr, pd.DataFrame({"lon": [far[0]], "lat": [far[1]], "t": [tr.t.max() + 1]})], ignore_index=True)
    pt = prepare_inputs(tr, net, "toy")
    X = pt.X.assign(score=lambda d: np.exp(-d.dist / 5.0))
    assert pt.ct.df.gap_before.sum() >= 1
    pred = viterbi(build_lattice(pt.ct.df, pt.pn, X), HMMParams())
    assert len(pred) == len(pt.ct.df) and pred[-1] == -1 and (pred[:-1] >= 0).all()


def test_invalid_rows_are_dropped():
    net, proj = _toy_network()
    tr = _toy_track(proj)
    tr.loc[3, "lat"] = np.nan
    tr.loc[4, ["lon", "lat"]] = 0.0
    pt = prepare_inputs(tr, net, "toy")
    assert pt.ct.n_dropped == 2 and len(pt.ct.df) == len(tr) - 2
