"""Trip-level preparation shared by training, evaluation and the API."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import io
from .features import trip_design
from .labels import align_route
from .network import ProjectedNetwork
from .preprocessing import CleanTrack, clean_track


@dataclass
class PreparedTrip:
    trip_id: str
    ct: CleanTrack
    pn: ProjectedNetwork
    X: pd.DataFrame  # candidate rows with features (+ 'label' if labelled)


def prepare_inputs(track: pd.DataFrame, net: io.Network, trip_id: str = "") -> PreparedTrip:
    """Label-free preparation: used by the API for prediction."""
    ct = clean_track(track)
    xy = ct.df[["x", "y"]].to_numpy()
    bbox = (xy.min(0), xy.max(0)) if len(xy) else None
    pn = ProjectedNetwork(net, ct.proj, bbox_xy=bbox)
    X = trip_design(ct, pn)
    X.insert(0, "trip_id", trip_id)
    return PreparedTrip(trip_id, ct, pn, X)


def reference_labels(pt: PreparedTrip, route: np.ndarray):
    """Per-observation reference arc (index aligned with pt.ct.df) via monotone alignment."""
    xy = pt.ct.df[["x", "y"]].to_numpy()
    _, lab, ldist = align_route(xy, route, pt.pn.A, pt.pn.B)
    return lab, ldist


def prepare_labelled(trip_id: str, partition: str, data_root=io.DATA_RAW) -> PreparedTrip:
    pt = prepare_inputs(io.load_track(trip_id, partition, data_root), io.load_network(trip_id, partition, data_root), trip_id)
    lab, ldist = reference_labels(pt, io.load_route(trip_id, partition, data_root))
    pt.ct.df["ref_arc"] = lab
    pt.ct.df["ref_dist"] = ldist
    pt.X["label"] = (pt.X.arc_id.to_numpy() == lab[pt.X.obs.to_numpy()]).astype(np.int8)
    return pt
