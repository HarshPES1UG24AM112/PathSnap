"""Raw-file loading for PathSnap trips.

Each trip directory ``<partition>/<trip_id>/`` contains
``<trip_id>.track`` (lon, lat, t), ``<trip_id>.nodes`` (lon, lat; row index = node id),
``<trip_id>.arcs`` (from_node, to_node; row index = arc id) and, for labelled
partitions, ``<trip_id>.route`` (chronological arc ids).

The route is deliberately loaded by a *separate* function so that the prediction
pipeline (``load_network`` + ``load_track``) never touches reference labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
PARTITIONS = ("train", "student_test")


def _read_table(path: Path, ncols: int, dtype) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ncols), dtype=dtype)
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    arr = df.to_numpy()
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr[:, :ncols].astype(dtype)


@dataclass
class Network:
    node_xy_ll: np.ndarray  # (n_nodes, 2) lon, lat
    arcs: np.ndarray  # (n_arcs, 2) from_node, to_node


def trip_dir(trip_id: str, partition: str, data_root: Path = DATA_RAW) -> Path:
    return Path(data_root) / partition / trip_id


def list_trips(partition: str, data_root: Path = DATA_RAW) -> list[str]:
    p = Path(data_root) / partition
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def load_track(trip_id: str, partition: str, data_root: Path = DATA_RAW) -> pd.DataFrame:
    """Raw GPS track as DataFrame[lon, lat, t] (no cleaning)."""
    arr = _read_table(trip_dir(trip_id, partition, data_root) / f"{trip_id}.track", 3, float)
    return pd.DataFrame(arr, columns=["lon", "lat", "t"])


def load_network(trip_id: str, partition: str, data_root: Path = DATA_RAW) -> Network:
    d = trip_dir(trip_id, partition, data_root)
    nodes = _read_table(d / f"{trip_id}.nodes", 2, float)
    arcs = _read_table(d / f"{trip_id}.arcs", 2, np.int64)
    return Network(nodes, arcs)


def load_route(trip_id: str, partition: str, data_root: Path = DATA_RAW) -> np.ndarray:
    """Reference arc-id sequence. Used ONLY for label derivation / server-side evaluation."""
    arr = _read_table(trip_dir(trip_id, partition, data_root) / f"{trip_id}.route", 1, np.int64)
    return arr[:, 0]
