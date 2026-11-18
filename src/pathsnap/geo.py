"""Geodesic helpers.

Coordinates are WGS84 lon/lat (degrees). For metric computations each trip is projected
onto a local azimuthal-equidistant-like tangent plane (equirectangular about the trip's
centroid): x = R*cos(lat0)*dlon, y = R*dlat. Trips span < ~40 km, where the scale error
of this projection is < 0.1 %, far below GPS noise (~5-20 m), and it is ~100x cheaper than
calling a full UTM transform for millions of point-segment distances.
"""
from __future__ import annotations

import numpy as np

R_EARTH = 6_371_008.8


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = np.deg2rad(1.0) * R_EARTH * np.cos(np.deg2rad(self.lat0))
        self.ky = np.deg2rad(1.0) * R_EARTH

    @classmethod
    def for_points(cls, lon, lat):
        return cls(np.nanmean(lon), np.nanmean(lat))

    def to_xy(self, lon, lat):
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        return np.stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky], axis=-1)

    def to_ll(self, xy):
        xy = np.asarray(xy, float)
        return np.stack([xy[..., 0] / self.kx + self.lon0, xy[..., 1] / self.ky + self.lat0], axis=-1)


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_xy(dx, dy):
    """Compass bearing in degrees [0,360) from projected displacement (north = +y)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Absolute smallest difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


def point_segment(p, a, b):
    """Distance from points p (..,2) to segments a->b (..,2); returns (dist, t in [0,1])."""
    ab = b - a
    ap = p - a
    L2 = np.sum(ab * ab, axis=-1)
    t = np.where(L2 > 0, np.sum(ap * ab, axis=-1) / np.where(L2 > 0, L2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1), t
