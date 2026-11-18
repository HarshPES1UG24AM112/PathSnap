"""Folium map builders (layers: road network, raw GPS, reference route, predicted route)."""
from __future__ import annotations

import folium
import numpy as np

COLORS = {"gps": "#1f6feb", "network": "#9aa0a6", "reference": "#1a9850", "predicted": "#d62728", "classifier": "#ff7f0e"}


def _legend(items):
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};margin-right:6px;vertical-align:middle"></span>{n}</div>'
        for n, c in items)
    html = ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
            'border:1px solid #888;border-radius:6px;font:12px sans-serif">' f"<b>PathSnap</b>{rows}</div>")
    return folium.Element(html)


def arcs_polylines(pn, arc_ids):
    """Lat/lon polylines for given arc ids (straight node-to-node segments)."""
    ll = pn.net.node_xy_ll
    return [[(ll[pn.u[a], 1], ll[pn.u[a], 0]), (ll[pn.v[a], 1], ll[pn.v[a], 0])] for a in arc_ids]


def route_path(pn, arc_seq):
    """Collapse consecutive duplicates and return list of polylines."""
    seq = [a for k, a in enumerate(arc_seq) if k == 0 or a != arc_seq[k - 1]]
    return arcs_polylines(pn, seq)


def network_arcs_in_view(pn, track_xy, margin=150.0, max_arcs=20000):
    lo, hi = track_xy.min(0) - margin, track_xy.max(0) + margin
    mid = (pn.A + pn.B) / 2
    keep = np.nonzero((mid[:, 0] >= lo[0]) & (mid[:, 0] <= hi[0]) & (mid[:, 1] >= lo[1]) & (mid[:, 1] <= hi[1]))[0]
    return keep[:max_arcs]


def trip_map(title, pn, track_df, reference=None, predicted=None, classifier=None, show_network=True):
    lat, lon = track_df.lat.to_numpy(), track_df.lon.to_numpy()
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    legend = [("Raw GPS observations", COLORS["gps"])]
    if show_network:
        fg = folium.FeatureGroup("Road network (local)", show=True)
        ids = network_arcs_in_view(pn, track_df[["x", "y"]].to_numpy())
        folium.PolyLine(arcs_polylines(pn, ids), color=COLORS["network"], weight=1, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.insert(0, ("Road network", COLORS["network"]))
    for name, seq, key, w in (("Reference route (server-side)", reference, "reference", 7),
                              ("Classifier-only argmax", classifier, "classifier", 3),
                              ("Predicted route (classifier + HMM)", predicted, "predicted", 4)):
        if seq is None:
            continue
        fg = folium.FeatureGroup(name, show=key != "classifier")
        folium.PolyLine(route_path(pn, list(seq)), color=COLORS[key], weight=w, opacity=0.85).add_to(fg)
        fg.add_to(m)
        legend.append((name, COLORS[key]))
    fg = folium.FeatureGroup("Raw GPS observations")
    step = max(1, len(lat) // 3000)
    for la, lo_ in zip(lat[::step], lon[::step]):
        folium.CircleMarker((la, lo_), radius=2, color=COLORS["gps"], fill=True, weight=0, fill_opacity=0.7).add_to(fg)
    fg.add_to(m)
    folium.Marker((lat[0], lon[0]), tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((lat[-1], lon[-1]), tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(legend))
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:4px 8px;font:bold 13px sans-serif;border-radius:4px">{title}</div>'))
    return m
