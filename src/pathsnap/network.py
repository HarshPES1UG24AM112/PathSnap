"""Projected road network with arc geometry, adjacency and a spatial index."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjection, bearing_xy, point_segment
from .io import Network

SAMPLE_STEP = 10.0  # m between index samples along an arc


class ProjectedNetwork:
    def __init__(self, net: Network, proj: LocalProjection, bbox_xy=None, margin=300.0):
        self.net = net
        self.proj = proj
        self.nxy = proj.to_xy(net.node_xy_ll[:, 0], net.node_xy_ll[:, 1])
        self.u = net.arcs[:, 0]
        self.v = net.arcs[:, 1]
        self.n_nodes, self.n_arcs = len(self.nxy), len(self.u)
        self.A = self.nxy[self.u]
        self.B = self.nxy[self.v]
        d = self.B - self.A
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = bearing_xy(d[:, 0], d[:, 1])
        self.out_deg = np.bincount(self.u, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.v, minlength=self.n_nodes)
        pair = {}
        for k, (a, b) in enumerate(zip(self.u.tolist(), self.v.tolist())):
            pair.setdefault((a, b), k)
        self.has_twin = np.array([(b, a) in pair for a, b in zip(self.u.tolist(), self.v.tolist())])
        # outgoing adjacency (node -> list of (arc, to_node, length))
        order = np.argsort(self.u, kind="stable")
        self._out_ptr = np.r_[0, np.cumsum(self.out_deg)]
        self._out_arcs = order
        self._sp_cache: dict[int, dict[int, float]] = {}
        self._parent_cache: dict[int, dict[int, int]] = {}
        self._build_index(bbox_xy, margin)

    def _build_index(self, bbox_xy, margin):
        keep = np.ones(self.n_arcs, bool)
        if bbox_xy is not None:
            (x0, y0), (x1, y1) = bbox_xy
            lo = np.minimum(self.A, self.B)
            hi = np.maximum(self.A, self.B)
            keep = (hi[:, 0] >= x0 - margin) & (lo[:, 0] <= x1 + margin) & (hi[:, 1] >= y0 - margin) & (lo[:, 1] <= y1 + margin)
        ids = np.nonzero(keep)[0]
        nseg = np.maximum(1, np.ceil(self.length[ids] / SAMPLE_STEP).astype(int))
        rep = np.repeat(ids, nseg + 1)
        frac = np.concatenate([np.linspace(0, 1, k + 1) for k in nseg]) if len(ids) else np.empty(0)
        pts = self.A[rep] + frac[:, None] * (self.B[rep] - self.A[rep])
        self._idx_arc = rep
        self._tree = cKDTree(pts) if len(pts) else None

    def arcs_near(self, xy: np.ndarray, radius: float) -> list[np.ndarray]:
        """Arc ids having an index sample within radius + SAMPLE_STEP/2 of each point."""
        if self._tree is None:
            return [np.empty(0, int) for _ in range(len(xy))]
        hits = self._tree.query_ball_point(xy, r=radius + SAMPLE_STEP / 2)
        return [np.unique(self._idx_arc[h]) if len(h) else np.empty(0, int) for h in hits]

    def dist_to_arcs(self, p: np.ndarray, arcs: np.ndarray):
        return point_segment(p[None, :], self.A[arcs], self.B[arcs])

    def out_arcs(self, node: int) -> np.ndarray:
        return self._out_arcs[self._out_ptr[node]:self._out_ptr[node + 1]]

    def shortest_from(self, src: int, cutoff: float) -> dict[int, float]:
        """Bounded Dijkstra over directed arcs (cached per source; cache grows cutoff)."""
        c = self._sp_cache.get(src)
        if c is not None and c.get(-1, 0.0) >= cutoff:
            return c
        dist = {src: 0.0}
        parent = {}
        pq = [(0.0, src)]
        while pq:
            d, n = heapq.heappop(pq)
            if d > dist.get(n, np.inf) or d > cutoff:
                continue
            for a in self.out_arcs(n):
                m = int(self.v[a])
                nd = d + float(self.length[a])
                if nd <= cutoff and nd < dist.get(m, np.inf):
                    dist[m] = nd
                    parent[m] = int(a)
                    heapq.heappush(pq, (nd, m))
        dist[-1] = cutoff
        self._sp_cache[src] = dist
        self._parent_cache[src] = parent
        return dist

    def path_arcs(self, src: int, dst: int, cutoff: float = 5000.0) -> list[int] | None:
        """Arc ids of the shortest directed path src -> dst (None if not reachable within cutoff)."""
        if src == dst:
            return []
        dist = self.shortest_from(src, cutoff)
        if dst not in dist:
            dist = self.shortest_from(src, max(cutoff, 4 * dist.get(-1, 0)))
            if dst not in dist:
                return None
        par = self._parent_cache[src]
        out, n = [], dst
        while n != src:
            a = par[n]
            out.append(a)
            n = int(self.u[a])
        return out[::-1]
