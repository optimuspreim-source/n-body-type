"""
Octree – 3D-Raumaufteilung für Barnes-Hut und Nachbarsuche
═══════════════════════════════════════════════════════════════
Recursive Octree mit
  • insert(point, data)      – O(log N)
  • query_radius(center, r)  – O(k log N)  mit k = Treffer
  • query_box(aabb)          – AABB-Schnittmächtigkeitstest
  • all_data()               – lineare Traversierung aller Blätter

Für sehr große N (> 50 000) bevorzuge SpatialHash aus AABB.py:
der Octree ist tiefer aber hat mehr Python-Overhead pro Knoten.
═══════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import numpy as np
from engine.physics.AABB import AABB


class Octree:
    """Rekursiver Octree. Rückwärtskompatibel mit altem insert/subdivide-Interface."""

    __slots__ = ('boundary', 'capacity', 'points', 'children', 'depth', 'max_depth')

    def __init__(self, boundary: AABB, capacity: int = 8,
                 depth: int = 0, max_depth: int = 16):
        self.boundary  = boundary
        self.capacity  = capacity
        self.points:   list[tuple] = []      # (point_array, data)
        self.children: list['Octree'] = []
        self.depth     = depth
        self.max_depth = max_depth

    # ── Einfügen ───────────────────────────────────────────────────────────

    def insert(self, point, data=None) -> bool:
        pt = np.asarray(point, dtype=np.float64)
        if not self.boundary.contains(pt):
            return False
        if len(self.points) < self.capacity or self.depth >= self.max_depth:
            self.points.append((pt, data))
            return True
        if not self.children:
            self.subdivide()
        for child in self.children:
            if child.insert(pt, data):
                return True
        # Fallback: Kapazität trotz Unterteilung überschritten (Randfall)
        self.points.append((pt, data))
        return True

    def subdivide(self) -> None:
        mn = np.array(self.boundary.min, dtype=np.float64)
        mx = np.array(self.boundary.max, dtype=np.float64)
        mid = (mn + mx) * 0.5
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    bmin = np.where([dx, dy, dz], mid, mn)
                    bmax = np.where([dx, dy, dz], mx, mid)
                    self.children.append(
                        Octree(AABB(tuple(bmin), tuple(bmax)),
                               self.capacity, self.depth + 1, self.max_depth))

    # ── Abfragen ───────────────────────────────────────────────────────────

    def query_radius(self, center, radius: float) -> list:
        """
        Gibt alle (point, data)-Tupel zurück, deren Abstand zu `center` ≤ radius.
        Schneidet erst AABB, dann exakten Kugeltest.
        """
        c    = np.asarray(center, dtype=np.float64)
        r2   = radius * radius
        # Bounding-Box der Kugel testen
        sphere_aabb = AABB(
            tuple(float(v) for v in (c - radius)),
            tuple(float(v) for v in (c + radius)),
        )
        if not self.boundary.intersects(sphere_aabb):
            return []
        result = []
        for pt, data in self.points:
            if float(((pt - c) ** 2).sum()) <= r2:
                result.append((pt, data))
        for child in self.children:
            result.extend(child.query_radius(c, radius))
        return result

    def query_box(self, box: AABB) -> list:
        """Gibt alle (point, data)-Tupel innerhalb einer AABB zurück."""
        if not self.boundary.intersects(box):
            return []
        result = []
        for pt, data in self.points:
            if box.contains(pt):
                result.append((pt, data))
        for child in self.children:
            result.extend(child.query_box(box))
        return result

    def all_data(self) -> list:
        """Traversiert alle Blätter und gibt alle (point, data) zurück."""
        result = list(self.points)
        for child in self.children:
            result.extend(child.all_data())
        return result

    # ── Hilfsmethode ──────────────────────────────────────────────────────────

    @staticmethod
    def build(pos: np.ndarray, data: list | None = None,
              capacity: int = 8, max_depth: int = 16,
              margin: float = 1.0) -> 'Octree':
        """
        Convenience-Methode: Erstellt einen Octree für eine Punktwolke.

        Parameters
        ----------
        pos      : (N, 3) float64
        data     : optionale Nutz-Daten pro Punkt (z.B. Index-Liste)
        capacity : Max. Punkte pro Knoten vor Unterteilung
        """
        active = pos[~np.all(pos == 0, axis=1)]  # 0-Punkte ignorieren
        if len(active) == 0:
            return Octree(AABB((-1., -1., -1.), (1., 1., 1.)), capacity, max_depth=max_depth)
        boundary = AABB.from_points(pos, margin=margin)
        tree = Octree(boundary, capacity=capacity, max_depth=max_depth)
        for i in range(len(pos)):
            d = data[i] if data is not None else i
            tree.insert(pos[i], d)
        return tree
