"""
AABB – Axis-Aligned Bounding Box + SpatialHash für 3D-Kollisionserkennung
═══════════════════════════════════════════════════════════════
AABB
    Klassische Bounding-Box für Octree-Strukturen (unverändert rückwärtskompatibel).

SpatialHash
    O(1) amortisierte Einfüge- und Abfragezeit für Punkt-in-Kugel-Suchen.
    Funktionsweise:
      - Raum in gleichmäßige Voxel der Größe `cell_size` aufgeteilt
      - Jeder Punkt einem Voxel (ix, iy, iz) zugeordnet
      - query_radius(p, r) durchsucht nur die 27 Nachbarvoxel  (3³)
    Komplexität:
      - build():        O(N)
      - query_radius(): O(k)  mit k = Partikel im Suchvolumen
      - vs. Brute-Force: O(N)  → 27× schneller bei typischer Dichte
═══════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable
import numpy as np


class AABB:
    """Axis-Aligned Bounding Box (rückwärtskompatibel)."""

    __slots__ = ('min', 'max')

    def __init__(self, min_point, max_point):
        self.min = min_point  # (x, y, z)
        self.max = max_point  # (x, y, z)

    def contains(self, point) -> bool:
        return all(self.min[i] <= point[i] <= self.max[i] for i in range(3))

    def intersects(self, other: 'AABB') -> bool:
        return all(self.min[i] <= other.max[i] and self.max[i] >= other.min[i] for i in range(3))

    def expand(self, margin: float) -> 'AABB':
        """Gibt eine um `margin` vergrößerte Kopie zurück."""
        mn = tuple(v - margin for v in self.min)
        mx = tuple(v + margin for v in self.max)
        return AABB(mn, mx)

    @staticmethod
    def from_points(pts: np.ndarray, margin: float = 1.0) -> 'AABB':
        """Berechnet die minimale AABB für eine Punktwolke."""
        return AABB(
            tuple(float(v) for v in pts.min(axis=0) - margin),
            tuple(float(v) for v in pts.max(axis=0) + margin),
        )


class SpatialHash:
    """
    Voxel-basierter Spatial-Index für O(1) Nearest-Neighbor-Suche.

    Typische Verwendung im MergerKernel:
        sh = SpatialHash(cell_size=r_accrete)
        sh.build(pos, active_mask)
        candidates = sh.query_radius(bh_pos, r_accrete)
    """

    def __init__(self, cell_size: float):
        if cell_size <= 0:
            raise ValueError('cell_size muss positiv sein')
        self._cell  = float(cell_size)
        self._grid: dict[tuple[int,int,int], list[int]] = defaultdict(list)
        self._built = False

    # ── Aufbau ────────────────────────────────────────────────────────────

    def build(self, pos: np.ndarray, active: np.ndarray | None = None) -> None:
        """
        Fügt alle Partikel in den Spatial-Index ein.

        Parameters
        ----------
        pos    : (N, 3) float64 Positionen
        active : (N,) bool-Maske  (None → alle Partikel)
        """
        self._grid.clear()
        c = self._cell
        if active is None:
            indices: Iterable[int] = range(len(pos))
        else:
            indices = np.where(active)[0]
        for i in indices:
            key = (
                int(np.floor(pos[i, 0] / c)),
                int(np.floor(pos[i, 1] / c)),
                int(np.floor(pos[i, 2] / c)),
            )
            self._grid[key].append(int(i))
        self._built = True

    # ── Abfrage ───────────────────────────────────────────────────────────

    def query_radius(self, center: np.ndarray, radius: float) -> list[int]:
        """
        Gibt alle Partikel-Indizes zurück, deren Gitterzelle innerhalb
        von ceil(radius / cell_size) Zellen liegt.

        Hinweis: Gibt auch Partikel zurück, die > radius entfernt liegen
        (Zellgranularität). Die genaue Entfernung muss der Aufrufer prüfen.
        """
        c   = self._cell
        r_c = int(np.ceil(radius / c))          # Zell-Suchradius
        cx  = int(np.floor(center[0] / c))
        cy  = int(np.floor(center[1] / c))
        cz  = int(np.floor(center[2] / c))
        result: list[int] = []
        for dx in range(-r_c, r_c + 1):
            for dy in range(-r_c, r_c + 1):
                for dz in range(-r_c, r_c + 1):
                    key = (cx + dx, cy + dy, cz + dz)
                    if key in self._grid:
                        result.extend(self._grid[key])
        return result

    def query_radius_exact(self, center: np.ndarray, radius: float,
                           pos: np.ndarray) -> list[int]:
        """
        Wie query_radius, filtert aber exakt auf euklidische Distanz ≤ radius.
        """
        r2  = radius * radius
        raw = self.query_radius(center, radius)
        return [i for i in raw
                if float(((pos[i] - center) ** 2).sum()) <= r2]
