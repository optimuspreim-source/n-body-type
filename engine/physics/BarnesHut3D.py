"""
BarnesHut3D  –  Pure-Python Barnes-Hut N-Body-Solver (Fallback / Diagnose)
═══════════════════════════════════════════════════════════════════════════════
Reine Python/NumPy-Implementierung ohne Numba JIT.

Wann nutzen:
  • N < 300   (Numba-Overhead > Python-Overhead)
  • Debugging  (kein JIT-Kompilier-Delay, klarer Stack-Trace)
  • Plattformen ohne Numba (ARM, alte CUDA)

Backend-Kennung im GalaxySimVispyViewer: _BACKEND_PY

Schnittstelle (identisch mit BarnesHutNumba):
    acc = BarnesHut3D.solve(pos, mass, G, eps, theta)
    # acc: (N, 3) float64 Beschleunigungen

Achtung: ~50–200× langsamer als BarnesHutNumba für N > 1 000.
═══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from engine.physics.AABB import AABB


class BarnesHutNode:
    __slots__ = ['boundary', 'mass', 'center_of_mass', 'body', 'children']

    def __init__(self, boundary: AABB):
        self.boundary = boundary
        self.mass = 0.0
        self.center_of_mass = np.zeros(3, dtype=np.float64)
        self.body: tuple | None = None   # (np.ndarray pos, float mass) at leaf, else None
        self.children: list['BarnesHutNode'] = []


class BarnesHut3D:
    def __init__(self, boundary: AABB, theta: float = 0.5, G: float = 1.0, eps: float = 0.5):
        self.root  = BarnesHutNode(boundary)
        self.theta = theta
        self.G     = G
        self.eps   = eps  # Softening-Länge verhindert Singularitäten

    # ── Octree-Aufbau ──────────────────────────────────────────────────────────

    def insert(self, node: BarnesHutNode, point, mass: float) -> None:
        pt = np.asarray(point, dtype=np.float64)
        m  = float(mass)
        if node.mass == 0.0:
            node.mass = m
            node.center_of_mass = pt.copy()
            node.body = (pt, m)
            return
        total = node.mass + m
        node.center_of_mass = (node.center_of_mass * node.mass + pt * m) / total
        node.mass = total
        if node.body is not None:
            self._subdivide(node)
            old_pt, old_m = node.body
            node.body = None
            self._insert_child(node, old_pt, old_m)
        self._insert_child(node, pt, m)

    def _insert_child(self, node: BarnesHutNode, pt: np.ndarray, mass: float) -> None:
        for child in node.children:
            if child.boundary.contains(pt):
                self.insert(child, pt, mass)
                return

    def _subdivide(self, node: BarnesHutNode) -> None:
        mn  = np.array(node.boundary.min, dtype=np.float64)
        mx  = np.array(node.boundary.max, dtype=np.float64)
        mid = (mn + mx) * 0.5
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    bmin = np.where([dx, dy, dz], mid, mn)
                    bmax = np.where([dx, dy, dz], mx, mid)
                    node.children.append(BarnesHutNode(AABB(tuple(bmin), tuple(bmax))))

    # ── Kraft-/Beschleunigungsberechnung ──────────────────────────────────────

    def _force_on(self, node: BarnesHutNode, pos: np.ndarray) -> np.ndarray:
        if node.mass == 0.0:
            return np.zeros(3)
        dr    = node.center_of_mass - pos
        dist2 = float(np.dot(dr, dr))
        if dist2 < 1e-10:
            if not node.children:
                return np.zeros(3)
            return sum((self._force_on(c, pos) for c in node.children), np.zeros(3))
        dist_soft = np.sqrt(dist2 + self.eps * self.eps)
        size      = node.boundary.max[0] - node.boundary.min[0]
        if not node.children or size / np.sqrt(dist2) < self.theta:
            F = self.G * node.mass / (dist_soft * dist_soft)
            return F * dr / dist_soft
        return sum((self._force_on(c, pos) for c in node.children), np.zeros(3))

    def calculate_force(self, node: BarnesHutNode, point, G: float | None = None) -> np.ndarray:
        """Einzelne Kraftberechnung (G-Override möglich)."""
        saved = self.G
        if G is not None:
            self.G = G
        result = self._force_on(node, np.asarray(point, dtype=np.float64))
        self.G = saved
        return result

    def calculate_all_forces(self, positions: np.ndarray, masses: np.ndarray,
                              workers: int = 4) -> np.ndarray:
        """
        Parallelisierte Kraftberechnung für alle N Partikel via ThreadPoolExecutor.
        numpy-Operationen geben die GIL frei → echte Parallelität.
        """
        N      = len(positions)
        forces = np.zeros((N, 3), dtype=np.float64)

        def calc(i):
            return self._force_on(self.root, positions[i])

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(calc, range(N)))
        for i, f in enumerate(results):
            forces[i] = f
        return forces

    # ── Viewer-Backend-Schnittstelle (identisch mit BarnesHutNumba) ───────────

    @classmethod
    def solve(cls, pos: np.ndarray, mass: np.ndarray,
              G: float = 1.0, eps: float = 0.5, theta: float = 0.5,
              workers: int = 2) -> np.ndarray:
        """
        Berechnet Beschleunigungen für alle aktiven Partikel.

        Parameters
        ----------
        pos    : (N, 3) float64
        mass   : (N,) float64  (mass=0 → inaktiv, wird übersprungen)
        G, eps, theta: Physik-Parameter

        Rückgabe: (N, 3) float64 Beschleunigungen
        """
        active = mass > 0.
        if active.sum() < 2:
            return np.zeros_like(pos)

        ap   = pos[active]
        am   = mass[active]
        mn   = ap.min(axis=0) - eps * 2
        mx   = ap.max(axis=0) + eps * 2
        bh   = cls(AABB(tuple(mn), tuple(mx)), theta=theta, G=G, eps=eps)
        for i in range(len(ap)):
            bh.insert(bh.root, ap[i], am[i])
        acc_active = bh.calculate_all_forces(ap, am, workers=workers)
        # Kraft → Beschleunigung: a = F / m
        acc_active /= am[:, np.newaxis]

        acc = np.zeros_like(pos)
        acc[active] = acc_active
        return acc
