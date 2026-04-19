"""
Barnes-Hut-Algorithmus für 3D-N-Body-Simulation mit Octree/AABB
- NumPy-vektorisiert, parallelisierbar via ThreadPoolExecutor
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
        self.body = None   # (np.ndarray pos, float mass) at leaf, else None
        self.children = []


class BarnesHut3D:
    def __init__(self, boundary: AABB, theta=0.5, G=1.0, eps=0.5):
        self.root = BarnesHutNode(boundary)
        self.theta = theta
        self.G = G
        self.eps = eps  # Softening-Länge verhindert Singularitäten

    # ------------------------------------------------------------------
    # Octree-Aufbau
    # ------------------------------------------------------------------
    def insert(self, node: BarnesHutNode, point, mass):
        pt = np.asarray(point, dtype=np.float64)
        m = float(mass)

        if node.mass == 0.0:
            # Leerer Blatt-Knoten → direkt belegen
            node.mass = m
            node.center_of_mass = pt.copy()
            node.body = (pt, m)
            return

        # Schwerpunkt und Gesamtmasse aktualisieren
        total = node.mass + m
        node.center_of_mass = (node.center_of_mass * node.mass + pt * m) / total
        node.mass = total

        if node.body is not None:
            # Blatt → innerer Knoten: alten Body nach unten schieben
            self._subdivide(node)
            old_pt, old_m = node.body
            node.body = None
            self._insert_child(node, old_pt, old_m)

        self._insert_child(node, pt, m)

    def _insert_child(self, node: BarnesHutNode, pt: np.ndarray, mass: float):
        for child in node.children:
            if child.boundary.contains(pt):
                self.insert(child, pt, mass)
                return

    def _subdivide(self, node: BarnesHutNode):
        mn = np.array(node.boundary.min, dtype=np.float64)
        mx = np.array(node.boundary.max, dtype=np.float64)
        mid = (mn + mx) * 0.5
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    bmin = np.where([dx, dy, dz], mid, mn)
                    bmax = np.where([dx, dy, dz], mx, mid)
                    node.children.append(BarnesHutNode(AABB(tuple(bmin), tuple(bmax))))

    # ------------------------------------------------------------------
    # Kraftberechnung (einzeln + parallelisiert)
    # ------------------------------------------------------------------
    def _force_on(self, node: BarnesHutNode, pos: np.ndarray) -> np.ndarray:
        if node.mass == 0.0:
            return np.zeros(3)

        dr = node.center_of_mass - pos
        dist2 = np.dot(dr, dr)

        if dist2 < 1e-10:
            # Selbst-Interaktion überspringen, in Kinder abtauchen
            if not node.children:
                return np.zeros(3)
            return sum((self._force_on(c, pos) for c in node.children), np.zeros(3))

        dist_soft = np.sqrt(dist2 + self.eps * self.eps)
        size = node.boundary.max[0] - node.boundary.min[0]

        # Barnes-Hut-Kriterium: Näherung wenn Blatt oder s/d < θ
        if not node.children or size / np.sqrt(dist2) < self.theta:
            F = self.G * node.mass / (dist_soft * dist_soft)
            return F * dr / dist_soft

        return sum((self._force_on(c, pos) for c in node.children), np.zeros(3))

    def calculate_force(self, node: BarnesHutNode, point, G=None):
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
        N = len(positions)
        forces = np.zeros((N, 3), dtype=np.float64)

        def calc(i):
            return self._force_on(self.root, positions[i])

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(calc, range(N)))

        for i, f in enumerate(results):
            forces[i] = f
        return forces
