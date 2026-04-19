"""
Octree – 3D-Raumaufteilung für Barnes-Hut und Kollisionserkennung
"""
from engine.physics.AABB import AABB

class Octree:
    def __init__(self, boundary: AABB, capacity=8, depth=0, max_depth=10):
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.children = []
        self.depth = depth
        self.max_depth = max_depth

    def insert(self, point, data=None):
        if not self.boundary.contains(point):
            return False
        if len(self.points) < self.capacity or self.depth >= self.max_depth:
            self.points.append((point, data))
            return True
        if not self.children:
            self.subdivide()
        for child in self.children:
            if child.insert(point, data):
                return True
        return False

    def subdivide(self):
        min_pt, max_pt = self.boundary.min, self.boundary.max
        mx, my, mz = [(min_pt[i] + max_pt[i]) / 2 for i in range(3)]
        x0, y0, z0 = min_pt
        x1, y1, z1 = max_pt
        boxes = [
            ((x0, y0, z0), (mx, my, mz)),
            ((mx, y0, z0), (x1, my, mz)),
            ((x0, my, z0), (mx, y1, mz)),
            ((mx, my, z0), (x1, y1, mz)),
            ((x0, y0, mz), (mx, my, z1)),
            ((mx, y0, mz), (x1, my, z1)),
            ((x0, my, mz), (mx, y1, z1)),
            ((mx, my, mz), (x1, y1, z1)),
        ]
        for bmin, bmax in boxes:
            self.children.append(Octree(AABB(bmin, bmax), self.capacity, self.depth+1, self.max_depth))
