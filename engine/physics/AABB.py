"""
AABB – Axis-Aligned Bounding Box für 3D-Kollisionserkennung
"""

class AABB:
    def __init__(self, min_point, max_point):
        self.min = min_point  # (x, y, z)
        self.max = max_point  # (x, y, z)

    def contains(self, point):
        return all(self.min[i] <= point[i] <= self.max[i] for i in range(3))

    def intersects(self, other):
        return all(self.min[i] <= other.max[i] and self.max[i] >= other.min[i] for i in range(3))
