"""
Camera – 3D-Kamera für die Szene
"""
import math

class Camera:
    def __init__(self, position=(0,0,5), target=(0,0,0), up=(0,1,0), fov=60, aspect=16/9, near=0.1, far=1000):
        self.position = position
        self.target = target
        self.up = up
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

    def get_view_matrix(self):
        # Platzhalter für View-Matrix-Berechnung
        return None

    def get_projection_matrix(self):
        # Platzhalter für Projektionsmatrix
        return None
