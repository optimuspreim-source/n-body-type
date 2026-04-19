"""
GalaxyViewer – Einfache 3D-Visualisierung der Szene (Platzhalter)
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GalaxyViewer:
    def __init__(self, scene):
        self.scene = scene

    def show(self):
        xs, ys, zs = [], [], []
        for obj in self.scene.objects:
            x, y, z = obj['position']
            xs.append(x)
            ys.append(y)
            zs.append(z)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, s=1)
        ax.set_title('Spiralgalaxie')
        plt.show()
