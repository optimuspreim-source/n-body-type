"""
GalaxySimViewer – Echtzeit-3D-Animation der Spiralgalaxie mit N-Body-Dynamik
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from engine.physics.BarnesHut3D import BarnesHut3D, BarnesHutNode
from engine.physics.AABB import AABB

class GalaxySimViewer:
    def __init__(self, stars, steps_per_frame=1, dt=0.01):
        self.stars = stars
        self.steps_per_frame = steps_per_frame
        self.dt = dt
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scat = None

    def update_positions(self):
        # Simuliere N-Body mit Barnes-Hut
        positions = np.array([s['position'] for s in self.stars])
        masses = np.array([s['mass'] for s in self.stars])
        velocities = np.array([s['velocity'] for s in self.stars])
        min_pt = positions.min(axis=0) - 1
        max_pt = positions.max(axis=0) + 1
        boundary = AABB(tuple(min_pt), tuple(max_pt))
        bh = BarnesHut3D(boundary)
        nodes = []
        for i, s in enumerate(self.stars):
            bh.insert(bh.root, s['position'], s['mass'])
            nodes.append(s)
        for i, s in enumerate(self.stars):
            force = bh.calculate_force(bh.root, s['position'])
            acc = np.array(force) / s['mass']
            velocities[i] += acc * self.dt
            positions[i] += velocities[i] * self.dt
            s['position'] = tuple(positions[i])
            s['velocity'] = tuple(velocities[i])

    def animate(self, frame):
        for _ in range(self.steps_per_frame):
            self.update_positions()
        pos = np.array([s['position'] for s in self.stars])
        self.scat._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        return self.scat,

    def show(self):
        pos = np.array([s['position'] for s in self.stars])
        self.scat = self.ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=1)
        self.ax.set_title('Spiralgalaxie – Echtzeit N-Body')
        anim = FuncAnimation(self.fig, self.animate, interval=30, blit=False)
        plt.show()
