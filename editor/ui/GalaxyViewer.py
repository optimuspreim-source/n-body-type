"""
GalaxyViewer – Matplotlib-basierter Snapshot-Exporter
═══════════════════════════════════════════════════════════════
Erzeugt hochauflösende Standbilder der Simulation:
  • save_snapshot(pos, color)   – PNG/SVG aus numpy-Arrays
  • show_snapshot(pos, color)   – nicht-blockierendes Vorschaufenster
  • save_from_scene(scene)      – Legacy-API (scene.objects Liste)

Verwendung in GalaxySimVispyViewer:
    from editor.ui.GalaxyViewer import GalaxyViewer
    GalaxyViewer.save_snapshot(pos, colors, path='snapshot.png', dpi=300)
═══════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import os
import numpy as np


class GalaxyViewer:
    """
    Statischer Snapshot-Exporter. Alle Methoden können klassenmethodisch
    oder über eine Instanz aufgerufen werden.
    """

    # ── Haupt-API ──────────────────────────────────────────────────────────

    @staticmethod
    def save_snapshot(
        pos:    np.ndarray,
        color:  np.ndarray | None = None,
        mass:   np.ndarray | None = None,
        path:   str = 'snapshot.png',
        dpi:    int = 200,
        title:  str = 'N-Body Simulation',
        proj:   str = 'xy',         # 'xy' | 'xz' | 'yz' | '3d'
        figsize: tuple = (10, 10),
        bg_color: str  = '#05050f',
    ) -> str:
        """
        Speichert einen Snapshot als PNG oder SVG.

        Parameters
        ----------
        pos   : (N, 3) float64  Partikel-Positionen
        color : (N, 3) float32  RGB-Farben [0..1] (None = weiß)
        mass  : (N,) float64    Massen (für Punktgröße, None = uniform)
        path  : Ausgabepfad inkl. Endung .png oder .svg
        proj  : Projektionsebene oder '3d' für Axonometrie
        """
        import matplotlib
        matplotlib.use('Agg')   # kein Display-Backend nötig
        import matplotlib.pyplot as plt

        pos = np.asarray(pos, dtype=np.float64)
        active = np.ones(len(pos), dtype=bool)
        if mass is not None:
            active = np.asarray(mass) > 0.
        pos = pos[active]

        # Farben vorbereiten
        if color is None:
            c = np.ones((len(pos), 3), dtype=np.float32)
        else:
            c = np.asarray(color, dtype=np.float32)[active]
            c = np.clip(c, 0., 1.)

        # Punktgröße nach Masse
        if mass is not None:
            m = np.asarray(mass)[active]
            s = np.clip(np.log1p(m) * 0.3, 0.05, 2.5)
        else:
            s = np.full(len(pos), 0.3)

        fig = plt.figure(figsize=figsize, facecolor=bg_color)

        if proj == '3d':
            from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]  # noqa: F401
            ax = fig.add_subplot(111, projection='3d', facecolor=bg_color)
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                       c=c, s=s, linewidths=0, alpha=0.6)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        else:
            ax = fig.add_subplot(111, facecolor=bg_color)
            xi = {'xy': 0, 'xz': 0, 'yz': 1}[proj]
            yi = {'xy': 1, 'xz': 2, 'yz': 2}[proj]
            ax.scatter(pos[:, xi], pos[:, yi], c=c, s=s, linewidths=0, alpha=0.7)
            ax.set_xlabel(proj[0].upper())
            ax.set_ylabel(proj[1].upper())
            ax.set_aspect('equal', adjustable='datalim')

        ax.set_title(title, color='#8888ff', fontsize=12, pad=8)
        for spine in ax.spines.values():
            spine.set_color('#222244')
        ax.tick_params(colors='#555588')

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=bg_color)
        plt.close(fig)
        return os.path.abspath(path)

    @staticmethod
    def show_snapshot(
        pos:   np.ndarray,
        color: np.ndarray | None = None,
        mass:  np.ndarray | None = None,
        title: str = 'N-Body Simulation',
        proj:  str = 'xy',
        block: bool = False,
    ) -> None:
        """Zeigt einen nicht-blockierenden Vorschau-Plot (für Debugging)."""
        import matplotlib
        matplotlib.use('TkAgg' if block else 'Qt5Agg')
        import matplotlib.pyplot as plt

        pos = np.asarray(pos, dtype=np.float64)
        scatter_c: list | str
        if color is None:
            scatter_c = 'white'
        else:
            scatter_c = np.clip(np.asarray(color, dtype=np.float32), 0., 1.).tolist()  # type: ignore[assignment]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='#05050f')
        ax.set_facecolor('#05050f')
        xi = {'xy': 0, 'xz': 0, 'yz': 1}[proj]
        yi = {'xy': 1, 'xz': 2, 'yz': 2}[proj]
        ax.scatter(pos[:, xi], pos[:, yi], c=scatter_c, s=0.5, linewidths=0, alpha=0.8)
        ax.set_title(title, color='#8888ff')
        ax.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        plt.show(block=block)
        plt.pause(0.001)

    # ── Legacy-API (scene.objects-Format) ───────────────────────────────────

    @staticmethod
    def save_from_scene(scene, path: str = 'snapshot.png', **kw) -> str:
        """
        Erzeugt Snapshot aus einer Scene im alten scene.objects-Format
        (Liste von Dicts mit 'position' und optionalem 'color').
        """
        objects = getattr(scene, 'objects', scene)
        positions = []
        colors    = []
        for obj in objects:
            positions.append(obj['position'])
            colors.append(obj.get('color', (1., 1., 1.)))
        pos = np.array(positions, dtype=np.float64)
        col = np.array(colors,    dtype=np.float32)
        return GalaxyViewer.save_snapshot(pos, col, path=path, **kw)

    # ── Instanz-API (Rückwärtskompatibilität) ──────────────────────────────

    def __init__(self, scene=None):
        self.scene = scene

    def show(self, **kw) -> None:
        """Zeigt einen Snapshot der gespeicherten Scene (Legacy)."""
        if self.scene is not None:
            objects = getattr(self.scene, 'objects', self.scene)
            positions = [obj['position'] for obj in objects]
            colors    = [obj.get('color', (1., 1., 1.)) for obj in objects]
            GalaxyViewer.show_snapshot(
                np.array(positions), np.array(colors), **kw
            )
