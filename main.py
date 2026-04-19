"""
main.py – Einstiegspunkt für die N-Body Galaxien-Simulation
════════════════════════════════════════════════════════════
Startet den Qt-Launcher mit vollständigem Physik-Parameter-Panel.
Dort können Galaxienanzahl (2–20), Partikelzahl (100–500 000) und
alle Simulations-Parameter eingestellt werden.

Fallback:  Falls PyQt5 nicht verfügbar ist, wird die Simulation direkt
           mit Standardwerten gestartet (3 Galaxien, 16 000 Partikel).

Steuerung (im Vispy-Fenster):
  SPACE      Pause / Weiter
  +  /  -    Zeitschritt dt erhöhen / verringern
  R          Zeitschritt zurücksetzen
  T          Statistik in Konsole ausgeben
  M          Farbmodus umschalten: Standard ↔ Metallizität
  Maus       Kamera drehen, Scrollen zum Zoomen
"""
import sys
import os

# Projekt-Root zum Suchpfad hinzufügen
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _run_launcher():
    """Qt-Launcher starten (Standardpfad)."""
    import vispy
    vispy.use('pyqt5')

    from PyQt5.QtWidgets import QApplication
    from editor.ui.SimulationLauncher import SimulationLauncher

    qt_app = QApplication.instance() or QApplication(sys.argv)
    launcher = SimulationLauncher()
    launcher.show()
    sys.exit(qt_app.exec_())


def _run_direct():
    """Direktstart ohne Qt-Launcher (Fallback wenn PyQt5 fehlt)."""
    from assets.presets.triple_galaxy_disks import generate_triple_galaxy_disks
    from editor.ui.GalaxySimVispyViewer import GalaxySimVispyViewer
    from vispy import app

    print("=== N-Body Galaxien-Simulation (Direktstart ohne Launcher) ===")
    print("Generiere 3 Galaxienscheiben …")
    galaxies, dm_cfgs = generate_triple_galaxy_disks(
        n_stars=16_000, disk_radius=80.0, sep=750.0)
    total = sum(len(g) for g in galaxies)
    print(f"Partikel gesamt: {total}  ({[len(g) for g in galaxies]})")

    GalaxySimVispyViewer(
        galaxies,
        dm_halo_configs=dm_cfgs,
        dt=0.7, G=1.0, eps=1.3, theta=0.65, steps_per_frame=1,
    )
    print("Simulation gestartet.")
    app.run()


def main():
    try:
        _run_launcher()
    except ImportError as exc:
        print(f"[Warnung] PyQt5 nicht verfügbar ({exc}). Starte ohne Launcher …")
        _run_direct()


if __name__ == "__main__":
    main()
