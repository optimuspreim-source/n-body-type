"""
SimulationLauncher.py – Vollständiges Startfenster der N-Body-Simulation
═════════════════════════════════════════════════════════════════════════
Layout
──────
  Links  : PhysicsControlPanel (Tabs mit allen Parametern)
  Rechts : Info-Panel (Live-Statistik, Performance-Schätzung, Shortcuts)
  Unten  : [Zurücksetzen]  [▶ Simulation starten]

Workflow
────────
  1. Benutzer stellt Parameter ein → Live-Vorschau aktualisiert sich.
  2. Klick auf "Simulation starten" → Preset wird erzeugt, Vispy-Fenster öffnet.
  3. Launcher bleibt offen; neue Simulation kann gestartet oder Parameter
     können geändert und ein zweites Fenster geöffnet werden.

Abhängigkeiten
──────────────
  PyQt5  (pip install PyQt5)
  vispy  (pip install vispy)
  numpy  (pip install numpy)
"""

from __future__ import annotations

import math

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFrame, QScrollArea,
    QPlainTextEdit, QSizePolicy, QApplication, QMessageBox,
    QMenuBar, QAction, QStatusBar,
)
from PyQt5.QtCore import Qt

from editor.ui.PhysicsControlPanel import PhysicsControlPanel


# ─────────────────────────────────────────────────────────────────────────────
# Dark-Theme Stylesheet
# ─────────────────────────────────────────────────────────────────────────────

_STYLE = """
/* ── Fenster & allgemeine Widgets ── */
QMainWindow, QDialog {
    background-color: #12121f;
    color: #dcdcef;
}
QWidget {
    background-color: #12121f;
    color: #dcdcef;
    font-size: 12px;
}

/* ── Gruppen-Rahmen ── */
QGroupBox {
    border: 1px solid #2e2e52;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 6px;
    color: #9090cc;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #7070bb;
}

/* ── Tabs ── */
QTabWidget::pane {
    border: 1px solid #2e2e52;
    border-top: none;
}
QTabBar::tab {
    background: #1c1c35;
    color: #7070a0;
    padding: 5px 14px;
    border: 1px solid #2e2e52;
    border-bottom: none;
    border-radius: 3px 3px 0 0;
    min-width: 80px;
}
QTabBar::tab:selected {
    background: #12121f;
    color: #b0b0ff;
    border-bottom: 2px solid #5555cc;
}
QTabBar::tab:hover:!selected {
    background: #20203a;
    color: #9090c0;
}

/* ── Slider ── */
QSlider::groove:horizontal {
    height: 4px;
    background: #2e2e52;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #5555bb;
    width: 13px;
    height: 13px;
    margin: -5px 0;
    border-radius: 7px;
    border: 1px solid #7777dd;
}
QSlider::sub-page:horizontal {
    background: #3a3a88;
    border-radius: 2px;
}
QSlider::handle:horizontal:hover {
    background: #7777dd;
}

/* ── SpinBoxen ── */
QDoubleSpinBox, QSpinBox {
    background: #1c1c35;
    color: #dcdcef;
    border: 1px solid #3a3a70;
    border-radius: 3px;
    padding: 2px 5px;
}
QDoubleSpinBox:focus, QSpinBox:focus {
    border-color: #6666bb;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {
    background: #2a2a50;
    border: none;
    width: 16px;
}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {
    background: #3a3a70;
}

/* ── Buttons ── */
QPushButton {
    background-color: #1e1e3c;
    color: #a0a0d0;
    border: 1px solid #3a3a70;
    border-radius: 4px;
    padding: 5px 16px;
}
QPushButton:hover {
    background-color: #2a2a58;
    border-color: #6060aa;
    color: #c0c0f0;
}
QPushButton:pressed {
    background-color: #141430;
}
QPushButton#start_btn {
    background-color: #0f2f0f;
    color: #60ee60;
    border: 1px solid #30aa30;
    font-weight: bold;
    font-size: 13px;
    padding: 7px 28px;
    min-height: 36px;
}
QPushButton#start_btn:hover {
    background-color: #1a4a1a;
    border-color: #55cc55;
    color: #80ff80;
}
QPushButton#reset_btn {
    background-color: #2a1a0f;
    color: #e0a060;
    border: 1px solid #7a4a20;
}
QPushButton#reset_btn:hover {
    background-color: #3a2a18;
    border-color: #bb7733;
}

/* ── Checkbox ── */
QCheckBox {
    color: #b0b0d8;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #4040a0;
    border-radius: 3px;
    background: #1c1c35;
}
QCheckBox::indicator:checked {
    background: #4444aa;
    border-color: #7777cc;
}

/* ── ScrollArea ── */
QScrollArea {
    border: none;
}
QScrollBar:vertical {
    background: #12121f;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #2e2e52;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #4040a0;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ── PlainTextEdit (Shortcuts-Box) ── */
QPlainTextEdit {
    background: #0a0a18;
    color: #7070a0;
    border: 1px solid #1e1e38;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11px;
}

/* ── Labels ── */
QLabel#title_lbl {
    color: #8888ff;
    font-size: 17px;
    font-weight: bold;
    qproperty-alignment: AlignCenter;
}
QLabel#stats_lbl {
    color: #55dd55;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11px;
}
QLabel#hint_lbl {
    color: #5050a0;
    font-size: 10px;
}

/* ── Trennlinie ── */
QFrame[frameShape="4"] {
    color: #2a2a50;
}

/* ── Statusleiste ── */
QStatusBar {
    background: #0a0a18;
    color: #505070;
    font-size: 11px;
}
QStatusBar::item {
    border: none;
}
"""

_SHORTCUTS = """\
Tastatursteuerung (während der Simulation):

  SPACE      Pause / Weiter
  +  /  -    Zeitschritt dt ×1.4 / ÷1.4
  R          dt auf Startwert zurücksetzen
  T          Statistik in Konsole ausgeben
  M          Farbmodus: Standard ↔ Metallizität
  Maus       Kamera drehen (linke Taste)
  Scroll     Zoom

Legende (Farbmodi):
  Standard     Galaxienfarbe pro Scheibe
  Metallizität Blau (metallarm) → Rot (metallreich)

Hinweise:
  • Große Partikelzahlen benötigen mehr RAM und GPU.
  • Backend wechselt automatisch:
      N ≤ 2 000  → NumPy  O(N²)
      N > 2 000  → Numba Barnes-Hut  O(N log N)
  • Erste JIT-Kompilierung kann einige Sekunden dauern.
"""

_ABOUT_TEXT = """\
N-Body Galaxien-Simulation
Version 1.0  –  April 2026

Physik-Backend:
  Barnes-Hut O(N log N) mit Numba JIT
  Leapfrog-Integration (symplektisch)
  NFW-Dunkle-Materie-Halos
  AGN-Jets & Stellares Feedback

Rendering:
  Vispy (OpenGL)  –  Echtzeit-3D
"""


# ─────────────────────────────────────────────────────────────────────────────
# SimulationLauncher
# ─────────────────────────────────────────────────────────────────────────────

class SimulationLauncher(QMainWindow):
    """
    Vollständiges Startfenster der N-Body-Galaxien-Simulation.

    Öffnet sich beim Programmstart und erlaubt die Konfiguration aller
    Physik- und Simulations-Parameter, bevor die Vispy-Simulation gestartet wird.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('N-Body Galaxien-Simulation  –  Konfiguration')
        self.resize(1000, 660)
        self.setMinimumSize(760, 520)
        self.setStyleSheet(_STYLE)

        self._viewers: list = []   # Referenzen auf aktive Viewer (verhindert GC)
        self._sb: QStatusBar       # wird in _build_statusbar() gesetzt
        self._last_stats: str = ''  # Cache: vermeidet redundante setText-Calls
        self._last_sb_msg: str = ''

        self._build_menu()
        self._build_central()
        self._build_statusbar()

        # Initialer Status
        self._ctrl.parameters_changed.connect(self._on_params_changed)
        self._on_params_changed(self._ctrl.get_parameters())

    # ─────────────────────────────────────────────────────────────────────────
    # Aufbau
    # ─────────────────────────────────────────────────────────────────────────

    def _build_menu(self):
        bar = self.menuBar()

        # Datei-Menü
        file_menu = bar.addMenu('Datei')
        act_new = QAction('Neue Simulation', self)
        act_new.setShortcut('Ctrl+N')
        act_new.triggered.connect(self._start_simulation)
        file_menu.addAction(act_new)

        act_reset = QAction('Parameter zurücksetzen', self)
        act_reset.setShortcut('Ctrl+R')
        act_reset.triggered.connect(self._reset_defaults)
        file_menu.addAction(act_reset)

        file_menu.addSeparator()
        act_quit = QAction('Beenden', self)
        act_quit.setShortcut('Ctrl+Q')
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Hilfe-Menü
        help_menu = bar.addMenu('Hilfe')
        act_about = QAction('Über …', self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(12)

        # ── Links: Parameterpanel ─────────────────────────────────────────────
        self._ctrl = PhysicsControlPanel()
        root.addWidget(self._ctrl, stretch=3)

        # ── Vertikale Trennlinie ──────────────────────────────────────────────
        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setFrameShadow(QFrame.Sunken)
        root.addWidget(vline)

        # ── Rechts: Info-Panel ────────────────────────────────────────────────
        right = QWidget()
        right.setMaximumWidth(270)
        right.setMinimumWidth(220)
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(4, 0, 0, 0)
        rlay.setSpacing(8)

        # Titel
        title = QLabel('N-Body\nGalaxien-Simulation')
        title.setObjectName('title_lbl')
        rlay.addWidget(title)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine); sep1.setFrameShadow(QFrame.Sunken)
        rlay.addWidget(sep1)

        # Live-Statistik
        stat_hdr = QLabel('Simulations-Info:')
        stat_hdr.setStyleSheet('color: #6060aa; font-weight: bold;')
        rlay.addWidget(stat_hdr)

        self._stats_lbl = QLabel()
        self._stats_lbl.setObjectName('stats_lbl')
        self._stats_lbl.setWordWrap(True)
        self._stats_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._stats_lbl.setMinimumHeight(130)
        rlay.addWidget(self._stats_lbl)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setFrameShadow(QFrame.Sunken)
        rlay.addWidget(sep2)

        # Tastaturkürzel
        sc_hdr = QLabel('Steuerung:')
        sc_hdr.setStyleSheet('color: #6060aa; font-weight: bold;')
        rlay.addWidget(sc_hdr)

        sc_box = QPlainTextEdit()
        sc_box.setReadOnly(True)
        sc_box.setPlainText(_SHORTCUTS)
        sc_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rlay.addWidget(sc_box, stretch=1)

        # ── Button-Zeile ──────────────────────────────────────────────────────
        btn_row = QWidget()
        btn_lay = QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 4, 0, 0)
        btn_lay.setSpacing(8)

        reset_btn = QPushButton('↺  Zurücksetzen')
        reset_btn.setObjectName('reset_btn')
        reset_btn.setFixedHeight(34)
        reset_btn.clicked.connect(self._reset_defaults)
        reset_btn.setToolTip('Alle Parameter auf Standardwerte zurücksetzen  (Ctrl+R)')
        btn_lay.addWidget(reset_btn)

        btn_lay.addStretch()

        start_btn = QPushButton('▶  Simulation starten')
        start_btn.setObjectName('start_btn')
        start_btn.setFixedHeight(40)
        start_btn.clicked.connect(self._start_simulation)
        start_btn.setToolTip('Startet die Simulation mit den aktuellen Parametern  (Ctrl+N)')
        btn_lay.addWidget(start_btn)

        rlay.addWidget(btn_row)
        root.addWidget(right, stretch=0)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._sb = sb   # direkte Referenz, da statusBar() ggf. None zurückgibt

    # ─────────────────────────────────────────────────────────────────────────
    # Slot-Handler
    # ─────────────────────────────────────────────────────────────────────────

    def _on_params_changed(self, params: dict):
        """Aktualisiert das Info-Panel live bei jeder Parameteränderung."""
        n_gal   = max(2, int(round(params.get('n_galaxies', 3))))
        n_total = max(100, int(round(params.get('n_stars', 50_000))))
        dm_en   = bool(params.get('dm_enabled', True))
        G       = float(params.get('G', 1.0))
        dt      = float(params.get('dt', 0.7))
        theta   = float(params.get('theta', 0.65))
        eps_s   = float(params.get('eps_star', 1.2))
        spf     = int(round(params.get('steps_per_frame', 1)))

        # Backend-Schätzung
        if n_total <= 2_000:
            backend  = 'NumPy  O(N²)'
            fps_est  = max(1, int(60 * math.sqrt(2_000 / max(n_total, 1))))
        else:
            backend  = 'Numba Barnes-Hut  O(N log N)'
            fps_est  = max(1, int(35 * (50_000 / max(n_total, 1)) ** 0.65))
        fps_est = min(fps_est, 120)

        # Sterne pro Galaxie
        sterne_pro = n_total // n_gal

        text = (
            f'Galaxien:          {n_gal}\n'
            f'Partikel gesamt:   {n_total:,}\n'
            f'Sterne / Galaxie:  {sterne_pro:,}\n'
            f'Backend:           {backend}\n'
            f'Est. FPS:          ~{fps_est}\n'
            f'Schritte/Frame:    {spf}\n'
            f'DM-Halos:          {"✓ aktiv" if dm_en else "✗ deaktiviert"}\n'
            f'\n'
            f'G  = {G:.3f}\n'
            f'dt = {dt:.3f}\n'
            f'θ  = {theta:.2f}\n'
            f'ε★ = {eps_s:.2f} lu\n'
        )
        # Nur neu setzen wenn sich tatsächlich etwas geändert hat (spart Repaint)
        if text != self._last_stats:
            self._stats_lbl.setText(text)
            self._last_stats = text

        sb_msg = (
            f'  {n_gal} Galaxien  \u00b7  {n_total:,} Partikel  \u00b7  '
            f'Backend: {backend}  \u00b7  Est. {fps_est} FPS')
        if sb_msg != self._last_sb_msg:
            self._sb.showMessage(sb_msg)
            self._last_sb_msg = sb_msg

    def _reset_defaults(self):
        self._ctrl.reset_defaults()

    def _show_about(self):
        QMessageBox.information(self, 'Über diese Simulation', _ABOUT_TEXT)

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation starten
    # ─────────────────────────────────────────────────────────────────────────

    def _start_simulation(self):
        params = self._ctrl.get_parameters()

        # Cursor auf Warten schalten
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage('  Simulation wird initialisiert …')
        QApplication.processEvents()

        try:
            self._launch(params)
            self._sb.showMessage(
                f'  Simulation gestartet  ·  '
                f'{int(params.get("n_galaxies", 3))} Galaxien  ·  '
                f'{int(params.get("n_stars", 50000)):,} Partikel')
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self, 'Fehler beim Starten',
                f'Die Simulation konnte nicht gestartet werden:\n\n{exc}')
            self._sb.showMessage('  Fehler – Simulation nicht gestartet.')
            raise
        finally:
            QApplication.restoreOverrideCursor()

    def _launch(self, params: dict):
        """Erzeugt Preset und öffnet GalaxySimVispyViewer."""
        # Lazy import: vispy-Backend muss vorher gesetzt sein (siehe main.py)
        from assets.presets.n_galaxy_disks import generate_n_galaxy_disks
        from editor.ui.GalaxySimVispyViewer import GalaxySimVispyViewer

        n_gal   = max(2, min(20, int(round(params['n_galaxies']))))
        n_stars = max(100, int(round(params['n_stars'])))
        dm_en   = bool(params.get('dm_enabled', True))
        eps_s   = float(params.get('eps_star', 1.2))
        eps_bh  = float(params.get('eps_bh',   6.0))

        print(
            f'\n[Launcher] ══════ Neue Simulation ══════\n'
            f'[Launcher] Galaxien: {n_gal}  |  Partikel gesamt: {n_stars:,}\n'
            f'[Launcher] G={params["G"]:.3f}  dt={params["dt"]:.3f}  '
            f'θ={params["theta"]:.2f}  ε★={eps_s:.2f}  ε●={eps_bh:.2f}')

        # ── Preset generieren ────────────────────────────────────────────────
        galaxies, dm_cfgs = generate_n_galaxy_disks(
            n_galaxies      = n_gal,
            n_stars         = n_stars,
            disk_radius     = float(params.get('disk_radius',     80.0)),
            sep             = float(params.get('sep',            750.0)),
            eccentricity    = float(params.get('eccentricity',    0.78)),
            G               = float(params.get('G',               1.0)),
            dm_enabled      = dm_en,
            dm_M_vir        = float(params.get('dm_M_vir',   520_000.0)),
            dm_r_s          = float(params.get('dm_r_s',         180.0)),
            dm_c            = float(params.get('dm_c',            12.0)),
            bh_mass         = float(params.get('bh_mass',      14_000.0)),
            thickness_ratio = float(params.get('thickness_ratio',  0.012)),
            v_dispersion    = float(params.get('v_dispersion',    0.045)),
        )

        # Softening per Partikel überschreiben (UI-Werte haben Priorität)
        for gal in galaxies:
            for s in gal:
                s['softening'] = eps_bh if s.get('is_bh', False) else eps_s

        # ── Viewer erstellen (öffnet Vispy-Fenster) ──────────────────────────
        viewer = GalaxySimVispyViewer(
            galaxies,
            dm_halo_configs  = dm_cfgs if dm_en else None,
            dt               = float(params.get('dt',             0.7)),
            G                = float(params.get('G',              1.0)),
            eps              = eps_s,
            theta            = float(params.get('theta',          0.65)),
            steps_per_frame  = max(1, int(round(params.get('steps_per_frame', 1)))),
        )

        # Referenz halten → verhindert vorzeitigen GC
        self._viewers.append(viewer)
        print(f'[Launcher] Vispy-Fenster geöffnet.')

    # ─────────────────────────────────────────────────────────────────────────
    # Fenster schließen
    # ─────────────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # Alle aktiven Viewer schließen
        for v in self._viewers:
            try:
                v.canvas.close()
            except Exception:
                pass
        self._viewers.clear()
        event.accept()
