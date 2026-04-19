"""
PhysicsControlPanel.py – Vollständiges Physik- & Simulations-Parameter-Panel
═════════════════════════════════════════════════════════════════════════════
Stellt alle steuerbaren Parameter der N-Body-Galaxien-Simulation als
PyQt5-Widget bereit.

Tabs
────
  Simulation   – Galaxienanzahl, Partikelzahl, Schritte/Frame
  Physik       – G, Barnes-Hut θ, Zeitschritt dt, Softening
  Galaxien     – Scheibengeometrie, Orbitaldynamik, SMBH
  Dunkle Mat.  – NFW-Halo-Parameter

Öffentliche API
───────────────
  panel.get_parameters() → dict
  panel.set_parameters(dict)
  panel.reset_defaults()
  Signal: parameters_changed(dict)
"""

import math

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QDoubleSpinBox, QSpinBox,
    QGroupBox, QCheckBox, QTabWidget, QScrollArea,
    QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal


# ─────────────────────────────────────────────────────────────────────────────
# Basis-Widget: Slider ↔ SpinBox (verknüpft, optional logarithmische Skala)
# ─────────────────────────────────────────────────────────────────────────────

class _SliderSpinBox(QWidget):
    """
    Horizontal gruppiertes Label | Slider | SpinBox.

    Optionen
    ────────
    log_scale   Slider-Position linear in log(value) – ideal für
                große Wertebereiche (z.B. Partikelzahl 100 … 500 000).
    decimals    0 → QSpinBox (Ganzzahl), > 0 → QDoubleSpinBox
    unit        Einheit-Suffix im SpinBox
    """

    value_changed = pyqtSignal(float)

    _STEPS = 400    # Slider-Auflösung (400 Schritte = ausreichende Präzision, viel weniger Events)

    def __init__(
        self,
        label:     str,
        min_v:     float,
        max_v:     float,
        default:   float,
        decimals:  int   = 2,
        log_scale: bool  = False,
        unit:      str   = '',
        parent=None,
    ):
        super().__init__(parent)
        self._log   = log_scale
        self._decs  = decimals
        self._min   = float(min_v)
        self._max   = float(max_v)
        self._block = False   # Rekursionsschutz
        # Gecachte Log-Werte: vermeidet math.log() bei jedem Slider-Event
        if log_scale:
            self._log_lo    = math.log(float(min_v))
            self._log_range = math.log(float(max_v)) - self._log_lo
        else:
            self._log_lo    = 0.0
            self._log_range = 1.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)

        # Label
        lbl = QLabel(label)
        lbl.setFixedWidth(160)
        lbl.setToolTip(f'Minimum: {min_v}   Maximum: {max_v}   Einheit: {unit or "–"}')
        layout.addWidget(lbl)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)  # type: ignore[attr-defined]
        self._slider.setRange(0, self._STEPS)
        self._slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self._slider)

        # SpinBox (Ganzzahl oder Dezimal)
        if decimals == 0:
            sb = QSpinBox()
            sb.setRange(int(min_v), int(max_v))
            if unit:
                sb.setSuffix(f' {unit}')
            sb.setFixedWidth(100)
            sb.valueChanged.connect(self._spin_int_changed)
            self._spin = sb
        else:
            sb = QDoubleSpinBox()
            sb.setRange(float(min_v), float(max_v))
            sb.setDecimals(decimals)
            sb.setSingleStep(10 ** (-decimals))
            if unit:
                sb.setSuffix(f' {unit}')
            sb.setFixedWidth(120)
            sb.valueChanged.connect(self._spin_float_changed)
            self._spin = sb

        layout.addWidget(self._spin)
        # sliderMoved: SpinBox live aktualisieren (visuell, KEIN Signal)
        # sliderReleased: erst beim Loslassen Signal emittieren → massiv weniger Events
        self._slider.sliderMoved.connect(self._slider_moved)
        self._slider.sliderReleased.connect(self._slider_released)

        # Startwert setzen
        self._set_value_internal(float(default))

    # ── Konversion: Wert ↔ Slider-Position ───────────────────────────────────

    def _val_to_pos(self, v: float) -> int:
        v = max(self._min, min(self._max, v))
        if self._log:
            return round((math.log(v) - self._log_lo) / self._log_range * self._STEPS)
        return round((v - self._min) / (self._max - self._min) * self._STEPS)

    def _pos_to_val(self, p: int) -> float:
        t = p / self._STEPS
        if self._log:
            return math.exp(self._log_lo + t * self._log_range)
        return self._min + t * (self._max - self._min)

    # ── Slot-Handler ──────────────────────────────────────────────────────────

    def _slider_moved(self, pos: int):
        """Slider bewegt (gedrückt): SpinBox visuell aktualisieren, KEIN Signal."""
        if self._block:
            return
        v = self._pos_to_val(pos)
        self._block = True
        if self._decs == 0:
            self._spin.setValue(int(round(v)))  # type: ignore[arg-type]
        else:
            self._spin.setValue(round(v, self._decs))  # type: ignore[arg-type]
        self._block = False

    def _slider_released(self):
        """Slider losgelassen: jetzt Signal emittieren (einmalig, nicht 400×)."""
        self.value_changed.emit(float(self._spin.value()))

    def _spin_float_changed(self, v: float):
        if self._block:
            return
        self._block = True
        self._slider.setValue(self._val_to_pos(v))
        self._block = False
        self.value_changed.emit(v)

    def _spin_int_changed(self, v: int):
        self._spin_float_changed(float(v))

    # ── Öffentliche API ───────────────────────────────────────────────────────

    def _set_value_internal(self, v: float):
        self._block = True
        self._slider.setValue(self._val_to_pos(v))
        if self._decs == 0:
            self._spin.setValue(int(round(v)))  # type: ignore[arg-type]
        else:
            self._spin.setValue(round(v, self._decs))  # type: ignore[arg-type]
        self._block = False

    def set_value(self, v: float):
        """Setzt Wert ohne Signal auszulösen."""
        self._set_value_internal(v)

    @property
    def value(self) -> float:
        return float(self._spin.value())


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Panel
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsControlPanel(QWidget):
    """
    Vollständiges Physik- und Simulations-Parameter-Kontrolpanel.

    Verwendung
    ──────────
        panel = PhysicsControlPanel()
        panel.parameters_changed.connect(my_callback)
        params = panel.get_parameters()   # dict mit allen Werten
    """

    parameters_changed = pyqtSignal(dict)

    # ── Default-Werte ─────────────────────────────────────────────────────────
    DEFAULTS: dict = {
        # Simulation
        'n_galaxies':      3,
        'n_stars':         50_000,
        'steps_per_frame': 1,
        # Physik
        'G':               1.0,
        'theta':           0.65,
        'dt':              0.7,
        'eps_star':        1.2,
        'eps_bh':          6.0,
        # Galaxien
        'disk_radius':     80.0,
        'sep':             750.0,
        'eccentricity':    0.78,
        'bh_mass':         14_000.0,
        'thickness_ratio': 0.012,
        'v_dispersion':    0.045,
        # Dunkle Materie
        'dm_enabled':      True,
        'dm_M_vir':        520_000.0,
        'dm_r_s':          180.0,
        'dm_c':            12.0,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets: dict[str, _SliderSpinBox] = {}
        self._dm_chk: QCheckBox | None = None
        self._dm_group_widgets: list[_SliderSpinBox] = []
        # Debounce-Timer: bei schnellen Eingaben (Tastatur, SpinBox) nur einmal
        # pro 60 ms das teure parameters_changed-Signal auslösen.
        self._debounce: QTimer = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(60)
        self._debounce.timeout.connect(self._emit_params)
        self._build()

    # ─────────────────────────────────────────────────────────────────────────
    # Aufbau
    # ─────────────────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        tabs = QTabWidget()
        tabs.addTab(self._tab_simulation(), 'Simulation')
        tabs.addTab(self._tab_physics(),    'Physik')
        tabs.addTab(self._tab_galaxies(),   'Galaxien')
        tabs.addTab(self._tab_dm(),         'Dunkle Materie')
        root.addWidget(tabs)

    # ── Tab: Simulation ───────────────────────────────────────────────────────

    def _tab_simulation(self) -> QWidget:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setSpacing(10)
        lay.setContentsMargins(8, 8, 8, 8)

        # Partikel & Galaxien
        g = self._group('Partikel & Galaxien')
        gl = QVBoxLayout(g)
        gl.addWidget(self._register(
            'n_galaxies', 'Galaxien-Anzahl', 2, 20, 3, 0,
            log_scale=False))
        gl.addWidget(self._register(
            'n_stars', 'Partikel gesamt', 100, 500_000, 50_000, 0,
            log_scale=True))
        lay.addWidget(g)

        # Zeitsteuerung
        g2 = self._group('Zeitsteuerung')
        g2l = QVBoxLayout(g2)
        g2l.addWidget(self._register(
            'steps_per_frame', 'Schritte / Frame', 1, 20, 1, 0))
        lay.addWidget(g2)

        lay.addStretch()
        sa.setWidget(inner)
        return sa

    # ── Tab: Physik ───────────────────────────────────────────────────────────

    def _tab_physics(self) -> QWidget:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setSpacing(10)
        lay.setContentsMargins(8, 8, 8, 8)

        # Gravitation
        g = self._group('Gravitation')
        gl = QVBoxLayout(g)
        gl.addWidget(self._register(
            'G', 'Gravitationskonstante G', 0.01, 10.0, 1.0, 3,
            log_scale=True))
        gl.addWidget(self._register(
            'theta', 'Barnes-Hut θ', 0.10, 1.50, 0.65, 2))
        lay.addWidget(g)

        # Integration
        g2 = self._group('Zeitintegration')
        g2l = QVBoxLayout(g2)
        g2l.addWidget(self._register(
            'dt', 'Zeitschritt dt', 0.01, 5.0, 0.7, 3,
            log_scale=True))
        lay.addWidget(g2)

        # Softening
        g3 = self._group('Gravitations-Softening')
        g3l = QVBoxLayout(g3)
        g3l.addWidget(self._register(
            'eps_star', 'Softening Sterne  ε★', 0.1, 20.0, 1.2, 2, unit='lu'))
        g3l.addWidget(self._register(
            'eps_bh',   'Softening SMBH    ε●', 0.5, 50.0, 6.0, 2, unit='lu'))
        lay.addWidget(g3)

        lay.addStretch()
        sa.setWidget(inner)
        return sa

    # ── Tab: Galaxien ─────────────────────────────────────────────────────────

    def _tab_galaxies(self) -> QWidget:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setSpacing(10)
        lay.setContentsMargins(8, 8, 8, 8)

        # Geometrie
        g = self._group('Scheiben-Geometrie')
        gl = QVBoxLayout(g)
        gl.addWidget(self._register(
            'disk_radius', 'Scheibenradius', 10.0, 500.0, 80.0, 1, unit='lu'))
        gl.addWidget(self._register(
            'sep', 'Galaxienabstand', 100.0, 5000.0, 750.0, 0,
            log_scale=True, unit='lu'))
        gl.addWidget(self._register(
            'thickness_ratio', 'Dicken-Verhältnis', 0.003, 0.15, 0.012, 4))
        lay.addWidget(g)

        # Orbitaldynamik
        g2 = self._group('Orbitale Anfangsbedingungen')
        g2l = QVBoxLayout(g2)
        g2l.addWidget(self._register(
            'eccentricity', 'Exzentrizität  e', 0.0, 0.99, 0.78, 2))
        g2l.addWidget(self._register(
            'v_dispersion', 'Geschwindigkeitsdispersion', 0.0, 0.3, 0.045, 3))
        lay.addWidget(g2)

        # SMBH
        g3 = self._group('Supermassives Schwarzes Loch (SMBH)')
        g3l = QVBoxLayout(g3)
        g3l.addWidget(self._register(
            'bh_mass', 'SMBH-Masse  M\u25cf', 0.0, 200_000.0, 14_000.0, 0))
        lay.addWidget(g3)

        lay.addStretch()
        sa.setWidget(inner)
        return sa

    # ── Tab: Dunkle Materie ───────────────────────────────────────────────────

    def _tab_dm(self) -> QWidget:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setSpacing(10)
        lay.setContentsMargins(8, 8, 8, 8)

        g = self._group('NFW Dunkle-Materie-Halos')
        gl = QVBoxLayout(g)

        # Aktivierungs-Checkbox
        self._dm_chk = QCheckBox('DM-Halos aktivieren  (NFW-Profil)')
        self._dm_chk.setChecked(True)
        self._dm_chk.stateChanged.connect(self._dm_toggled)
        gl.addWidget(self._dm_chk)

        sep_line = QFrame()
        sep_line.setFrameShape(QFrame.HLine)
        sep_line.setFrameShadow(QFrame.Sunken)
        gl.addWidget(sep_line)

        w_mvir = self._register(
            'dm_M_vir', 'Viriale Masse  M_vir',
            1_000.0, 10_000_000.0, 520_000.0, 0, log_scale=True)
        w_rs = self._register(
            'dm_r_s', 'Skalenradius  r_s',
            10.0, 2_000.0, 180.0, 0, unit='lu')
        w_c = self._register(
            'dm_c', 'Konzentrations-Parameter  c',
            2.0, 40.0, 12.0, 1)

        gl.addWidget(w_mvir)
        gl.addWidget(w_rs)
        gl.addWidget(w_c)

        # Merke DM-Widgets für Enable/Disable
        self._dm_group_widgets = [w_mvir, w_rs, w_c]

        lay.addWidget(g)

        # Info-Label
        info = QLabel(
            'Hinweis: NFW-Halos erzeugen flache Rotationskurven.\n'
            'M_vir ≈ 8–10 × M_baryon  ist physikalisch realistisch.\n'
            'r_s ≈ 2 × disk_radius  als Faustregel.')
        info.setStyleSheet('color: #7070a0; font-size: 10px;')
        info.setWordWrap(True)
        lay.addWidget(info)

        lay.addStretch()
        sa.setWidget(inner)
        return sa

    # ─────────────────────────────────────────────────────────────────────────
    # Hilfsmethoden
    # ─────────────────────────────────────────────────────────────────────────

    def _group(self, title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet('QGroupBox { font-weight: bold; }')
        return g

    def _register(
        self,
        key:       str,
        label:     str,
        min_v:     float,
        max_v:     float,
        default:   float,
        decimals:  int,
        log_scale: bool = False,
        unit:      str  = '',
    ) -> _SliderSpinBox:
        w = _SliderSpinBox(
            label, min_v, max_v, default,
            decimals=decimals, log_scale=log_scale, unit=unit)
        w.value_changed.connect(lambda v, k=key: self._on_widget_changed(k, v))
        self._widgets[key] = w
        return w

    def _on_widget_changed(self, _key: str, _val: float):
        # Timer (neu)starten: läuft bei schnellen Änderungen durch, bis 60 ms Ruhe
        self._debounce.start()

    def _emit_params(self):
        """Vom Debounce-Timer aufgerufen: einmalig Signal senden."""
        self.parameters_changed.emit(self.get_parameters())

    def _dm_toggled(self, state: int):
        enabled = bool(state)
        for w in self._dm_group_widgets:
            w.setEnabled(enabled)
        self._debounce.start()

    # ─────────────────────────────────────────────────────────────────────────
    # Öffentliche API
    # ─────────────────────────────────────────────────────────────────────────

    def get_parameters(self) -> dict:
        """Gibt alle aktuellen Parameter als dict zurück."""
        params: dict = {}
        for key, w in self._widgets.items():
            params[key] = w.value
        # dm_enabled kommt aus Checkbox
        if self._dm_chk is not None:
            params['dm_enabled'] = self._dm_chk.isChecked()
        return params

    def set_parameters(self, params: dict):
        """Setzt Parameter aus einem dict (unbekannte Schlüssel werden ignoriert)."""
        for key, w in self._widgets.items():
            if key in params:
                w.set_value(float(params[key]))
        if 'dm_enabled' in params and self._dm_chk is not None:
            self._dm_chk.setChecked(bool(params['dm_enabled']))
            self._dm_toggled(int(bool(params['dm_enabled'])))

    def reset_defaults(self):
        """Setzt alle Parameter auf Default-Werte zurück."""
        self.set_parameters(self.DEFAULTS)
