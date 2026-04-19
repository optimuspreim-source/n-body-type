"""
ParticleMesh.py  –  PM (Particle Mesh) Gravitations-Löser
════════════════════════════════════════════════════════════════════════════════
Algorithmus (PM)
────────────────
  1. Dichte-Assignment auf reguläres 3D-Gitter  (CIC – Cloud-In-Cell)
     Jedes Partikel verteilt seine Masse bilinear auf 8 Nachbar-Zellen.
     → δ(x,y,z)  Gitter-Dichte

  2. Poisson-Gleichung im Fourier-Raum  (FFT + Greenscher Kern)
     ∇²Φ = 4π G ρ
     Φ̂(k) = -4π G ρ̂(k) / k²       (Gitter-Greens-Funktion)

  3. Gradienten-Berechnung im Fourier-Raum  →  Kräfte
     a_i = -∇Φ  →  â(k) = -ik · Φ̂(k)    (zentrale Differenz)

  4. Force-Interpolation (CIC)  →  Partikel-Beschleunigungen

Komplexität: O(N + M³ log M)  (M = Gitterpunkte pro Achse)
Vorteil:    Sehr schnell für N >> M³  –  skaliert nicht mit N²
Nachteil:   Auflösungs-Untergrenze bei Gitterabstand h = L/M
            Kein Nahfeld: kombiniere mit BH oder direktem P3M für close encounters

Verwendung:
    pm = ParticleMeshSolver(N_grid=128, box_size=2500., G=1.0)
    pos_new, vel_new = pm.step(pos, vel, mass, dt)

Tastatur-Backend-Wechsel in Viewer: Taste 'B'
════════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from numba import njit, prange  # type: ignore[import-untyped]

# ─ CuPy-GPU-Erkennung (cuFFT ≈10–100× schneller als numpy.fft für 128³/256³-Gitter) ─
try:
    import cupy as _cp  # type: ignore
    _cp.zeros(1)        # testet ob CUDA tatsächlich verfügbar ist
    GPU_AVAILABLE = True
except Exception:
    _cp = None
    GPU_AVAILABLE = False

# ─ scipy.fft: multithreaded FFT (3–8× schneller als numpy.fft, nutzt alle CPU-Kerne) ─
try:
    import os as _os
    from scipy.fft import rfftn as _rfftn, irfftn as _irfftn  # type: ignore
    _FFT_KW = {'workers': max(2, (_os.cpu_count() or 4) // 2)}   # halbe Kernzahl
except ImportError:
    _rfftn  = np.fft.rfftn
    _irfftn = np.fft.irfftn
    _FFT_KW = {}   # numpy.fft kennt kein workers-Argument


# ─────────────────────────────────────────────────────────────────────────────
#  CIC Masse-Assignment  (Numba, parallel)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _cic_deposit(pos, mass, rho, M, h, x0, y0, z0):
    """
    Cloud-In-Cell Masse-Assignment.
    pos  : (N,3) float64   – Positionen
    mass : (N,)  float64   – Massen
    rho  : (M,M,M) float64  – Gitter (in-place akkumuliert, vorher 0 setzen!)
    h    : Gitterzellen-Größe
    x0,y0,z0 : untere Gitterecke
    """
    N = pos.shape[0]
    Mf = float(M)
    inv_h = 1.0 / h

    for i in range(N):
        if mass[i] <= 0.: continue

        # Gitter-Koordinaten (kontinuierlich, 0..M-1)
        gx = (pos[i, 0] - x0) * inv_h - 0.5
        gy = (pos[i, 1] - y0) * inv_h - 0.5
        gz = (pos[i, 2] - z0) * inv_h - 0.5

        # Ganzzahl-Index des linken Gitterpunkts
        ix = int(np.floor(gx))
        iy = int(np.floor(gy))
        iz = int(np.floor(gz))

        # Fractional offset [0, 1)
        dx = gx - ix
        dy = gy - iy
        dz = gz - iz

        # CIC-Gewichte: bilineares 3D-Interpolationsschema
        wx0 = 1.0 - dx;  wx1 = dx
        wy0 = 1.0 - dy;  wy1 = dy
        wz0 = 1.0 - dz;  wz1 = dz

        m = mass[i]
        # Periodische Indizes (wrap-around)
        for di in range(2):
            jx = (ix + di) % M
            wx = wx0 if di == 0 else wx1
            for dj in range(2):
                jy = (iy + dj) % M
                wy = wy0 if dj == 0 else wy1
                for dk in range(2):
                    jz = (iz + dk) % M
                    wz = wz0 if dk == 0 else wz1
                    rho[jx, jy, jz] += m * wx * wy * wz


@njit(cache=True, fastmath=True, parallel=True)
def _cic_interpolate(pos, mass, force_grid, accel, M, h, x0, y0, z0):
    """
    CIC Force-Interpolation  vom Gitter auf Partikel.
    force_grid : (M,M,M,3) float64  – Kräfte auf Gitter
    accel      : (N,3) float64       – Output (in-place geschrieben)
    Parallelisiert mit prange – jedes Partikel schreibt in seinen eigenen Index.
    """
    N = pos.shape[0]
    inv_h = 1.0 / h

    for i in prange(N):   # parallel: jede Zeile in accel gehört exklusiv zu Partikel i
        if mass[i] <= 0.:
            accel[i, 0] = 0.; accel[i, 1] = 0.; accel[i, 2] = 0.
            continue

        gx = (pos[i, 0] - x0) * inv_h - 0.5
        gy = (pos[i, 1] - y0) * inv_h - 0.5
        gz = (pos[i, 2] - z0) * inv_h - 0.5

        ix = int(np.floor(gx))
        iy = int(np.floor(gy))
        iz = int(np.floor(gz))

        dx = gx - ix
        dy = gy - iy
        dz = gz - iz

        wx0 = 1.0 - dx;  wx1 = dx
        wy0 = 1.0 - dy;  wy1 = dy
        wz0 = 1.0 - dz;  wz1 = dz

        ax = ay = az = 0.0
        for di in range(2):
            jx = (ix + di) % M
            wx = wx0 if di == 0 else wx1
            for dj in range(2):
                jy = (iy + dj) % M
                wy = wy0 if dj == 0 else wy1
                for dk in range(2):
                    jz = (iz + dk) % M
                    wz = wz0 if dk == 0 else wz1
                    w = wx * wy * wz
                    ax += w * force_grid[jx, jy, jz, 0]
                    ay += w * force_grid[jx, jy, jz, 1]
                    az += w * force_grid[jx, jy, jz, 2]
        accel[i, 0] = ax
        accel[i, 1] = ay
        accel[i, 2] = az


# ─────────────────────────────────────────────────────────────────────────────
#  PM-Löser-Klasse
# ─────────────────────────────────────────────────────────────────────────────

class ParticleMeshSolver:
    """
    PM-Solver (Particle Mesh) für N-Body-Gravitation.

    Parameter
    ---------
    N_grid    : Gitterpunkte pro Achse (z. B. 64, 128, 256)
                Auflösung h = box_size / N_grid
                Empfehlung: N_grid ≥ ∛N  für gute Sampling-Dichte
    box_size  : physikalische Ausdehnung der simulierten Box (dynamisch angepasst)
    G         : Gravitationskonstante (Simulationseinheiten)
    eps       : Softening-Länge (verhindert k→∞ Divergenz im Greens-Kern)
    padding   : relativer Rand über System-Ausdehnung (Standard 0.25 = 25 %)
    """

    def __init__(self, N_grid=128, box_size=3000., G=1.0, eps=1.3, padding=0.25):
        self.M       = int(N_grid)
        self.G       = G
        self.eps     = eps
        self.padding = padding

        # Gitter-Arrays (einmalig allokiert, Größe konstant)
        M = self.M
        self._rho         = np.zeros((M, M, M), dtype=np.float64)
        self._force_grid  = np.zeros((M, M, M, 3), dtype=np.float64)
        self._accel_out   = None   # (N, 3), lazy-alloc bei erstem Aufruf

        # Greens-Funktion im k-Raum vorberechnen (periodisches Gitter)
        # Φ̂(k) = -4πG ρ̂(k) / k²_gitter
        # k_gitter = 2πn/L, aber für diskrete FFT: sin-basierter Kernel
        self._green_k: "np.ndarray | None" = None   # wird bei erstem compute_accel aufgebaut  # type: ignore[annotation-unchecked]
        self._last_box_size = None

        # ── GPU-Puffer (CuPy) ───────────────────────────────────────────────
        self._use_gpu = GPU_AVAILABLE
        if self._use_gpu:
            try:
                self._rho_g       = _cp.zeros((M, M, M), dtype=_cp.float64)  # type: ignore[union-attr]
                # Kernel-Arrays werden in _build_green hochgeladen
                self._green_k_g   = None
                self._kx_g = self._ky_g = self._kz_g = None
            except Exception:
                self._use_gpu = False

        gpu_tag = '  [GPU-cuFFT]' if self._use_gpu else ''
        print(f'[PM] Gitter={M}³={M**3}  Auflösung≈{box_size/M:.1f}  G={G}{gpu_tag}')

    # ── interne Helfer ───────────────────────────────────────────────────

    def _update_box(self, pos, mass):
        """Berechnet Bounding Box aller aktiven Partikel + Padding."""
        active = mass > 0.
        if not active.any():
            return False
        ap = pos[active]
        mn = ap.min(axis=0)
        mx = ap.max(axis=0)
        span = (mx - mn).max()
        pad  = span * self.padding + 2.0 * self.eps + 10.
        self._x0 = float(mn[0] - pad)
        self._y0 = float(mn[1] - pad)
        self._z0 = float(mn[2] - pad)
        self._box_size = float(span + 2. * pad)
        self._h = self._box_size / self.M
        return True

    def _build_green(self):
        """
        Vorberechnet den diskreten Greens-Kern im Fourier-Raum.
        Verwendet den optimalen Gitter-Greens-Kern mit CIC-Korrektur:
            G_green(k) = -4πG / k²_eff · W_CIC²(k)⁻²
        Für die Praxis: Standard-1/k²-Kernel mit Softening-Regularisierung.
        """
        M    = self.M
        h    = self._h
        L    = self._box_size
        eps2 = self.eps * self.eps

        # Wellenvektor-Frequenzen (periodisch)
        freq = np.fft.fftfreq(M, d=1.0/M)   # in Einheiten 1/Zelle → [0..M/2, -M/2+1..-1]

        # k-Vektor in physikalischen Einheiten
        kx = (2 * np.pi / L) * freq[:, np.newaxis, np.newaxis]
        ky = (2 * np.pi / L) * freq[np.newaxis, :, np.newaxis]
        kz = (2 * np.pi / L) * freq[np.newaxis, np.newaxis, :]

        k2 = kx**2 + ky**2 + kz**2
        k2[0, 0, 0] = 1.0   # k=0 vermeiden (DC-Komponente = 0 setzen → keine mittlere Kraft)

        # Softened Greens-Funktion: Φ̂ = -4πG / (k² + ε²k⁴...)
        # Einfache Regularisierung: k² → k² + (2π eps/L)²
        k2_soft = k2 + (2.0 * np.pi * self.eps / L) ** 2
        self._green_k = -4.0 * np.pi * self.G / k2_soft
        self._green_k[0, 0, 0] = 0.0   # mittlere Beschleunigung = 0

        # k-Vektoren für Gradient (in-place gespeichert)
        self._kx = kx
        self._ky = ky
        self._kz = kz

        self._last_box_size = self._box_size

        # GPU-Kernel-Arrays hochladen (nach jedem _build_green-Aufruf)
        if self._use_gpu:
            try:
                Mh = self.M // 2 + 1
                self._green_k_g = _cp.asarray(self._green_k[:, :, :Mh])  # type: ignore[union-attr]
                self._kx_g      = _cp.asarray(self._kx[:, :, :Mh])       # type: ignore[union-attr]
                self._ky_g      = _cp.asarray(self._ky[:, :, :Mh])       # type: ignore[union-attr]
                self._kz_g      = _cp.asarray(self._kz[:, :, :Mh])       # type: ignore[union-attr]
            except Exception:
                self._use_gpu = False

    def compute_accel(self, pos, mass):
        """
        Berechnet PM-Beschleunigungen für alle aktiven Partikel.

        pos  : (N,3) float64
        mass : (N,)  float64
        return: (N,3) float64 Beschleunigungen
        """
        N = pos.shape[0]

        # Box anpassen (muss nach erstem Aufruf nur bei Galaxien-Bewegung aktualisiert werden)
        if not self._update_box(pos, mass):
            return np.zeros((N, 3), dtype=np.float64)

        # Greens-Funktion neu aufbauen wenn Box-Größe sich geändert hat (>1%)
        if (self._last_box_size is None
                or abs(self._box_size - self._last_box_size) / self._last_box_size > 0.01):
            self._build_green()

        # 1. CIC Masse-Assignment
        self._rho[:] = 0.
        _cic_deposit(pos, mass, self._rho, self.M, self._h,
                     self._x0, self._y0, self._z0)
        # In Dichte umrechnen: ρ = m / h³
        self._rho /= self._h ** 3

        # 2–5. FFT + Potential + Gradient + IFFT  (GPU wenn verfügbar, sonst CPU)
        M  = self.M
        Mh = M // 2 + 1

        if self._use_gpu:
            try:
                # rho in GPU-Puffer kopieren (kein neues Alloc)
                _cp.copyto(self._rho_g, _cp.asarray(self._rho))              # type: ignore[union-attr]
                rho_k = _cp.fft.rfftn(self._rho_g)                           # type: ignore[union-attr]
                phi_k = rho_k * self._green_k_g
                ax_r  = _cp.fft.irfftn((-1j * self._kx_g) * phi_k, s=(M, M, M)).get()  # type: ignore[union-attr]
                ay_r  = _cp.fft.irfftn((-1j * self._ky_g) * phi_k, s=(M, M, M)).get()  # type: ignore[union-attr]
                az_r  = _cp.fft.irfftn((-1j * self._kz_g) * phi_k, s=(M, M, M)).get()  # type: ignore[union-attr]
            except Exception:
                self._use_gpu = False   # GPU-Fehler → einmalig auf CPU zurückfallen
                assert self._green_k is not None
                rho_k = np.fft.rfftn(self._rho)
                phi_k = rho_k * self._green_k[:, :, :Mh]
                ax_r  = np.fft.irfftn((-1j * self._kx[:, :, :Mh]) * phi_k, s=(M, M, M))
                ay_r  = np.fft.irfftn((-1j * self._ky[:, :, :Mh]) * phi_k, s=(M, M, M))
                az_r  = np.fft.irfftn((-1j * self._kz[:, :, :Mh]) * phi_k, s=(M, M, M))
        else:
            # CPU-Pfad: scipy.fft (multithreaded, workers=-1) oder numpy.fft
            assert self._green_k is not None
            rho_k = _rfftn(self._rho, **_FFT_KW)  # type: ignore[call-arg]
            phi_k = rho_k * self._green_k[:, :, :Mh]
            ax_r  = _irfftn((-1j * self._kx[:, :, :Mh]) * phi_k, s=(M, M, M), **_FFT_KW)  # type: ignore[call-arg]
            ay_r  = _irfftn((-1j * self._ky[:, :, :Mh]) * phi_k, s=(M, M, M), **_FFT_KW)  # type: ignore[call-arg]
            az_r  = _irfftn((-1j * self._kz[:, :, :Mh]) * phi_k, s=(M, M, M), **_FFT_KW)  # type: ignore[call-arg]

        # 6. CIC Force-Interpolation → Partikel
        if self._accel_out is None or self._accel_out.shape[0] != N:
            self._accel_out = np.zeros((N, 3), dtype=np.float64)

        _cic_interpolate(pos, mass, self._force_grid, self._accel_out,
                         self.M, self._h, self._x0, self._y0, self._z0)
        return self._accel_out

    def step(self, pos, vel, mass, dt):
        """
        Vollständiger Leapfrog-Schritt via PM.
        Gibt (pos_new, vel_new) zurück (neue Arrays).
        """
        accel = self.compute_accel(pos, mass)
        active = mass > 0.
        vel_new  = vel.copy()
        pos_new  = pos.copy()
        vel_new[active] = vel[active] + accel[active] * dt
        pos_new[active] = pos[active] + vel_new[active] * dt
        return pos_new, vel_new
