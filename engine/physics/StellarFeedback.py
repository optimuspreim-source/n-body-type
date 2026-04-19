"""
StellarFeedback.py  –  Subgrid-Modell: Supernovae, Stellarwinde, Metallizitäts-Tracking
════════════════════════════════════════════════════════════════════════════════
Physik
──────
  Stellare Lebenszeit (Hauptreihe):
      τ(m) = τ₀ · (m/m₀)^{−2.5}  [Schritt-Einheiten]
      Massereiche Sterne (m > 8 M_sun) leben kurz → frühe SN II.
      Leichte Sterne (m < 8 M_sun) durchlaufen AGB-Phase → stellare Winde.

  Supernovae Typ II  (core-collapse, m ≥ m_SN_min):
      E_SN ≈ 10^51 erg  →  kinetischer Anteil η_kin ≈ 8%
      Δv_nb = √(2·η·E_SN / N_nb)  radial nach außen über alle Nachbarn im r_fb
      Remnant: Neutronenstern oder stellares BH  (≈12% der Sternmasse)

  Stellare Winde – AGB-Phase  (späte Entwicklung, m < m_SN_min):
      Massenverlustrate Ṁ ~ 10^{−8}–10^{−4} M_sun/yr
      → Modelliert als gentle Velocity-Dispersion Perturbation

  Kroupa-IMF-basierte Massen  (wird in galaxy_disk.py gesampelt):
      ξ(m) ∝ m^{−1.3}  für  0.08 < m < 0.5  [M_sun]
      ξ(m) ∝ m^{−2.3}  für  0.5  ≤ m < 100  [M_sun]

  Metallizitäts-Tracking:
      Jeder Stern trägt Z (Massenbruch an Metallen, Z_solar = 0.02).
      SN-Ejecta: ~30% der Sternmasse in schweren Elementen (O, Mg, Si, Fe).
      AGB-Winde:  ~0.01% Z-Zunahme pro Schritt (C, N).
      Z = 0 → metallarm (Population III), Z → 0.25 → super-solar, metallreich.

Numerik
───────
  Lebenszeit: τ ∝ m^{−2.5}, ±20% Streuung (natürliche Variation).
  Altersverteilung: initial uniform in [0, 0.4·τ] → realistische Sternschiebe.
  SN-Schleife: O(k·N) für k SN pro Frame (k ≈ 1–5, N ≈ 50k → vernachlässigbar).
  Alle Array-Operationen vektorisiert (NumPy).
════════════════════════════════════════════════════════════════════════════════
"""
import math
import numpy as np
from numba import njit  # type: ignore[import-untyped]

_SN_MASS_MIN  = 2.5     # Minimalmasse für core-collapse SN [Simulationseinheiten]
_ETA_KINETIC  = 0.08    # Kinetischer Wirkungsgrad der SN-Energie
_WIND_FRAC    = 0.0010  # Relative Massenverlustrate pro Schritt (AGB-Winde)
_WIND_DISP    = 0.006   # Velocity-Dispersion durch stellare Winde [Simul.-Einh./Schritt]
_Z_SOLAR      = 0.02    # Solare Metallizität (Massenbruch)


# ─ Numba-JIT Batch-SN-Kernel ───────────────────────────────────────────────
# Verarbeitet alle SN-Ereignisse eines Schritts in zwei O(N)-Pässen pro SN:
# 1. Nachbarn zählen (kein Alloc), 2. Geschwindigkeit + Metallizität updaten.
# Im Vergleich zur NumPy-Version: kein temp-Array-Alloc, kein Python-Loop-Overhead.

@njit(cache=True, fastmath=True)
def _sn_kernel(sn_pos, sn_mass, pos, vel, Z, mass, r_fb2, eta_kin, E_SN, yield_frac):
    """
    Führt ALLE SN-Explosionen + Metallizitäts-Anreicherung des aktuellen Schritts durch.
    sn_pos  : (K, 3) float64  – Positionen der K Supernovae
    sn_mass : (K,)   float64  – Massen der K Supernovae (vor Remnant-Cut)
    Modifiziert vel und Z **in-place** (Massen der SN-Sterne werden außerhalb behandelt).
    """
    N   = pos.shape[0]
    K   = sn_pos.shape[0]
    for s in range(K):
        px = sn_pos[s, 0]; py = sn_pos[s, 1]; pz = sn_pos[s, 2]
        m_s = sn_mass[s]   # korrekte Masse des s-ten SN-Sterns

        # Pass 1: Nachbarn zählen
        n_nb = 0
        for j in range(N):
            if mass[j] <= 0.:
                continue
            dx = pos[j, 0] - px
            dy = pos[j, 1] - py
            dz = pos[j, 2] - pz
            r2 = dx*dx + dy*dy + dz*dz
            if 1e-4 < r2 < r_fb2:
                n_nb += 1

        if n_nb == 0:
            continue

        dv_mag = math.sqrt(2.0 * eta_kin * E_SN / n_nb)
        dZ_each = yield_frac * m_s * 4e-5 / n_nb

        # Pass 2: Velocity-Kick + Metallizitäts-Anreicherung
        for j in range(N):
            if mass[j] <= 0.:
                continue
            dx = pos[j, 0] - px
            dy = pos[j, 1] - py
            dz = pos[j, 2] - pz
            r2 = dx*dx + dy*dy + dz*dz
            if 1e-4 < r2 < r_fb2:
                r = math.sqrt(r2)
                vel[j, 0] += (dx / r) * dv_mag
                vel[j, 1] += (dy / r) * dv_mag
                vel[j, 2] += (dz / r) * dv_mag
                new_Z = Z[j] + dZ_each
                Z[j] = new_Z if new_Z < 0.25 else 0.25


class StellarFeedback:
    """
    Verwaltet stellare Altersstruktur, SN-Explosionen, AGB-Winde und
    Metallizitäts-Anreicherung für alle Partikel des Systems.
    """

    def __init__(self, N, mass, bh_mask,
                 r_fb      = 18.0,    # Feedback-Radius [Simulationseinheiten]
                 E_SN      = 3200.,   # SN-Energie [Simulationseinheiten]
                 tau_ref   = 2000,    # Referenz-Lebensdauer in Schritten (m=1.0)
                 seed      = None):
        """
        N        : Gesamtanzahl der Partikel
        mass     : (N,) float64  – Anfangsmassen (werden in-place modifiziert)
        bh_mask  : (N,) bool     – True für SMBHs (kein Feedback)
        r_fb     : Radius für SN-Energiedissipation
        E_SN     : Gesamtenergie einer Supernova (Simulationseinheiten)
        tau_ref  : Lebenszeit eines Sterns der Masse 1.0 in Simulationsschritten
        """
        rng = np.random.default_rng(seed)
        self.N       = N
        self.r_fb    = float(r_fb)
        self.r_fb2   = r_fb * r_fb
        self.E_SN    = float(E_SN)
        self.bh_mask = bh_mask.copy()

        # ── Stellare Lebenszeiten: τ ∝ m^{−2.5} ─────────────────────────
        m_safe    = np.maximum(mass, 0.01)
        raw_life  = tau_ref * (m_safe ** -2.5)
        raw_life *= rng.uniform(0.80, 1.20, N)   # natürliche Streuung ±20%
        raw_life[bh_mask] = 1e18                  # SMBHs sterben nicht
        self.lifetime = raw_life.astype(np.float64)

        # ── Initiales Alter: uniform in [0, 40% der Lebenszeit] ─────────
        # Repräsentiert eine bestehende Galaxienscheibe mit Sternpopulationen
        # aller Altersklassen (Hauptreihe bis kurz vor Ende).
        self.age           = rng.uniform(0., 0.40, N) * raw_life
        self.age[bh_mask]  = 0.

        # ── Metallizität: normalverteilt um Z_solar, zufällige Variation ─
        # Leicht erhöht im Zentrum (vorangegangene SN-Episoden).
        self.Z = rng.normal(_Z_SOLAR, 0.003, N).clip(0.002, 0.30).astype(np.float64)  # type: ignore[union-attr]
        self.Z[bh_mask] = 0.

        # ── Diagnostik-Zähler ────────────────────────────────────────────
        self.total_sn    = 0    # Supernova-Ereignisse gesamt
        self._total_sn   = 0    # interner Alias (Konsistenz mit Viewer)
        self.last_sn     = 0    # SNe im letzten Schritt

    # ─────────────────────────────────────────────────────────────────────
    def step(self, pos, vel, mass):
        """
        Führt einen Feedback-Schritt aus.
        Modifiziert vel und mass **in-place**.
        Gibt (n_sn, n_wind) zurück.
        """
        self.age += 1.
        self.last_sn = 0
        n_wind       = 0

        active_star = (~self.bh_mask) & (mass > 0.)

        # ── Supernovae: massereiche Sterne, Lebenszeit überschritten ─────
        sn_mask = active_star & (self.age >= self.lifetime) & (mass >= _SN_MASS_MIN)
        sn_idx  = np.where(sn_mask)[0]

        if len(sn_idx) > 0:
            # Batch-Kernel: alle SNe dieses Schritts in einer Numba-Routine
            sn_pos  = pos[sn_idx].copy()
            sn_mass = mass[sn_idx].copy()   # Massen VOR Remnant-Cut
            _sn_kernel(sn_pos, sn_mass, pos, vel, self.Z, mass,
                       self.r_fb2, _ETA_KINETIC, self.E_SN, 0.30)
            # Remnant-Masse + Alters-Reset (Python-Schleife über K ≪ N)
            for i in sn_idx:
                mass[i] = max(float(mass[i]) * 0.12, 0.5)
                self.age[i]      = 0.
                self.lifetime[i] = 1e18
            self._total_sn += len(sn_idx)
            self.total_sn  += len(sn_idx)
        self.last_sn = len(sn_idx)

        # ── AGB-Stellarwinde: ältere leichte Sterne (späte Entwicklung) ──
        agb_mask = (active_star
                    & (self.age > 0.72 * self.lifetime)
                    & (mass < _SN_MASS_MIN))
        agb_idx  = np.where(agb_mask)[0]

        if len(agb_idx) > 0:
            # Thermischer Wind: Velocity-Dispersion leicht anheben
            noise         = np.random.normal(0., _WIND_DISP, (len(agb_idx), 3))
            vel[agb_idx] += noise
            # Massenverlust durch Hüllenabwurf (Planetary Nebula)
            mass[agb_idx] = np.maximum(mass[agb_idx] * (1. - _WIND_FRAC), 0.08)
            # AGB produziert C, N – leichte Metallizitätszunahme
            self.Z[agb_idx] = np.minimum(self.Z[agb_idx] + 1.5e-5, 0.25)
            n_wind = len(agb_idx)

        return self.last_sn, n_wind

    # ─────────────────────────────────────────────────────────────────────
    def metallicity_colors(self):
        """
        Berechnet RGBA-Farben aus Metallizitäts-Array Z.
        Farbskala:  blau (Z≈0) → gelb (Z=solar) → weiß (Z≫solar)
        Gibt (N, 4) float32 zurück.
        """
        t = np.clip(self.Z / (3. * _Z_SOLAR), 0., 1.)   # 0 → arm, 1 → dreifach solar
        r = np.clip(0.1 + 1.5 * t,         0., 1.)
        g = np.clip(0.1 + 1.4 * t - t**2,  0., 1.)
        b = np.clip(0.9 - 2.0 * t,         0., 1.)
        a = np.ones(self.N, dtype=np.float32) * 0.85
        return np.stack([r, g, b, a], axis=1).astype(np.float32)
