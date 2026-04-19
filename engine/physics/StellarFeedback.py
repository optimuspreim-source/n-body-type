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
import numpy as np

_SN_MASS_MIN  = 2.5     # Minimalmasse für core-collapse SN [Simulationseinheiten]
_ETA_KINETIC  = 0.08    # Kinetischer Wirkungsgrad der SN-Energie
_WIND_FRAC    = 0.0010  # Relative Massenverlustrate pro Schritt (AGB-Winde)
_WIND_DISP    = 0.006   # Velocity-Dispersion durch stellare Winde [Simul.-Einh./Schritt]
_Z_SOLAR      = 0.02    # Solare Metallizität (Massenbruch)


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
        self.Z = rng.normal(_Z_SOLAR, 0.003, N).clip(0.002, 0.30).astype(np.float64)
        self.Z[bh_mask] = 0.

        # ── Diagnostik-Zähler ────────────────────────────────────────────
        self.total_sn    = 0    # Supernova-Ereignisse gesamt
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

        for i in sn_idx:
            self._explode(i, pos, vel, mass)
            # Remnant: Neutronenstern / stellares BH  (~12% Sternmasse, min 0.5)
            mass[i] = max(float(mass[i]) * 0.12, 0.5)
            # SN-Ejecta reichern Nachbarn mit Metallen an (O, Mg, Si, Fe)
            self._enrich_sn(i, pos, mass)
            # Remnant altert nicht mehr (sehr lange Lebensdauer)
            self.age[i]      = 0.
            self.lifetime[i] = 1e18

        self.last_sn   = len(sn_idx)
        self.total_sn += self.last_sn

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
    def _explode(self, i, pos, vel, mass):
        """
        SN-Explosion: kinetische Energie gleichmäßig auf Nachbarn im r_fb.
        Δv = √(2·η·E_SN / N_nb)  radial nach außen.
        """
        dr   = pos - pos[i][np.newaxis, :]
        r2   = (dr * dr).sum(axis=1)
        mask = (r2 < self.r_fb2) & (r2 > 1e-4) & (mass > 0.)
        nb   = np.where(mask)[0]
        if len(nb) == 0:
            return

        dv_mag = np.sqrt(2. * _ETA_KINETIC * self.E_SN / len(nb))
        r_nb   = np.sqrt(r2[nb])
        # Radiale Einheitsvektoren (SN-Zentrum → Nachbar)
        vel[nb] += (dr[nb] / r_nb[:, np.newaxis]) * dv_mag

    def _enrich_sn(self, i, pos, mass, yield_frac=0.30):
        """
        Metallizitäts-Anreicherung durch SN-Ejecta.
        ~30% der Sternmasse wird als Metall-Yield an Nachbarn verteilt.
        """
        dr   = pos - pos[i][np.newaxis, :]
        r2   = (dr * dr).sum(axis=1)
        mask = (r2 < self.r_fb2) & (r2 > 1e-4) & (mass > 0.)
        nb   = np.where(mask)[0]
        if len(nb) > 0:
            dZ = yield_frac * float(mass[i]) * 4e-5 / max(len(nb), 1)
            self.Z[nb] = np.minimum(self.Z[nb] + dZ, 0.25)

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
