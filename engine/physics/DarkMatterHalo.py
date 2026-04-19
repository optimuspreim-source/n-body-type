"""
DarkMatterHalo.py  –  NFW-Dunkle-Materie-Halo als statisches Hintergrundpotential
════════════════════════════════════════════════════════════════════════════════
Physik
──────
  Navarro-Frenk-White (NFW) Dichte-Profil:
      ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]

  Eingeschlossene Masse:
      M_enc(r) = 4π ρ_s r_s³ · [ln(1 + r/r_s) − r/r_s / (1 + r/r_s)]

  Beschleunigung (radial zum Halo-Zentrum):
      a⃗(r) = −G·M_enc(r) / r²  ·  r̂

  Normierungs-Dichte aus Virialmasse und Konzentrationsparameter c = r_vir/r_s:
      ρ_s = M_vir / [4π r_s³ · f(c)]    f(c) = ln(1+c) − c/(1+c)

  Effekt auf Rotationskurve:
      Ohne DM:  v_circ(r) ∝ r^{−1/2}  (Keplerian außerhalb der Scheibe)
      Mit DM:   v_circ(r) ≈ const.     (flache Rotationskurve – beobachtet!)

Numerik
───────
  Vollständig vektorisiert (NumPy), O(N) pro Halo, O(K·N) für K Halos.
  Halo-Zentren folgen dem zugehörigen SMBH (dynamisch aktualisierbar).
════════════════════════════════════════════════════════════════════════════════
"""
import numpy as np


class NFWHalo:
    """NFW-Dunkle-Materie-Halo mit analytisch berechnetem Gravitationspotential."""

    def __init__(self, center, M_vir, r_s, c=12.0, G=1.0):
        """
        center : array-like (3,)  – Halo-Zentrum (dynamisch per update_center)
        M_vir  : float            – Viriale Gesamtmasse des Halos
        r_s    : float            – NFW-Skalenradius  (r_s = r_vir / c)
        c      : float            – Konzentrationsparameter  (typ. 8–20)
        G      : float            – Gravitationskonstante
        """
        self.center = np.array(center, dtype=np.float64)
        self.M_vir  = float(M_vir)
        self.r_s    = float(r_s)
        self.G      = float(G)
        # 4π ρ_s r_s³ = M_vir / f(c)
        f_c = np.log(1. + c) - c / (1. + c)
        self._norm = M_vir / f_c        # = 4π ρ_s r_s³

    def update_center(self, new_center):
        """Verschiebt das Halo-Zentrum (folgt dem SMBH bei Kollisionen)."""
        self.center[:] = new_center

    def acceleration(self, pos):
        """
        Vektorisierte NFW-Beschleunigung aller Partikel.

        pos    : (N, 3) float64
        return : (N, 3) Beschleunigung in Simulationseinheiten
        """
        dr  = pos - self.center[np.newaxis, :]        # (N, 3)
        r   = np.linalg.norm(dr, axis=1)              # (N,)
        r   = np.maximum(r, 0.5)                       # Regularisierung

        x     = r / self.r_s
        M_enc = self._norm * (np.log(1. + x) - x / (1. + x))  # (N,)

        a_mag = self.G * M_enc / (r * r)               # (N,)
        return -(a_mag / r)[:, np.newaxis] * dr         # (N, 3)

    def circular_velocity(self, r_arr):
        """
        Kreisbahngeschwindigkeit im NFW-Potential.
        r_arr : 1D-Array  [Simulationseinheiten]
        return: v_circ(r)
        """
        r   = np.maximum(np.asarray(r_arr, dtype=np.float64), 0.5)
        x   = r / self.r_s
        M_enc = self._norm * (np.log(1. + x) - x / (1. + x))
        return np.sqrt(self.G * M_enc / r)


class DarkMatterSystem:
    """
    Verwaltet ein Ensemble von NFW-Halos und summiert ihre Beschleunigungen.
    Typisch: ein Halo pro Galaxie, verknüpft mit dem jeweiligen SMBH.
    """

    def __init__(self, halos):
        self.halos = list(halos)

    def acceleration(self, pos):
        """Gesamte DM-Beschleunigung aller Halos (O(K·N))."""
        acc = np.zeros_like(pos)
        for h in self.halos:
            acc += h.acceleration(pos)
        return acc

    def update_centers(self, centers):
        """
        Aktualisiert Halo-Zentren aus einer Liste von Positionen.
        Fehlende Einträge (nach BH-Merger) werden ignoriert.
        """
        for h, c in zip(self.halos, centers):
            h.update_center(c)
