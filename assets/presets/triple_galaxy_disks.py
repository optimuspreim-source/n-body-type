"""
triple_galaxy_disks.py – Drei 3D-Galaxienscheiben mit großem Abstand, Inklination, SMBH, Softening
Enthält NFW-DM-Halo-Konfiguration für realistische flache Rotationskurven.
"""
import numpy as np
from assets.presets.galaxy_disk import generate_galaxy_disk

# ── NFW-Dunkle-Materie-Halo-Parameter (geteilt über alle drei Galaxien) ──────
# Kompakt genug, dass r_vir < sep → minimale inter-galaktische DM-Kräfte
# r_vir = c × r_s = 10 × 55 = 550  <  sep=750  ✓
_DM_M_VIR = 50_000.    # Viriale DM-Masse pro Galaxie [Simulationseinheiten]
_DM_R_S   = 55.        # NFW-Skalenradius  [Simulationseinheiten]
_DM_C     = 10.        # Konzentrationsparameter c = r_vir / r_s

# ── Baryonische Schwerpunkt-Masse pro Galaxie (für Einfall-Geschwindigkeit) ──
_M_BARY   = 62_000.    # BH (14k) + mittlere Scheibenmasse (~48k)


def _compute_infall_velocities(centers, bary_masses, G=1.0,
                               v_factor=0.30, v_tang_frac=0.18):
    """
    Einfall-Geschwindigkeiten: jede Galaxie fliegt auf den Gruppen-Schwerpunkt zu.

      v_radial = −v_factor × v_ff   (einwärts)
      v_tang   = v_tang_frac × |v_radial|  (tangential, erzeugt Tidalschwänze)

    v_factor  : Anteil der freien Fallgeschwindigkeit (0 = Ruhe, 1 = Flucht)
    CoM-Impuls wird auf 0 normiert.
    """
    pos  = np.array(centers,      dtype=np.float64)
    mass = np.array(bary_masses,  dtype=np.float64)
    Mt   = mass.sum()
    com  = (pos * mass[:, np.newaxis]).sum(axis=0) / Mt
    vel  = np.zeros((len(pos), 3), dtype=np.float64)

    for i in range(len(pos)):
        rv  = pos[i] - com
        r   = np.linalg.norm(rv)
        if r < 1e-6:
            continue
        Mo  = Mt - mass[i]
        # Freie-Fall-Geschwindigkeit (halbe Escape-Velocity)
        v_ff = np.sqrt(G * Mo / r)
        rh   = rv / r
        # Einwärtiger Radial-Anteil
        v_rad = -rh * v_ff * v_factor
        # Tangential-Anteil (immer in derselben Rotationsrichtung → koordinierter Flyby)
        up = np.array([0., 0., 1.]) if abs(rh[2]) < 0.9 else np.array([1., 0., 0.])
        t  = np.cross(rh, up);  t /= np.linalg.norm(t)
        vel[i] = v_rad + t * (v_ff * v_factor * v_tang_frac)

    # CoM-Impuls auf null bringen
    p_total = (vel * mass[:, np.newaxis]).sum(axis=0)
    vel -= p_total[np.newaxis, :] / Mt
    return vel


def generate_triple_galaxy_disks(n_stars=16000, disk_radius=80.0, sep=750.0):
    """
    Drei Galaxien mit großem Abstand, unterschiedlichen Inklinationen und SMBHs.
    Softening: leicht für Sterne, stark für SMBH (per Attribut, falls unterstützt)

    Gibt zurück: (galaxies, dm_halo_configs)
      galaxies        : Liste von 3 Galaxien-Partikellisten
      dm_halo_configs : Liste von 3 Dicts mit NFW-Parametern pro Galaxie
    """
    # Zentren und Inklinationen
    centers = [
        (-sep, 0, 0),
        (sep, 0, 0),
        (0, sep * 0.8, 0)
    ]
    normals = [
        (0, 0, 1),
        (0.2, 0.95, 0.22),
        (-0.7, 0.1, 0.7)
    ]
    colors = [
        (1.0, 0.90, 0.35),   # gold
        (0.3, 0.95, 1.0),    # cyan
        (1.0, 0.35, 0.7)     # magenta
    ]

    # ── Einfall-Anfangsgeschwindigkeiten ─────────────────────────────────
    bulk_vels = _compute_infall_velocities(
        centers, [_M_BARY] * 3, G=1.0, v_factor=0.30, v_tang_frac=0.18)
    print(f'[Preset] Einfall-v: {[f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})" for v in bulk_vels]}')

    galaxies = []
    for i in range(3):
        gal = generate_galaxy_disk(
            n_stars=n_stars,
            bh_mass=14000.0,
            disk_radius=disk_radius,
            thickness_ratio=0.012,
            center=centers[i],
            bulk_velocity=tuple(bulk_vels[i]),
            normal=normals[i],
            arms=2+i,
            arm_tightness=0.38+0.08*i,
            v_factor=1.16+0.03*i,
            v_dispersion=0.045,
            r_inner=4.0,
            mass_center=6.0,
            mass_edge=0.7,
            size_center=3.8,
            size_edge=0.9,
            disk_color=colors[i],
            n_accretion=220,
            retrograde_lower=(i==1),
            G=1.0,
            seed=42+i*100,
            # NFW-DM-Halo: korrigiert Kreisbahngeschwindigkeiten für flache Rotationskurve
            dm_M_vir=_DM_M_VIR,
            dm_r_s=_DM_R_S,
            dm_c=_DM_C,
        )
        # Softening-Attribut für SMBH und Sterne (falls unterstützt)
        for s in gal:
            if s.get('is_bh', False):
                s['softening'] = 6.0  # Stark für SMBH
            else:
                s['softening'] = 1.2  # Leicht für Sterne
        galaxies.append(gal)

    # ── DM-Halo-Konfiguration für den Viewer ─────────────────────────────
    dm_halo_configs = [
        {'M_vir': _DM_M_VIR, 'r_s': _DM_R_S, 'c': _DM_C, 'center': centers[i]}
        for i in range(3)
    ]
    return galaxies, dm_halo_configs

if __name__ == "__main__":
    galaxies, dm_cfgs = generate_triple_galaxy_disks()
    print(f"Galaxies: {[len(g) for g in galaxies]}")
    print(f"DM halos: {len(dm_cfgs)}  M_vir={dm_cfgs[0]['M_vir']:.0f}")
