"""
Preset: Zwei kollidierende Galaxienscheiben
═══════════════════════════════════════════════════════════════
Nutzt generate_galaxy_disk für physikalisch fundierte Scheiben (mit NFW-DM-Halos)
statt der veralteten generate_spiral_galaxy.

Rückgabe: (galaxies, dm_cfgs) – identisches Format wie generate_n_galaxy_disks.
Kann direkt im SimulationLauncher aufgerufen werden.
═══════════════════════════════════════════════════════════════
"""
import numpy as np
from assets.presets.galaxy_disk import generate_galaxy_disk

# ── NFW-Standard-Halos (für Rotationskurven-Unterstützung) ─────────────────
_DM_M_VIR = 520_000.0
_DM_R_S   = 180.0
_DM_C     = 12.0


def generate_colliding_galaxies(
    stars_per_galaxy: int   = 25_000,
    disk_radius:      float = 80.0,
    sep:              float = 550.0,
    approach:         float = 0.50,    # Annäherungsgeschwindigkeit
    transverse:       float = 0.12,    # Off-Center-Anteil
    bh_mass:          float = 14_000.0,
    dm_enabled:       bool  = True,
    dm_M_vir:         float = _DM_M_VIR,
    dm_r_s:           float = _DM_R_S,
    dm_c:             float = _DM_C,
    G:                float = 1.0,
    eps_star:         float = 1.2,
    eps_bh:           float = 6.0,
    seed:             int | None = None,
) -> tuple[list[list[dict]], list[dict] | None]:
    """
    Zwei Galaxienscheiben auf Off-Center-Kollisionskurs.

    Galaxie 1: links  (+x-Bewegung), prograde Rotation
    Galaxie 2: rechts (−x-Bewegung), retrograde Rotation  → maximale Gezeitenkräfte

    Rückgabe
    --------
    (galaxies, dm_cfgs)
      galaxies : [[dict, ...], [dict, ...]]  – zwei Partikellisten
      dm_cfgs  : list[dict] | None           – NFW-Konfigurationen oder None
    """
    rng_g1 = None if seed is None else seed
    rng_g2 = None if seed is None else seed + 1

    dm_kw = dict(dm_M_vir=dm_M_vir, dm_r_s=dm_r_s, dm_c=dm_c) if dm_enabled else {}

    # ── Galaxie 1: links, prograde ───────────────────────────────────
    g1 = generate_galaxy_disk(
        n_stars       = stars_per_galaxy,
        disk_radius   = disk_radius,
        center        = (-sep, +sep * 0.08, 0.),
        bulk_velocity = (+approach, +transverse, 0.),
        bh_mass       = bh_mass,
        disk_color    = (1.0, 0.90, 0.35),    # gelb
        G             = G,
        seed          = rng_g1,
        **dm_kw,
    )

    # ── Galaxie 2: rechts, retrograde (Rotationsrichtung umgekehrt) ──────
    # retrograde_lower=True dreht die untere Hemisphäre um (standard). Für
    # eine vollständig retrograde Scheibe spiegeln wir v_tangential:
    g2_raw = generate_galaxy_disk(
        n_stars       = stars_per_galaxy,
        disk_radius   = disk_radius * 0.9,
        center        = (+sep, -sep * 0.08, 0.),
        bulk_velocity = (-approach, -transverse, 0.),
        bh_mass       = bh_mass * 0.8,
        disk_color    = (0.35, 0.75, 1.0),    # blau
        G             = G,
        seed          = rng_g2,
        **dm_kw,
    )
    # Retrograde: v_xy umkehren (Bulk-Velocity beibehalten)
    bv2 = np.array([-approach, -transverse, 0.])
    g2: list[dict] = []
    for s in g2_raw:
        if s.get('is_bh', False):
            g2.append(s)
            continue
        v = np.asarray(s['velocity'])
        v_tang = v - bv2          # tangential-Anteil relativ zum Bulk
        v_tang[:2] *= -1.          # tangentiale Richtung in xy umkehren
        s['velocity'] = tuple((v_tang + bv2).tolist())
        g2.append(s)

    # ── DM-Konfigurationen ─────────────────────────────────────────────
    dm_cfgs: list[dict] | None = None
    if dm_enabled:
        dm_cfgs = [
            dict(center=[-sep, sep * 0.08, 0.], M_vir=dm_M_vir, r_s=dm_r_s, c=dm_c),
            dict(center=[+sep, -sep * 0.08, 0.], M_vir=dm_M_vir * 0.8, r_s=dm_r_s, c=dm_c),
        ]

    # Softening setzen
    for gal in [g1, g2]:
        for s in gal:
            s['softening'] = eps_bh if s.get('is_bh', False) else eps_star

    return [g1, g2], dm_cfgs


if __name__ == '__main__':
    galaxies, dm_cfgs = generate_colliding_galaxies(stars_per_galaxy=5_000)
    print(f'Galaxien: {[len(g) for g in galaxies]}')
    print(f'DM-Halos: {dm_cfgs is not None}')
