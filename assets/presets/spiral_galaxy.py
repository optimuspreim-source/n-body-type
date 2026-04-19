"""
Preset: 3D Spiralgalaxie  –  schnelle Variante
═══════════════════════════════════════════════════════════════
Generiert eine einzelne, physikalisch fundierte 3D-Spiralgalaxie.

Verglichen mit generate_galaxy_disk:
  + Schneller (kein Kroupa-IMF-Sampling, keine Akk.-Scheibe)
  + Weniger Parameter
  - Keine NFW-Dunkle-Materie-Option
  - Kein Akkretionstorus

Rückgabe
--------
Liste von Dicts:  position, velocity, mass, color, softening, is_bh (für 1 BH)
Kompatibel mit GalaxySimVispyViewer (identisches Format wie galaxy_disk.py).
═══════════════════════════════════════════════════════════════
"""
import numpy as np


def generate_spiral_galaxy(
    num_stars:     int   = 5_000,
    arms:          int   = 2,
    radius:        float = 80.0,
    spread:        float = 0.25,
    z_spread:      float = 1.5,
    bh_mass:       float = 12_000.0,
    softening:     float = 1.2,
    softening_bh:  float = 6.0,
    arm_tightness: float = 1.6,
    center         = (0., 0., 0.),
    bulk_velocity  = (0., 0., 0.),
    color          = (1.0, 0.85, 0.30),
    G:             float = 1.0,
    seed           = None,
) -> list[dict]:
    """
    Erzeugt eine Spiralgalaxie als Partikelliste.

    Parameters
    ----------
    num_stars     : Anzahl der Sterne (ohne zentrales BH)
    arms          : Anzahl der Spiralarme (1–6)
    radius        : Außenradius der Scheibe [lu]
    spread        : Streufaktor um die Arme (größer = diffuser)
    z_spread      : Vertikale Dispersion [σ in lu]
    bh_mass       : Masse des zentralen Schwarzen Lochs
    arm_tightness : Stärke der logarithmischen Spiralkrümmung
    center        : Mittelpunkt der Galaxie [lu]
    bulk_velocity : Schwerpunktsgeschwindigkeit [lu/tu]
    color         : RGB-Sternfarbe (Float 0–1)
    G             : Gravitationskonstante
    """
    rng    = np.random.default_rng(seed)
    center = np.asarray(center, dtype=np.float64)
    bv     = np.asarray(bulk_velocity, dtype=np.float64)

    N = num_stars
    r_inner = radius * 0.03

    # ── Radialverteilung: exponentiell abfallend ───────────────────────────
    r_sc = radius / 3.5
    u    = rng.uniform(0., 1., N)
    r    = -r_sc * np.log(1. - u * (1. - np.exp(-radius / r_sc)))
    r    = np.clip(r, r_inner, radius)

    # ── Spiralwinkel ───────────────────────────────────────────────────
    arm_idx   = rng.integers(0, arms, N)
    theta_arm = arm_idx * (2. * np.pi / arms)
    spiral    = arm_tightness * np.log1p(r / r_sc)
    scatter_w = rng.normal(0., spread * (1. + 0.5 * r / radius), N)
    theta     = theta_arm + spiral + scatter_w

    # ── Vertikale Streuung: sech²-Profil dünner am Rand ─────────────────
    z_scale = radius * 0.18
    z_sigma = z_spread / np.cosh(r / z_scale) ** 2 + z_spread * 0.03
    z_sign  = np.where(rng.random(N) > 0.5, 1., -1.)
    z       = z_sign * np.abs(rng.normal(0., z_sigma, N))

    pos_local = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    pos       = pos_local + center[np.newaxis, :]

    # ── Rotationskurve: Keplerian + eingeschlossene Sternmasse ───────────
    avg_m    = 1.5          # typische Sternmasse [G=1-Einheiten]
    M_enc    = bh_mass + N * avg_m * (r / radius) ** 1.5
    v_rot    = np.sqrt(G * M_enc / (r + 1e-6))

    # Tangentiale Einheitsrichtung: (-sinθ, cosθ, 0)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    vel   = np.zeros((N, 3), dtype=np.float64)
    vel[:, 0] = -sin_t * v_rot
    vel[:, 1] =  cos_t * v_rot
    vel[:, 2] =  rng.normal(0., 0.025 * v_rot, N)
    vel += bv[np.newaxis, :]

    # ── Massen: vereinfacht, keine IMF ────────────────────────────────
    masses = rng.uniform(0.5, 3.0, N)

    # ── Partikel-Dicts (kompatibel mit galaxy_disk-Format) ─────────────
    stars: list[dict] = []

    # Zentrales Schwarzes Loch
    stars.append({
        'position':  tuple(center.tolist()),
        'velocity':  tuple(bv.tolist()),
        'mass':      float(bh_mass),
        'color':     (1.0, 1.0, 1.0),
        'softening': softening_bh,
        'is_bh':     True,
    })

    # Sterne
    for i in range(N):
        stars.append({
            'position':  (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])),
            'velocity':  (float(vel[i, 0]), float(vel[i, 1]), float(vel[i, 2])),
            'mass':      float(masses[i]),
            'color':     color,
            'softening': softening,
            'is_bh':     False,
        })

    return stars


# ── Preset-Fabrik für SimulationLauncher ───────────────────────────────────

def build_spiral_galaxies_preset(
    n_galaxies:  int   = 2,
    n_stars:     int   = 20_000,
    disk_radius: float = 80.0,
    sep:         float = 750.0,
    G:           float = 1.0,
    bh_mass:     float = 12_000.0,
    **_kw,
) -> tuple[list[list[dict]], None]:
    """
    Erzeugt N Spiralgalaxien für den Launcher.

    Rückgabe: (galaxies, None)  – kein DM-Halo-Config (nicht unterstützt).
    Format identisch mit generate_n_galaxy_disks.
    """
    n_per = max(100, n_stars // max(1, n_galaxies))
    galaxies: list[list[dict]] = []
    colors = [
        (1.0, 0.85, 0.30),  # gelb
        (0.4, 0.7,  1.0),   # blau
        (1.0, 0.5,  0.3),   # orange
        (0.6, 1.0,  0.5),   # grün
        (0.9, 0.4,  0.9),   # violett
    ]
    for k in range(n_galaxies):
        angle = k * (2. * np.pi / n_galaxies)
        cx = sep * np.cos(angle)
        cy = sep * np.sin(angle)
        # Orbit-Geschwindigkeit damit alle Galaxien sich aufeinander zubewegen
        v_mag  = 0.35 * np.sqrt(G * bh_mass / max(sep, 1.))
        # Senkrecht zur Verbindungslinie = tangential
        vx = -np.sin(angle) * v_mag
        vy =  np.cos(angle) * v_mag
        gal = generate_spiral_galaxy(
            num_stars    = n_per,
            radius       = disk_radius,
            center       = (cx, cy, 0.),
            bulk_velocity= (vx, vy, 0.),
            color        = colors[k % len(colors)],
            bh_mass      = bh_mass,
            G            = G,
        )
        galaxies.append(gal)
    return galaxies, None


if __name__ == '__main__':
    stars = generate_spiral_galaxy()
    print(f'Generated {len(stars)} particles (1 BH + {len(stars)-1} stars).')
