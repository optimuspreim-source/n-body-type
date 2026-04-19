"""
n_galaxy_disks.py - Dynamisch konfigurierbares N-Galaxien-Preset (N = 2 ... 20)
================================================================================
Erzeugt N physikalisch fundierte 3D-Galaxienscheiben.

Die Startpositionen und -geschwindigkeiten werden ueber einen automatisch
gewaehlten Layout-Modus generiert, der auf der Galaxienzahl basiert:

  N=2         Flyby / Merger
  N=3         Hierarchisches Triplett
  N=4-6       Kompakte Gruppe (Hickson Compact Group)
  N=7-12      Lockere Gruppe
  N=13-20     Galaxienhaufen (NFW-Radialverteilung)
"""
import colorsys
import math
import numpy as np
from assets.presets.galaxy_disk import generate_galaxy_disk

# -- NFW-Standardwerte ---------------------------------------------------------
_DM_M_VIR_DEFAULT = 520_000.0
_DM_R_S_DEFAULT   = 180.0
_DM_C_DEFAULT     = 12.0

# -- Layout-Modi ---------------------------------------------------------------
_MODE_FLYBY   = 'Flyby / Merger'
_MODE_TRIPLE  = 'Hierarchisches Triplett'
_MODE_COMPACT = 'Kompakte Gruppe (Hickson)'
_MODE_GROUP   = 'Lockere Gruppe'
_MODE_CLUSTER = 'Galaxienhaufen (NFW)'


# ------------------------------------------------------------------------------
# Layout-Selektion & 3D-Positionen
# ------------------------------------------------------------------------------

def _layout_mode(n: int) -> str:
    if n == 2:  return _MODE_FLYBY
    if n == 3:  return _MODE_TRIPLE
    if n <= 6:  return _MODE_COMPACT
    if n <= 12: return _MODE_GROUP
    return _MODE_CLUSTER


def _layout_positions(n, sep, rng):
    """Physikalisch motivierte 3D-Startpositionen fuer N Galaxien.
    Rueckgabe: (pos (N,3) float64, mode_name)
    """
    mode = _layout_mode(n)

    if mode == _MODE_FLYBY:
        b = sep * 0.14
        pos = np.array([
            [-sep, +b * 0.5, 0.],
            [+sep, -b * 0.5, 0.],
        ], dtype=np.float64)

    elif mode == _MODE_TRIPLE:
        inner = sep * 0.32
        outer = sep * 1.05
        z_off = sep * 0.12
        pos = np.array([
            [-inner * 0.5,  0.,    0.   ],
            [+inner * 0.5,  0.,    0.   ],
            [0.,            outer, z_off],
        ], dtype=np.float64)

    elif mode == _MODE_COMPACT:
        r = sep * 0.50
        if n == 4:
            s = r / math.sqrt(3.0)
            pos = np.array([
                [ s,  s,  s],
                [ s, -s, -s],
                [-s,  s, -s],
                [-s, -s,  s],
            ], dtype=np.float64)
        elif n == 5:
            ang  = 2.0 * math.pi / 3.0
            r_eq = r * 0.88
            h    = r * 0.70
            pos  = np.array([
                [r_eq,                          0.,  0.],
                [r_eq * math.cos(ang),   r_eq * math.sin(ang),   0.],
                [r_eq * math.cos(2*ang), r_eq * math.sin(2*ang), 0.],
                [0., 0.,  h],
                [0., 0., -h],
            ], dtype=np.float64)
        else:  # n == 6
            pos = np.array([
                [ r,  0.,  0.], [-r,  0.,  0.],
                [ 0.,  r,  0.], [ 0., -r,  0.],
                [ 0.,  0.,  r], [ 0.,  0., -r],
            ], dtype=np.float64)
        pos += rng.normal(0., r * 0.08, pos.shape)

    elif mode == _MODE_GROUP:
        pos      = np.zeros((n, 3), dtype=np.float64)
        n_sub    = 3 if n >= 10 else 2
        sub_size = n // n_sub
        r_inner  = sep * 0.42
        r_outer  = sep * 1.08
        idx      = 0
        for k in range(n_sub):
            az  = 2.0 * math.pi * k / n_sub
            el  = (k - n_sub / 2.0 + 0.5) * 0.28
            sc  = np.array([
                r_inner * math.cos(az) * math.cos(el),
                r_inner * math.sin(az) * math.cos(el),
                r_inner * math.sin(el),
            ])
            n_k = sub_size + (1 if k < n - n_sub * sub_size else 0)
            for _ in range(n_k):
                pos[idx] = sc + rng.normal(0., sep * 0.065, 3)
                idx += 1
        while idx < n:
            cos_t = rng.uniform(-1.0, 1.0)
            phi   = rng.uniform(0.0, 2.0 * math.pi)
            r_i   = r_outer * rng.uniform(0.82, 1.22)
            sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
            pos[idx] = r_i * np.array([
                sin_t * math.cos(phi),
                sin_t * math.sin(phi),
                cos_t,
            ])
            idx += 1

    else:  # _MODE_CLUSTER
        pos    = np.zeros((n, 3), dtype=np.float64)
        r_core = sep * 0.18
        r_s    = sep * 0.55
        r_max  = sep * 1.75
        for i in range(n):
            r = r_core
            for _ in range(50_000):
                r_t = rng.uniform(r_core, r_max)
                p   = r_t / (1.0 + r_t / r_s) ** 2
                if rng.uniform(0.0, r_max / (4.0 * r_s)) < p:
                    r = r_t
                    break
            cos_t = rng.uniform(-1.0, 1.0)
            phi   = rng.uniform(0.0, 2.0 * math.pi)
            sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
            pos[i] = r * np.array([
                sin_t * math.cos(phi),
                sin_t * math.sin(phi),
                cos_t,
            ])

    return pos.astype(np.float64), mode


def _layout_normals(n, mode, seed):
    """Modus-spezifische Scheibenneigungen."""
    def _norm(v):
        a = np.array(v, dtype=np.float64)
        return tuple(a / np.linalg.norm(a))

    if mode == _MODE_FLYBY:
        return [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]
    if mode == _MODE_TRIPLE:
        return [
            (0.0, 0.0, 1.0),
            _norm((0.18, 0.18, 0.97)),
            _norm((-0.55, 0.10, 0.83)),
        ]
    # Fibonacci-Kugel fuer N > 3
    normals = [(0.0, 0.0, 1.0)]
    for i in range(1, n):
        frac = i / max(n - 1, 1)
        phi  = math.acos(1.0 - 2.0 * frac)
        az   = 2.0 * math.pi * i / 1.618033988749
        nv   = np.array([
            math.sin(phi) * math.cos(az),
            math.sin(phi) * math.sin(az),
            math.cos(phi),
        ])
        nv  /= np.linalg.norm(nv)
        normals.append(tuple(nv))
    return normals


def _hsv_colors(n):
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.72, 0.96)
        colors.append((r, g, b))
    return colors


# ------------------------------------------------------------------------------
# Orbital-Geschwindigkeiten (3D, modus-abhaengig)
# ------------------------------------------------------------------------------

def _orbital_velocities(pos, masses, G, eccentricity, mode, rng):
    """
    Physikalisch motivierte 3D-Orbital-Geschwindigkeiten.

    Nicht-Cluster-Modi (N<=12): kraftbasiert + kohaerente Rotationsachse,
    Vis-Viva Apoapse-Korrektur, Gesamtimpuls = 0.

    Cluster-Modus (N>=13): Virial-Theorem, sub-virial (70%) fuer
    'einfallendes' Aussehen.
    """
    N   = len(pos)
    Mt  = masses.sum()
    com = (pos * masses[:, np.newaxis]).sum(0) / Mt
    vel = np.zeros((N, 3), dtype=np.float64)
    e   = max(0.0, min(0.9999, eccentricity))

    if mode == _MODE_CLUSTER:
        V_grav = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = float(np.linalg.norm(pos[j] - pos[i]))
                if d > 1e-10:
                    V_grav -= G * masses[i] * masses[j] / d
        sigma = math.sqrt(abs(V_grav) / Mt / 3.0) * 0.70
        for i in range(N):
            vel[i] = rng.normal(0., sigma, 3)
    else:
        L_axis = np.array([0., 0., 1.])
        for i in range(N):
            rv = pos[i] - com
            r  = float(np.linalg.norm(rv))
            if r < 1e-8:
                continue
            r_hat = rv / r
            F = np.zeros(3, dtype=np.float64)
            for j in range(N):
                if i == j:
                    continue
                dv = pos[j] - pos[i]
                d  = float(np.linalg.norm(dv))
                if d < 1e-10:
                    continue
                F += G * masses[i] * masses[j] / (d * d) * (dv / d)
            F_c = float(abs(np.dot(F, r_hat)))
            if F_c < 1e-30:
                continue
            v_c = math.sqrt(F_c * r / masses[i])
            v_t = v_c * math.sqrt(2.0 * (1.0 - e) / (1.0 + e))
            ref = L_axis if abs(float(np.dot(r_hat, L_axis))) < 0.95 else np.array([1., 0., 0.])
            t   = np.cross(ref, r_hat)
            nt  = float(np.linalg.norm(t))
            if nt < 1e-10:
                continue
            vel[i] = (t / nt) * v_t

    p_total = (vel * masses[:, np.newaxis]).sum(0)
    vel    -= p_total[np.newaxis, :] / Mt
    return vel


# ------------------------------------------------------------------------------
# Haupt-Generator
# ------------------------------------------------------------------------------

def generate_n_galaxy_disks(
    n_galaxies:      int   = 3,
    n_stars:         int   = 50_000,
    disk_radius:     float = 80.0,
    sep:             float = 750.0,
    eccentricity:    float = 0.78,
    G:               float = 1.0,
    dm_enabled:      bool  = True,
    dm_M_vir:        float = _DM_M_VIR_DEFAULT,
    dm_r_s:          float = _DM_R_S_DEFAULT,
    dm_c:            float = _DM_C_DEFAULT,
    bh_mass:         float = 14_000.0,
    thickness_ratio: float = 0.012,
    v_dispersion:    float = 0.045,
    seed:            int   = 42,
) -> tuple:
    """
    Erzeuge n_galaxies (2...20) Galaxienscheiben.

    Der Layout-Modus wird automatisch aus n_galaxies bestimmt:
      N=2   Flyby / Merger
      N=3   Hierarchisches Triplett
      N=4-6  Kompakte Gruppe (Platonsiche 3D-Geometrie)
      N=7-12  Lockere Gruppe (Untergruppen + Ausreisser)
      N=13-20 Galaxienhaufen (NFW-Radialverteilung)

    Rueckgabe: (galaxies, dm_halo_configs)
    """
    n_galaxies = max(2, min(20, int(n_galaxies)))
    n_stars    = max(100, int(n_stars))

    rng = np.random.default_rng(seed)

    stars_per    = max(10, n_stars // n_galaxies)
    stars_counts = [stars_per] * n_galaxies
    stars_counts[-1] += n_stars - stars_per * n_galaxies

    # Layout
    pos_arr, mode = _layout_positions(n_galaxies, sep, rng)
    centers_list  = [tuple(pos_arr[i].tolist()) for i in range(n_galaxies)]
    normals       = _layout_normals(n_galaxies, mode, seed)
    colors        = _hsv_colors(n_galaxies)

    print(f'[Preset] Layout-Modus: {mode}')
    for i, c in enumerate(centers_list):
        print(f'  [{i}] ({c[0]:+.0f}, {c[1]:+.0f}, {c[2]:+.0f})')

    # Effektive Masse
    mean_star_mass = 2.5
    m_baryon       = bh_mass + stars_per * mean_star_mass
    if dm_enabled and dm_M_vir > 0.0:
        f_c      = math.log(1.0 + dm_c) - dm_c / (1.0 + dm_c)
        nfw_norm = dm_M_vir / f_c
        x_sep    = sep / dm_r_s
        m_enc_dm = nfw_norm * (math.log(1.0 + x_sep) - x_sep / (1.0 + x_sep))
        m_eff    = m_baryon + m_enc_dm
    else:
        m_eff = m_baryon
    masses_eff = np.full(n_galaxies, m_eff, dtype=np.float64)
    print(f'[Preset] m_baryon={m_baryon:.0f}  m_eff={m_eff:.0f} pro Galaxie')

    # Orbital-Geschwindigkeiten
    bulk_vels = _orbital_velocities(pos_arr, masses_eff, G, eccentricity, mode, rng)
    v_norms   = [float(np.linalg.norm(bulk_vels[i])) for i in range(n_galaxies)]
    print(f'[Preset] Orbital-|v|: {[f"{v:.1f}" for v in v_norms]}')

    # Galaxien erzeugen
    galaxies: list = []
    for i in range(n_galaxies):
        n_i        = stars_counts[i]
        arms_i     = 2 + (i % 4)
        tightness  = 0.35 + 0.08 * (i % 4)
        v_fac      = 1.15 + 0.04 * (i % 3)
        retrograde = (i % 3 == 1)
        n_accr     = max(30, min(300, 220 * n_i // 16000))

        gal = generate_galaxy_disk(
            n_stars         = n_i,
            bh_mass         = bh_mass,
            disk_radius     = disk_radius,
            thickness_ratio = thickness_ratio,
            center          = centers_list[i],
            bulk_velocity   = tuple(bulk_vels[i].tolist()),
            normal          = normals[i],
            arms            = arms_i,
            arm_tightness   = tightness,
            v_factor        = v_fac,
            v_dispersion    = v_dispersion,
            r_inner         = 4.0,
            mass_center     = 6.0,
            mass_edge       = 0.7,
            size_center     = 3.8,
            size_edge       = 0.9,
            disk_color      = colors[i],
            n_accretion     = n_accr,
            retrograde_lower= retrograde,
            G               = G,
            seed            = seed + i * 100,
            dm_M_vir        = dm_M_vir if dm_enabled else None,
            dm_r_s          = dm_r_s   if dm_enabled else None,
            dm_c            = dm_c,
        )
        for s in gal:
            s['softening'] = 6.0 if s.get('is_bh', False) else 1.2
        galaxies.append(gal)

    # DM-Halo-Konfigurationen
    dm_cfgs: list = []
    if dm_enabled:
        for i in range(n_galaxies):
            dm_cfgs.append({
                'center': centers_list[i],
                'M_vir':  dm_M_vir,
                'r_s':    dm_r_s,
                'c':      dm_c,
            })

    total     = sum(len(g) for g in galaxies)
    dist_info = [len(g) for g in galaxies]
    print(f'[Preset] {n_galaxies} Galaxien  .  {total:,} Partikel gesamt  {dist_info}')
    return galaxies, dm_cfgs