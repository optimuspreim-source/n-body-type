"""
galaxy_disk.py – Physikalisch fundierte 3D-Galaxienscheibe
  - Sech2-Dickenprofil: Durchmesser:Dicke ~ 100:1 im Schnitt, dicker Kern, dünnes Randband
  - Massegradienten: Kroupa-IMF-basierte Sternmassen, schwer im Kern, leicht am Rand
  - Zwei-lagige Scheibe (obere/untere Fläche, untere leicht retrograd)
  - SMBH mit photorealistischem Akkretionsring (Photon-Ring-Näherung)
  - Exzentrische gebundene Orbitalgeschwindigkeiten für sequenzielle Kollision
  - Optionale NFW-DM-Halo-Korrektur der Kreisbahngeschwindigkeit
"""
import numpy as np


def _sample_kroupa_imf(rng, n, m_min=0.08, m_max=80.0):
    """
    Kroupa (2001) IMF via Rejection-Sampling mit log-uniformem Proposal.
    Zwei-Segment Potenzgesetz:
      ξ(m) ∝ m^{-1.3}  für  m_min < m < 0.5  (M_sun)
      ξ(m) ∝ m^{-2.3}  für  0.5  ≤ m < m_max  (M_sun)
    """
    m_break = 0.5
    samples = []
    n_need  = n
    while n_need > 0:
        log_m = rng.uniform(np.log(m_min), np.log(m_max), n_need * 4)
        m = np.exp(log_m)
        # IMF × m  (Korrektur für log-uniformen Proposal p ∝ 1/m)
        xi = np.where(m < m_break,
                      m ** (1. - 1.3),
                      m_break ** (2.3 - 1.3) * m ** (1. - 2.3))
        xi /= xi.max()
        ok = m[rng.random(len(m)) < xi][:n_need]
        samples.append(ok)
        n_need -= len(ok)
    return np.concatenate(samples)[:n]


def rotation_matrix_to_normal(normal):
    n = np.array(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    z = np.array([0., 0., 1.])
    if np.allclose(n,  z): return np.eye(3)
    if np.allclose(n, -z): return np.diag([1., -1., -1.])
    v  = np.cross(z, n)
    s  = np.linalg.norm(v)
    c  = np.dot(z, n)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def generate_galaxy_disk(
    n_stars         = 15000,
    bh_mass         = 12000.0,
    disk_radius     = 80.0,
    thickness_ratio = 0.010,        # avg Dicke = disk_radius * ratio
    center          = (0, 0, 0),
    bulk_velocity   = (0, 0, 0),
    normal          = (0, 0, 1),
    arms            = 2,
    arm_tightness   = 0.40,
    v_factor        = 1.18,
    v_dispersion    = 0.04,
    r_inner         = 3.5,
    mass_center     = 5.0,
    mass_edge       = 0.6,
    size_center     = 0.18,
    size_edge       = 0.04,
    disk_color      = (1.0, 0.90, 0.35),   # RGB Zentrumsfarbe
    n_accretion     = 280,
    retrograde_lower= True,
    G               = 1.0,
    seed            = None,
    # ── Optionale NFW-DM-Halo-Parameter für realistische Rotationskurve ──
    dm_M_vir        = None,    # Viriale DM-Masse  (None = kein DM-Beitrag)
    dm_r_s          = None,    # NFW-Skalenradius
    dm_c            = 12.0,    # Konzentrationsparameter
):
    rng    = np.random.default_rng(seed)
    R      = rotation_matrix_to_normal(normal)
    center = np.array(center, dtype=np.float64)
    bv     = np.array(bulk_velocity, dtype=np.float64)
    r_sc   = disk_radius / 3.5

    # ── NFW-Dunkle-Materie: Normierungsfaktor für M_enc(r) ─────────────────
    _dm_norm = 0.  # = 4π ρ_s r_s³ = M_vir / f(c)
    _dm_rs   = 1.  # Skalenradius (dummy, überschrieben wenn dm_M_vir gesetzt)
    if dm_M_vir is not None and dm_r_s is not None:
        f_c      = np.log(1. + dm_c) - dm_c / (1. + dm_c)
        _dm_norm = dm_M_vir / f_c
        _dm_rs   = float(dm_r_s)

    # ── Radialverteilung: exponentiell ──────────────────────────────────
    u = rng.uniform(0., 1., n_stars)
    r = -r_sc * np.log(1. - u * (1. - np.exp(-disk_radius / r_sc)))
    r = np.clip(r, r_inner, disk_radius)

    # ── Spiralarme ──────────────────────────────────────────────────────
    arm_idx   = rng.integers(0, arms, n_stars)
    theta_arm = arm_idx * (2 * np.pi / arms)
    spiral    = arm_tightness * np.log1p(r / r_sc)
    scatter_w = rng.normal(0., 0.20 * (1. + 0.5 * r / disk_radius), n_stars)
    theta     = theta_arm + spiral + scatter_w

    # ── Zwei-lagige Scheibe (obere/untere Fläche) ────────────────────────
    upper  = rng.random(n_stars) > 0.5

    # Sech²-Dickenprofil: dick im Zentrum, hauchdünn am Rand
    z_max   = disk_radius * thickness_ratio * 2.8
    z_scale = disk_radius * 0.22
    z_sigma = z_max / np.cosh(r / z_scale)**2 + disk_radius * thickness_ratio * 0.04
    z_sign  = np.where(upper, 1., -1.)
    z_loc   = z_sign * np.abs(rng.normal(0., z_sigma, n_stars))

    x_loc = r * np.cos(theta)
    y_loc = r * np.sin(theta)
    local = np.stack([x_loc, y_loc, z_loc], axis=1)
    world = (R @ local.T).T + center

    # ── Tangentiale Geschwindigkeiten (─ mit optionalem DM-Beitrag) ────────
    M_enc_bary = bh_mass + n_stars * (mass_center + mass_edge) * 0.5 * (r / disk_radius) ** 1.5
    # NFW-DM-Beitrag: M_enc_DM(r) = _dm_norm × [ln(1+r/r_s) - (r/r_s)/(1+r/r_s)]
    if _dm_norm > 0.:
        x_dm   = r / _dm_rs
        M_enc_dm = _dm_norm * (np.log(1. + x_dm) - x_dm / (1. + x_dm))
    else:
        M_enc_dm = 0.
    M_enc  = M_enc_bary + M_enc_dm
    v_circ = np.sqrt(G * M_enc / r)
    v_tang = v_factor * v_circ

    tang_x = -np.sin(theta)
    tang_y =  np.cos(theta)
    tang_l = np.stack([tang_x, tang_y, np.zeros(n_stars)], axis=1)
    tang_w = (R @ tang_l.T).T

    # Untere Hälfte: leicht retrograd (-6%) → Doppel-Spiralstruktur sichtbar
    if retrograde_lower:
        tang_w[~upper]  *= -1.
        v_tang[~upper]  *= 0.94

    noise = rng.normal(0., v_dispersion * v_tang[:, np.newaxis], (n_stars, 3))
    vel_w = tang_w * v_tang[:, np.newaxis] + noise + bv

    # ── Masse- und Größengradienten (Kroupa-IMF + radialer Gradient) ────────
    t_r      = np.clip(r / disk_radius, 0., 1.)
    # Kroupa-IMF: massereiche Sterne häufiger im Zentrum, leichte am Rand
    imf_raw  = _sample_kroupa_imf(rng, n_stars, m_min=0.08, m_max=80.0)
    imf_min, imf_max = imf_raw.min(), imf_raw.max()
    imf_norm = (imf_raw - imf_min) / (imf_max - imf_min + 1e-12)  # [0, 1]
    # Radialer Gradient (60%) + IMF-Streuung (40%) = realistische Massenverteilung
    m_local  = mass_edge + (mass_center - mass_edge) * np.exp(-2.8 * t_r)
    m_imf    = mass_edge * 0.4 + imf_norm * (mass_center * 2.2 - mass_edge * 0.4)
    star_m   = 0.60 * m_local + 0.40 * m_imf
    star_m   = np.clip(star_m, mass_edge * 0.4, mass_center * 2.2)
    star_sz  = size_edge + (size_center - size_edge) * np.exp(-2.5 * t_r)

    # Farbe: Zentrum = helle Galaxiefarbe → Rand = dunkles Braun-Rot
    dc      = np.array(disk_color, dtype=np.float64)
    edge_c  = np.array([0.55, 0.20, 0.06], dtype=np.float64)
    blend   = np.exp(-2.5 * t_r)[:, np.newaxis]
    rgb     = blend * dc[np.newaxis, :] + (1. - blend) * edge_c[np.newaxis, :]
    alpha   = (0.95 - 0.58 * t_r).reshape(-1, 1)
    star_col = np.concatenate([rgb, alpha], axis=1).astype(np.float32)

    # ── SMBH ────────────────────────────────────────────────────────────
    bh_entry = {
        'position':    tuple(center),
        'velocity':    tuple(bv),
        'mass':        float(bh_mass),
        'is_bh':       True,
        'render_size': 0.1,                   # wird separat als Shadow gerendert
        'render_color': (0.02, 0.02, 0.02, 1.),
    }

    # ── Akkretionsring (Photon-Ring-Näherung) ─────────────────────────────
    n_acc   = n_accretion
    # Ring-Innenradius = r_inner (innerste stabile Kreisbahn der Scheibe),
    # Außenradius = r_inner * 2.5  →  kein Partikel innerhalb des Akkretionsradius
    acc_r   = np.linspace(r_inner, r_inner * 2.5, n_acc)
    acc_th  = np.linspace(0., 2 * np.pi, n_acc, endpoint=False)
    acc_z   = rng.normal(0., r_inner * 0.06, n_acc)
    acc_loc = np.stack([acc_r * np.cos(acc_th), acc_r * np.sin(acc_th), acc_z], axis=1)
    acc_wld = (R @ acc_loc.T).T + center
    v_ka    = np.sqrt(G * bh_mass / np.maximum(acc_r, 0.1)) * 1.02
    acc_tl  = np.stack([-np.sin(acc_th), np.cos(acc_th), np.zeros(n_acc)], axis=1)
    acc_tw  = (R @ acc_tl.T).T
    vel_acc = acc_tw * v_ka[:, np.newaxis] + bv

    # Photon-Ring Farbe: strahlend weiß-orange innen → dunkelrot außen
    t_acc   = np.clip((acc_r - acc_r.min()) / (acc_r.max() - acc_r.min() + 1e-8), 0., 1.)
    acc_col = np.stack([
        np.ones(n_acc),
        np.clip(0.88 - 0.55 * t_acc, 0.25, 0.88),
        np.clip(0.35 - 0.35 * t_acc, 0.,   0.35),
        np.ones(n_acc) * 0.98,
    ], axis=1).astype(np.float32)
    acc_sz  = (0.19 - 0.11 * t_acc).astype(np.float32)

    # ── Zusammenstellen ──────────────────────────────────────────────────
    stars = [bh_entry]
    for i in range(n_acc):
        stars.append({
            'position':    tuple(acc_wld[i]),
            'velocity':    tuple(vel_acc[i]),
            'mass':        float(bh_mass * 0.00015),
            'is_bh':       False,
            'render_size': float(acc_sz[i]),
            'render_color': tuple(acc_col[i]),
        })
    for i in range(n_stars):
        stars.append({
            'position':    tuple(world[i]),
            'velocity':    tuple(vel_w[i]),
            'mass':        float(star_m[i]),
            'is_bh':       False,
            'render_size': float(star_sz[i]),
            'render_color': tuple(star_col[i]),
        })
    return stars


# ── Orbitalgeschwindigkeiten für exzentrische gebundene 3-Körper-Bahnen ──────

def compute_orbital_velocities(positions, total_masses, G=1.0, eccentricity=0.75):
    """Vis-Viva: Apoapsis-Geschwindigkeit für e=eccentricity. CoM-Impuls=0."""
    pos  = np.array(positions,    dtype=np.float64)
    mass = np.array(total_masses, dtype=np.float64)
    N    = len(pos)
    Mt   = mass.sum()
    com  = (pos * mass[:, np.newaxis]).sum(axis=0) / Mt
    vel  = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        rv  = pos[i] - com
        r   = np.linalg.norm(rv)
        if r < 1e-6: continue
        Mo  = Mt - mass[i]
        vc  = np.sqrt(G * Mo / r)
        vm  = vc * np.sqrt(max(1. - eccentricity, 0.01))   # Apoapsis-Geschw.
        rh  = rv / r
        up  = np.array([0., 0., 1.]) if abs(rh[2]) < 0.9 else np.array([1., 0., 0.])
        t   = np.cross(rh, up);  t /= np.linalg.norm(t)
        vel[i] = t * vm * 0.72 + (-rh) * vm * 0.28     # 72% tang + 28% radial einwärts
    p = (vel * mass[:, np.newaxis]).sum(axis=0)
    for i in range(N): vel[i] -= p / Mt
    return vel


# ── Drei Galaxien-Varianten ───────────────────────────────────────────────────

GALAXY_PRESETS = {
    # A – Große 2-Arm-Spirale, face-on, golden
    'galaxy_A': dict(
        n_stars=13500, bh_mass=12000., disk_radius=82.,
        thickness_ratio=0.010, normal=(0, 0, 1), arms=2,
        arm_tightness=0.37, v_factor=1.18, v_dispersion=0.03,
        r_inner=4.0, mass_center=5.0, mass_edge=0.6,
        size_center=0.18, size_edge=0.04,
        disk_color=(1.0, 0.90, 0.30),
        n_accretion=240, retrograde_lower=True, seed=42,
    ),
    # B – Mittlere 3-Arm-Spirale, 60° geneigt, cyan
    'galaxy_B': dict(
        n_stars=14500, bh_mass=9500., disk_radius=72.,
        thickness_ratio=0.011, normal=(0., 0.866, 0.5), arms=3,
        arm_tightness=0.42, v_factor=1.20, v_dispersion=0.04,
        r_inner=3.5, mass_center=4.0, mass_edge=0.5,
        size_center=0.15, size_edge=0.035,
        disk_color=(0.25, 0.88, 1.0),
        n_accretion=210, retrograde_lower=True, seed=7,
    ),
    # C – Kompakte 4-Arm-Spirale, 45° anders geneigt, magenta (einzigartiges Objekt)
    'galaxy_C': dict(
        n_stars=11500, bh_mass=7000., disk_radius=60.,
        thickness_ratio=0.012, normal=(0.707, 0., 0.707), arms=4,
        arm_tightness=0.46, v_factor=1.15, v_dispersion=0.035,
        r_inner=3.0, mass_center=3.5, mass_edge=0.5,
        size_center=0.14, size_edge=0.035,
        disk_color=(1.0, 0.35, 0.85),
        n_accretion=170, retrograde_lower=False, seed=99,
    ),
}

_CENTERS = [(-500., -380., 180.), (550., -300., -120.), (-60., 650., -80.)]
_APPROX_MASSES = {'galaxy_A': 12000+13500*2.8, 'galaxy_B': 9500+14500*2.5, 'galaxy_C': 7000+11500*2.0}


def generate_formation(overrides=None, G=1.0, eccentricity=0.75):
    cfg    = {k: dict(v) for k, v in GALAXY_PRESETS.items()}
    if overrides:
        for k, ov in overrides.items():
            if k in cfg: cfg[k].update(ov)
    names  = ['galaxy_A', 'galaxy_B', 'galaxy_C']
    tmass  = [_APPROX_MASSES[n] for n in names]
    bv     = compute_orbital_velocities(_CENTERS, tmass, G, eccentricity)
    out    = []
    for i, name in enumerate(names):
        p = dict(cfg[name])
        p['center']        = _CENTERS[i]
        p['bulk_velocity'] = tuple(bv[i])
        p['G']             = G
        print(f'  {name}: {p["n_stars"]} Sterne + BH({p["bh_mass"]:.0f}M) | bv={tuple(round(x,2) for x in bv[i])}')
        out.append(generate_galaxy_disk(**p))
    return out


if __name__ == '__main__':
    gs = generate_formation()
    for i, g in enumerate(gs):
        print(f'Galaxie {i}: {len(g)} Partikel')
