"""
MergerKernel.py  –  Verschmelzungsphysik, AGN-Jet-Feedback & Systemstatistik
════════════════════════════════════════════════════════════════════════════════
Physikalische Grundlagen
────────────────────────
  Masseerhaltung:
      M_neu = M_1 + M_2

  Impulserhaltung (inelastischer Stoß):
      p_neu = p_1 + p_2  →  v_neu = (M_1·v_1 + M_2·v_2) / M_neu

  Schwerpunktserhaltung (Massenzentrum des Merger-Paares):
      r_neu = (M_1·r_1 + M_2·r_2) / M_neu

  Bindungsenergie / Zerlegungsrate (Fragmentierungsquote):
      E_kin,i = ½·m_i·v_i²
      E_pot,i ≈ -G·M_ges·m_i / (|r_i - r_cm| + ε)   [Punktmassen-Näherung]
      Ungebunden: E_tot,i = E_kin,i + E_pot,i > 0
      Zerlegungsrate = n_ungebunden / n_aktiv  ∈ [0, 1]

  Systemenergie (Virial-Näherung):
      E_ges = E_kin + ½·E_pot
      (½ weil jedes Paar-Potential doppelt gezählt würde)

Merger-Hierarchie
─────────────────
  1. BH–BH    : wenn d(BH_a, BH_b) < r_merge_bh
                → schwererer BH überlebt, leichterer inaktiv (mass=0)
  2. BH–Stern : für jeden aktiven BH vectorisiert über alle Sterne
                wenn d(Stern, BH) < r_accrete  → Stern in BH akkretion
  3. Stern–Stern: durch Softening-ε abgedeckt (keine explizite Merger nötig)

  AGN-Jet-Feedback
─────────────────
  Wenn ein BH genug Masse akkretiert hat, wird ein bipolarer Jet ausgelöst:
      Jet-Richtung:  aus dem Drehimpuls der benachbarten Sterne berechnet
      Jet-Konus:     halber Öffnungswinkel α ≈ 12°
      Jet-Geschw.:   v_jet = √(2·G·M_BH / r_s) × f_boost  (Escape Velocity × f)
      Betroffene:    Partikel im Konus und innerhalb r_jet

Inaktive Partikel
─────────────────
  mass[i] = 0.0  →  werden im Octree und in der Integration übersprungen
  pos[i]  = (OFF, OFF, OFF)  →  außerhalb des Sichtbereichs, GPU clippt sie
════════════════════════════════════════════════════════════════════════════════
"""
import numpy as np

_OFF = 1e7    # "off-screen"-Sentinel für inaktive Partikel


def apply_mergers(pos, vel, mass, bh_indices,
                  r_merge_bh: float = 6.0,
                  r_accrete:  float = 3.0):
    """
    Führt BH–BH-Verschmelzung und BH–Stern-Akkretion durch.
    Modifiziert pos / vel / mass **in-place**.

    Rückgabe: (n_bh_merged, n_stars_accreted, accreted_mass_per_bh)
    accreted_mass_per_bh: dict {bh_index: akkretion_masse_in_diesem_schritt}
    """
    active_bh   = [i for i in bh_indices if mass[i] > 0.]
    dead        = set()
    n_bh_merge  = 0
    n_acc_total = 0

    # ── BH–BH  (O(k²), k = Anzahl aktiver SMBHs ≤ 3) ──────────────────
    for ai in range(len(active_bh)):
        for bi in range(ai + 1, len(active_bh)):
            a, b = active_bh[ai], active_bh[bi]
            if a in dead or b in dead:
                continue
            dist = float(np.linalg.norm(pos[a] - pos[b]))
            if dist >= r_merge_bh:
                continue

            # Schwererer überlebt (physikalisch: der kompaktere Objekt akkretiert)
            if mass[b] > mass[a]:
                a, b = b, a

            mt     = mass[a] + mass[b]
            # Impulserhaltung: v_neu = (m_a·v_a + m_b·v_b) / M_neu
            vel[a] = (vel[a] * mass[a] + vel[b] * mass[b]) / mt
            # Schwerpunktserhaltung: r_neu = (m_a·r_a + m_b·r_b) / M_neu
            pos[a] = (pos[a] * mass[a] + pos[b] * mass[b]) / mt
            mass[a] = mt
            mass[b] = 0.
            pos[b]  = np.full(3, _OFF, dtype=np.float64)
            dead.add(b)
            n_bh_merge += 1
            print(f'  [BH-Merger] idx {b} → idx {a}  |  M_neu = {mt:.1f}')

    # ── BH–Stern-Akkretion  (O(k·N), vektorisiert via NumPy) ───────────
    r2_accrete       = r_accrete * r_accrete
    accreted_mass_bh = {}          # bh_idx → akkretierte Masse in diesem Schritt
    for bh in active_bh:
        if bh in dead or mass[bh] <= 0.:
            continue
        # Quadrat-Abstand aller Partikel zum BH (vektorisiert, kein Python-Loop)
        dr2      = ((pos - pos[bh]) ** 2).sum(axis=1)
        accreted = (dr2 < r2_accrete) & (mass > 0.)
        accreted[bh] = False
        if not accreted.any():
            continue
        m_acc   = mass[accreted].sum()
        p_acc   = (vel[accreted] * mass[accreted, np.newaxis]).sum(axis=0)
        mt      = mass[bh] + m_acc
        # Impulserhaltung für Akkretion
        vel[bh]       = (vel[bh] * mass[bh] + p_acc) / mt
        mass[bh]      = mt
        mass[accreted]= 0.
        pos[accreted] = _OFF
        n_acc_total  += int(accreted.sum())
        accreted_mass_bh[bh] = float(m_acc)

    return n_bh_merge, n_acc_total, accreted_mass_bh


def apply_agn_jets(bh_idx, accreted_mass_bh, pos, vel, mass,
                   jet_threshold = 20.,    # Mindest-Akk.-Masse für Jet-Auslösung
                   jet_boost     = 2.2,    # v_jet = jet_boost × Escape-Velocity
                   jet_half_angle= 0.21,   # Halbwinkel [rad] ≈ 12°
                   jet_radius    = 60.,    # Maximale Jet-Reichweite
                   G             = 1.0):
    """
    Bipolarer AGN-Jet-Feedback, ausgelöst nach signifikanter Akkretion.

    Physik:
      - Jet-Richtung: aus Drehimpuls benachbarter Sterne geschätzt (L = Σ r×v)
      - Geschwindigkeit: v_jet = jet_boost × v_esc = jet_boost × √(2GM_BH/r)
      - Öffnungswinkel: cos(α) > cos(jet_half_angle) → enggerichteter Jet
      - Beidseits (+L̂ und −L̂) → bipolarer Ausfluss

    Gibt Anzahl der vom Jet beeinflussten Partikel zurück.
    """
    n_kicked = 0
    r2_jet   = jet_radius * jet_radius

    for bh in bh_idx:
        if mass[bh] <= 0.:
            continue
        m_acc = accreted_mass_bh.get(bh, 0.)
        if m_acc < jet_threshold:
            continue

        # ── Jet-Achse: Drehimpuls der nächsten Sterne ──────────────────
        dr_all = pos - pos[bh][np.newaxis, :]
        r2_all = (dr_all * dr_all).sum(axis=1)
        near   = (r2_all < (jet_radius * 0.5) ** 2) & (r2_all > 1.) & (mass > 0.)
        near[bh] = False

        if near.sum() > 3:
            r_near = dr_all[near]
            v_near = vel[near]
            m_near = mass[near]
            # Gewichtetes Drehimpuls-Mittel: L = Σ m_i (r_i × v_i)
            L = np.cross(r_near, v_near)
            L_weighted = (L * m_near[:, np.newaxis]).sum(axis=0)
            L_norm = np.linalg.norm(L_weighted)
            jet_axis = L_weighted / L_norm if L_norm > 1e-6 else np.array([0., 0., 1.])
        else:
            jet_axis = np.array([0., 0., 1.])   # Fallback: z-Achse

        # ── Escape-Velocity des BH (Softening: r_min=2) ─────────────────
        v_esc = np.sqrt(2. * G * mass[bh] / max(4., 2.))
        v_jet = jet_boost * v_esc

        # ── Partikel im Jet-Konus finden ─────────────────────────────────
        mask = (r2_all < r2_jet) & (r2_all > 1.) & (mass > 0.)
        mask[bh] = False
        if not mask.any():
            continue

        dr_jet  = dr_all[mask]
        r_len   = np.sqrt(r2_all[mask])
        r_hat   = dr_jet / r_len[:, np.newaxis]
        cos_ang = (r_hat * jet_axis[np.newaxis, :]).sum(axis=1)

        # Jet-Konus: |cos(θ)| > cos(half_angle) → beide Pole
        in_cone = np.abs(cos_ang) > np.cos(jet_half_angle)
        jet_idx = np.where(mask)[0][in_cone]

        if len(jet_idx) == 0:
            continue

        # Kick-Richtung: entlang der Jet-Achse (Vorzeichen = jeweiliger Pol)
        dr_cone = dr_all[jet_idx]
        r_cone  = np.sqrt((dr_cone * dr_cone).sum(axis=1))
        sign    = np.sign((dr_cone / r_cone[:, np.newaxis] * jet_axis).sum(axis=1))
        vel[jet_idx] += jet_axis[np.newaxis, :] * (sign * v_jet)[:, np.newaxis]
        n_kicked += len(jet_idx)

    return n_kicked



def compute_stats(pos, vel, mass, G: float = 1.0):
    """
    Berechnet Systemstatistiken für den aktuellen Frame.

    Zerlegungsrate  (Fragmentierungsquote):
        Anteil der Partikel mit positiver Gesamtenergie → ungebunden.
        E_tot,i = ½·m_i·v_i² − G·M_ges·m_i / (r_i + 1)

    Systemenergie  (Virial-Näherung, O(N)):
        E = E_kin + ½·E_pot   (½ = Virial-Faktor, vermeidet Doppelzählung)

    Rückgabe: (zerlegungsrate [0..1], E_system, n_aktiv)
    """
    active = mass > 0.
    n_act  = int(active.sum())
    if n_act == 0:
        return 0., 0., 0

    ap  = pos[active]
    av  = vel[active]
    am  = mass[active]
    Mt  = am.sum()
    com = (ap * am[:, np.newaxis]).sum(axis=0) / Mt

    r   = np.sqrt(((ap - com) ** 2).sum(axis=1)) + 1.   # +1 = Softening
    v2  = (av ** 2).sum(axis=1)

    Ek  = 0.5 * am * v2                    # kinetische Energie
    Ep  = -G * Mt * am / r                 # pot. Energie (Punktmassen-Näherung)
    Et  = Ek + Ep

    frag   = float((Et > 0.).sum()) / n_act
    E_sys  = float(Ek.sum() + 0.5 * Ep.sum())
    return frag, E_sys, n_act
