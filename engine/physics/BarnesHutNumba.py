"""
BarnesHutNumba.py  –  O(N log N) Gravitations-Integrator
════════════════════════════════════════════════════════════════════════════════
Mathematische Grundlagen
────────────────────────
  Gravitationskraft (gesoftet):
      F_ij = G·m_i·m_j / (|r_ij|² + ε²)^(3/2) · r_ij

  Barnes-Hut-Kriterium  (θ = öffnungswinkel):
      s / d < θ  →  Zellen-Schwerpunkt als Näherung
      Komplexität: O(N log N) statt O(N²)

  Leapfrog-Integrator  (symplektisch, 2. Ordnung):
      v_(n+1/2) = v_(n-1/2) + a_n · dt
      x_(n+1)   = x_n       + v_(n+1/2) · dt
      Energiedrift ~ O(dt²)  –  stabil für konservative Systeme

  Octree / AABB  (Achsen-ausgerichtete Bounding Box):
      Jeder Knoten speichert: Gesamtmasse M, Schwerpunkt r_cm, 8 Kinder.
      Blatt: genau ein Partikel.  Innerer Knoten: kein Partikel, M = Σm_i.

Implementierungsdetails
───────────────────────
  • Stack-basierte Insertion  (kein Python-Rekursions-Overhead, kein GIL-Hold)
  • Pre-allokierte Baum-Arrays  (kein Heap-Alloc pro Frame)
  • prange-Parallelisierung der Kraftschleife  (OpenMP via Numba)
  • fastmath=True  (SIMD-Nutzung, ~2× schneller auf AVX2-CPUs)
  • mass=0 → Partikel überspringen  (inaktive nach Merger)
════════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from numba import njit, prange

_INS_STACK  = 256   # max. Stack-Tiefe pro Partikel-Insertion  (log8(N)·2 + Reserve)
_FORC_STACK = 512   # max. Stack-Tiefe für Kraft-Traversierung


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktion: Kind-Knoten initialisieren
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _make_child(c, parent, bits, m0, m1, m2,
                t_min, t_max, t_mass, t_com, t_children, t_particle, t_eps2_max):
    """
    Erstellt Kind-Knoten c als Oktant 'bits' von 'parent'.
    bits kodiert Oktant binär: bit0=x, bit1=y, bit2=z  (0=unten, 1=oben)
    """
    t_mass[c]      = 0.
    t_eps2_max[c]  = 0.
    t_com[c,0]    = 0.; t_com[c,1] = 0.; t_com[c,2] = 0.
    t_particle[c] = -1
    for k in range(8): t_children[c, k] = -1
    t_min[c,0] = m0 if (bits & 1) else t_min[parent,0]
    t_max[c,0] = t_max[parent,0] if (bits & 1) else m0
    t_min[c,1] = m1 if (bits & 2) else t_min[parent,1]
    t_max[c,1] = t_max[parent,1] if (bits & 2) else m1
    t_min[c,2] = m2 if (bits & 4) else t_min[parent,2]
    t_max[c,2] = t_max[parent,2] if (bits & 4) else m2


# ─────────────────────────────────────────────────────────────────────────────
#  Octree-Aufbau  (seriell, O(N log N))
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _build_tree(pos, mass, eps2_arr, t_mass, t_com, t_min, t_max,
                t_children, t_particle, n_count, t_eps2_max, max_nodes):
    """
    Baut den Octree für alle aktiven Partikel (mass>0) neu auf.

    Algorithmus (Stack-basiert, iterativ):
      Für jedes Partikel i: lege (Wurzel, i) auf Stack.
      Schleife:
        Pop (Knoten nd, Partikel pi).
        Ist nd leer → Partikel direkt einsetzen.
        Sonst:
          • Masse & Schwerpunkt von nd akkumulieren  (M += m_i, r_cm ← gewichteter Mittel)
          • Ist nd ein Blatt (hat alten Partikel q) → Blatt zum inneren Knoten machen,
            (Kind_q, q) auf Stack legen.
          • (Kind_i, i) auf Stack legen.
      Das ist korrekt auch wenn zwei Partikel im selben Oktanten landen.

    Reset: Nur die im letzten Frame genutzten n_count[0] Knoten werden zurückgesetzt.
    """
    N = pos.shape[0]

    # ── Bounding Box (nur aktive Partikel) ──────────────────────────────
    mn0 = mn1 = mn2 =  1e30
    mx0 = mx1 = mx2 = -1e30
    has_active = False
    for i in range(N):
        if mass[i] <= 0.: continue
        has_active = True
        x = pos[i,0]; y = pos[i,1]; z = pos[i,2]
        if x < mn0: mn0 = x
        if y < mn1: mn1 = y
        if z < mn2: mn2 = z
        if x > mx0: mx0 = x
        if y > mx1: mx1 = y
        if z > mx2: mx2 = z
    if not has_active: return

    pad = max(mx0-mn0, mx1-mn1, mx2-mn2) * 0.012 + 2.

    # ── Genutzte Knoten des letzten Frames zurücksetzen ─────────────────
    n_prev = int(n_count[0])
    for j in range(n_prev):
        t_mass[j]      = 0.
        t_eps2_max[j]  = 0.
        t_particle[j]  = -1
        for k in range(8): t_children[j, k] = -1

    # ── Wurzel initialisieren ────────────────────────────────────────────
    n_count[0] = 1
    t_min[0,0] = mn0-pad; t_min[0,1] = mn1-pad; t_min[0,2] = mn2-pad
    t_max[0,0] = mx0+pad; t_max[0,1] = mx1+pad; t_max[0,2] = mx2+pad
    t_com[0,0] = 0.; t_com[0,1] = 0.; t_com[0,2] = 0.

    # ── Stack-Puffer (einmalig allokiert, pro Partikel wiederverwendet) ──
    stk_n = np.empty(_INS_STACK, dtype=np.int32)
    stk_p = np.empty(_INS_STACK, dtype=np.int32)

    for i in range(N):
        if mass[i] <= 0.: continue

        top = 0
        stk_n[0] = 0; stk_p[0] = i

        while top >= 0:
            nd = stk_n[top]; pi = stk_p[top]; top -= 1
            px = pos[pi,0]; py = pos[pi,1]; pz = pos[pi,2]
            mp = mass[pi]

            # ── Fall 1: Knoten leer → direkt belegen ────────────────────
            if t_mass[nd] == 0.:
                t_mass[nd]     = mp
                t_eps2_max[nd] = eps2_arr[pi]
                t_com[nd,0]    = px; t_com[nd,1] = py; t_com[nd,2] = pz
                t_particle[nd] = pi
                continue

            # ── Masse & Schwerpunkt + eps²_max akkumulieren ──────────────
            #   r_cm_neu = (M_alt·r_cm_alt + m_i·r_i) / (M_alt + m_i)
            tm = t_mass[nd] + mp
            t_com[nd,0] = (t_com[nd,0]*t_mass[nd] + px*mp) / tm
            t_com[nd,1] = (t_com[nd,1]*t_mass[nd] + py*mp) / tm
            t_com[nd,2] = (t_com[nd,2]*t_mass[nd] + pz*mp) / tm
            t_mass[nd]  = tm
            e = eps2_arr[pi]
            if e > t_eps2_max[nd]: t_eps2_max[nd] = e

            # Mitte des aktuellen Knotens
            m0 = (t_min[nd,0]+t_max[nd,0])*.5
            m1 = (t_min[nd,1]+t_max[nd,1])*.5
            m2 = (t_min[nd,2]+t_max[nd,2])*.5

            # ── Fall 2: Blatt → aufteilen, alten Partikel verschieben ────
            if t_particle[nd] >= 0:
                op = t_particle[nd]; t_particle[nd] = -1
                ob = ((1 if pos[op,0]>=m0 else 0)
                     |(2 if pos[op,1]>=m1 else 0)
                     |(4 if pos[op,2]>=m2 else 0))
                if t_children[nd, ob] < 0:
                    c = int(n_count[0])
                    if c < max_nodes:
                        n_count[0] += 1
                        _make_child(c, nd, ob, m0, m1, m2,
                                    t_min, t_max, t_mass, t_com, t_children, t_particle, t_eps2_max)
                        t_children[nd, ob] = c
                if t_children[nd, ob] >= 0 and top < _INS_STACK-2:
                    top += 1; stk_n[top] = t_children[nd, ob]; stk_p[top] = op

            # ── Neues Partikel in Oktant einfügen ────────────────────────
            nb = ((1 if px>=m0 else 0)
                 |(2 if py>=m1 else 0)
                 |(4 if pz>=m2 else 0))
            if t_children[nd, nb] < 0:
                c = int(n_count[0])
                if c < max_nodes:
                    n_count[0] += 1
                    _make_child(c, nd, nb, m0, m1, m2,
                                t_min, t_max, t_mass, t_com, t_children, t_particle, t_eps2_max)
                    t_children[nd, nb] = c
            if t_children[nd, nb] >= 0 and top < _INS_STACK-1:
                top += 1; stk_n[top] = t_children[nd, nb]; stk_p[top] = pi


# ─────────────────────────────────────────────────────────────────────────────
#  Kraftberechnung  (pro Partikel, Stack-basiert)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, inline='always', fastmath=True)
def _force_on(px, py, pz,
              t_mass, t_com, t_min, t_max, t_children, t_eps2_max,
              G, eps2, theta2):
    """
    Berechnet die Gravitationsbeschleunigung auf (px,py,pz).

    Barnes-Hut-Kriterium:
        s² / d² < θ²  →  Zellen-Schwerpunkt als Näherung genügend genau
        (s = Zellenlänge, d = Abstand zum Schwerpunkt)

    Softened Gravity:
        a = G·M_zelle / (d² + ε²)^(3/2) · Δr
        ε verhindert Singularitäten bei d→0  (numerische Stabilität)
    """
    fx = fy = fz = 0.
    stk = np.empty(_FORC_STACK, dtype=np.int32)
    top = 0; stk[0] = 0

    while top >= 0:
        nd = stk[top]; top -= 1
        if t_mass[nd] == 0.: continue

        drx = t_com[nd,0]-px; dry = t_com[nd,1]-py; drz = t_com[nd,2]-pz
        d2  = drx*drx + dry*dry + drz*drz

        if d2 < 1e-12:
            # Selbst-Interaktion: in Kinder abtauchen
            for k in range(8):
                ch = t_children[nd, k]
                if ch >= 0 and top < _FORC_STACK-1:
                    top += 1; stk[top] = ch
            continue

        s    = t_max[nd,0] - t_min[nd,0]
        leaf = True
        for k in range(8):
            if t_children[nd, k] >= 0: leaf = False; break

        if leaf or s*s/d2 < theta2:
            # Symmetrisches Softening: ε²_eff = max(ε²_Partikel, ε²_max_Knoten)
            # → Sterne in BH-Nähe erhalten automatisch das starke BH-Softening
            eff_eps2 = eps2 if eps2 >= t_eps2_max[nd] else t_eps2_max[nd]
            inv3 = G * t_mass[nd] * (d2 + eff_eps2) ** -1.5
            fx += inv3*drx; fy += inv3*dry; fz += inv3*drz
        else:
            for k in range(8):
                ch = t_children[nd, k]
                if ch >= 0 and top < _FORC_STACK-1:
                    top += 1; stk[top] = ch
    return fx, fy, fz


# ─────────────────────────────────────────────────────────────────────────────
#  Haupt-Simulations-Schritt  (parallel via prange)
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def nbody_step(pos, vel, mass, dt, G, eps2_arr, theta,
               t_mass, t_com, t_min, t_max, t_children, t_particle, n_count, t_eps2_max, max_nodes):
    """
    Vollständiger Leapfrog-Schritt mit per-Partikel-Softening:
      eps2_arr : 1D-Array der Länge N  –  BHs erhalten großes eps², Sterne kleines.
      1.  Octree aufbauen          (seriell, O(N log N))
      2.  Kräfte berechnen         (prange über N, parallel O(N log N))
      3.  Velocity-Verlet Update   (prange, O(N))

    Softened Gravity pro Partikel i:
      a_i = G·M_j / (|r_ij|² + eps2_arr[i])^(3/2) · r_ij
    """
    _build_tree(pos, mass, eps2_arr, t_mass, t_com, t_min, t_max,
                t_children, t_particle, n_count, t_eps2_max, max_nodes)

    N      = pos.shape[0]
    theta2 = theta * theta
    npos   = np.empty_like(pos)
    nvel   = np.empty_like(vel)

    for i in prange(N):
        if mass[i] <= 0.:
            npos[i,0]=pos[i,0]; npos[i,1]=pos[i,1]; npos[i,2]=pos[i,2]
            nvel[i,0]=vel[i,0]; nvel[i,1]=vel[i,1]; nvel[i,2]=vel[i,2]
            continue
        ax, ay, az = _force_on(pos[i,0], pos[i,1], pos[i,2],
                                t_mass, t_com, t_min, t_max, t_children, t_eps2_max,
                                G, eps2_arr[i], theta2)
        im = 1. / mass[i]
        nvel[i,0] = vel[i,0] + ax*im*dt
        nvel[i,1] = vel[i,1] + ay*im*dt
        nvel[i,2] = vel[i,2] + az*im*dt
        npos[i,0] = pos[i,0] + nvel[i,0]*dt
        npos[i,1] = pos[i,1] + nvel[i,1]*dt
        npos[i,2] = pos[i,2] + nvel[i,2]*dt
    return npos, nvel


# ─────────────────────────────────────────────────────────────────────────────
#  Helfer für Viewer
# ─────────────────────────────────────────────────────────────────────────────

def make_tree_arrays(max_nodes):
    """Pre-allokiert alle Baum-Arrays (einmalig, kein Heap-Alloc pro Frame)."""
    return (
        np.zeros(max_nodes,           dtype=np.float64),   # t_mass
        np.zeros((max_nodes, 3),      dtype=np.float64),   # t_com
        np.zeros((max_nodes, 3),      dtype=np.float64),   # t_min
        np.zeros((max_nodes, 3),      dtype=np.float64),   # t_max
        np.full( (max_nodes, 8), -1,  dtype=np.int32),     # t_children
        np.full( max_nodes,      -1,  dtype=np.int32),     # t_particle
        np.zeros(1,                   dtype=np.int64),     # n_count
        np.zeros(max_nodes,           dtype=np.float64),   # t_eps2_max  (symmetrisches BH-Softening)
    )


def warmup(N=256, max_nodes=4096):
    import time
    print('[BH-Numba] Kompiliere JIT-Kernel... ', end='', flush=True)
    t0   = time.perf_counter()
    p    = np.random.randn(N, 3).astype(np.float64) * 10
    v    = np.zeros((N, 3), dtype=np.float64)
    m    = np.ones(N, dtype=np.float64)
    eps2_arr = np.full(N, .25, dtype=np.float64)
    tree = make_tree_arrays(max_nodes)
    nbody_step(p, v, m, .1, 1., eps2_arr, .5, *tree, max_nodes)
    print(f'fertig ({time.perf_counter()-t0:.1f}s)')
