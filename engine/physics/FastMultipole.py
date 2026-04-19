"""
FastMultipole.py  –  FMM (Fast Multipole Method) Gravitations-Löser
════════════════════════════════════════════════════════════════════════════════
Algorithmus  (auf BH-Octree aufgebaut, Ordnung p=2)
───────────────────────────────────────────────────
  Jede Baumzelle speichert einen Multipol-Vektor mit 10 Koeffizienten:
    [0]      Monopol:    M        = Σ m_i
    [1-3]    Dipol:      q_α      = Σ m_i (r_iα - R_α)      (≈0 am Schwerpunkt)
    [4-9]    Quadrupol:  Q_αβ     = Σ m_i Δr_α Δr_β         (6 sym. Terme)

  Pro Frame:
    1. Octree bauen        O(N log N)
    2. Bottom-Up P2M+M2M   O(N)  – Multipol-Koeffizienten für jeden Knoten
    3. Kraft-Traversierung O(N log N) mit prange – BH-Kriterium, aber
       bei Näherung: Monopol + Dipol + Quadrupol-Beschleunigung statt Monopol-only

  Vorteil gegenüber BH: gleiche θ → ~2–3× genauer; gleiche Genauigkeit → größeres θ möglich
  Komplexität: identisch O(N log N), aber kleinerer Fehler-Vorfaktor

  Tastatur: B  →  zyklischer Backend-Wechsel BH / FMM / PM / P3M
════════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from numba import njit, prange  # type: ignore

_INS_STACK  = 256
_FORC_STACK = 512
_N_COEFF    = 10   # Monopol(1) + Dipol(3) + Quadrupol(6)

# ── Koeffizienten-Index-Aliasse ─────────────────────────────────────────────
_I_M   = 0
_I_QX  = 1;  _I_QY  = 2;  _I_QZ  = 3
_I_QXX = 4;  _I_QXY = 5;  _I_QXZ = 6
_I_QYY = 7;  _I_QYZ = 8;  _I_QZZ = 9


# ─────────────────────────────────────────────────────────────────────────────
#  Octree-Aufbau
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _make_child(c, parent, bits, m0, m1, m2,
                t_min, t_max, t_mass, t_com, t_children, t_particle):
    t_mass[c]     = 0.
    t_com[c, 0]   = 0.; t_com[c, 1] = 0.; t_com[c, 2] = 0.
    t_particle[c] = -1
    for k in range(8):
        t_children[c, k] = -1
    t_min[c, 0] = m0 if (bits & 1) else t_min[parent, 0]
    t_max[c, 0] = t_max[parent, 0] if (bits & 1) else m0
    t_min[c, 1] = m1 if (bits & 2) else t_min[parent, 1]
    t_max[c, 1] = t_max[parent, 1] if (bits & 2) else m1
    t_min[c, 2] = m2 if (bits & 4) else t_min[parent, 2]
    t_max[c, 2] = t_max[parent, 2] if (bits & 4) else m2


@njit(cache=True)
def _build_tree(pos, mass, t_mass, t_com, t_min, t_max,
                t_children, t_particle, n_count, max_nodes):
    N = pos.shape[0]
    mn0 = mn1 = mn2 =  1e30
    mx0 = mx1 = mx2 = -1e30
    for i in range(N):
        if mass[i] <= 0.: continue
        x = pos[i, 0]; y = pos[i, 1]; z = pos[i, 2]
        if x < mn0: mn0 = x
        if y < mn1: mn1 = y
        if z < mn2: mn2 = z
        if x > mx0: mx0 = x
        if y > mx1: mx1 = y
        if z > mx2: mx2 = z

    pad = max(mx0-mn0, mx1-mn1, mx2-mn2) * 0.012 + 2.

    n_prev = int(n_count[0])
    for j in range(n_prev):
        t_mass[j]     = 0.
        t_particle[j] = -1
        for k in range(8): t_children[j, k] = -1

    n_count[0] = 1
    t_min[0, 0]=mn0-pad; t_min[0, 1]=mn1-pad; t_min[0, 2]=mn2-pad
    t_max[0, 0]=mx0+pad; t_max[0, 1]=mx1+pad; t_max[0, 2]=mx2+pad
    t_com[0, 0]=0.; t_com[0, 1]=0.; t_com[0, 2]=0.

    stk_n = np.empty(_INS_STACK, dtype=np.int32)
    stk_p = np.empty(_INS_STACK, dtype=np.int32)

    for i in range(N):
        if mass[i] <= 0.: continue
        top = 0; stk_n[0] = 0; stk_p[0] = i
        while top >= 0:
            nd = stk_n[top]; pi = stk_p[top]; top -= 1
            px = pos[pi, 0]; py = pos[pi, 1]; pz = pos[pi, 2]; mp = mass[pi]
            if t_mass[nd] == 0.:
                t_mass[nd]=mp; t_com[nd,0]=px; t_com[nd,1]=py; t_com[nd,2]=pz
                t_particle[nd]=pi; continue
            tm = t_mass[nd]+mp
            t_com[nd,0]=(t_com[nd,0]*t_mass[nd]+px*mp)/tm
            t_com[nd,1]=(t_com[nd,1]*t_mass[nd]+py*mp)/tm
            t_com[nd,2]=(t_com[nd,2]*t_mass[nd]+pz*mp)/tm
            t_mass[nd]=tm
            m0=(t_min[nd,0]+t_max[nd,0])*.5
            m1=(t_min[nd,1]+t_max[nd,1])*.5
            m2=(t_min[nd,2]+t_max[nd,2])*.5
            if t_particle[nd] >= 0:
                op=t_particle[nd]; t_particle[nd]=-1
                ob=((1 if pos[op,0]>=m0 else 0)|(2 if pos[op,1]>=m1 else 0)|(4 if pos[op,2]>=m2 else 0))
                if t_children[nd,ob]<0:
                    c=int(n_count[0])
                    if c<max_nodes:
                        n_count[0]+=1
                        _make_child(c,nd,ob,m0,m1,m2,t_min,t_max,t_mass,t_com,t_children,t_particle)
                        t_children[nd,ob]=c
                if t_children[nd,ob]>=0 and top<_INS_STACK-2:
                    top+=1; stk_n[top]=t_children[nd,ob]; stk_p[top]=op
            nb=((1 if px>=m0 else 0)|(2 if py>=m1 else 0)|(4 if pz>=m2 else 0))
            if t_children[nd,nb]<0:
                c=int(n_count[0])
                if c<max_nodes:
                    n_count[0]+=1
                    _make_child(c,nd,nb,m0,m1,m2,t_min,t_max,t_mass,t_com,t_children,t_particle)
                    t_children[nd,nb]=c
            if t_children[nd,nb]>=0 and top<_INS_STACK-1:
                top+=1; stk_n[top]=t_children[nd,nb]; stk_p[top]=pi


# ─────────────────────────────────────────────────────────────────────────────
#  P2M + M2M  (Bottom-Up Multipol-Aufbau)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _build_multipoles(n_nodes, pos, mass,
                      t_mass, t_com, t_children, t_particle, multipoles):
    """
    Baut Multipol-Koeffizienten bottom-up auf.

    Blattknoten (t_particle >= 0):
      P2M: M = m_i, q = 0, Q = 0  (Expansion genau am Partikel)

    Innere Knoten:
      M2M-Shift der Kinder-Multipole zum eigenen Schwerpunkt:
        M  = sum M_c
        q  = sum (q_c + M_c*d)
        Q  = sum (Q_c + q_c*d + d*q_c + M_c*d*d)
    """
    for nd in range(n_nodes-1, -1, -1):
        if t_mass[nd] <= 0.:
            multipoles[nd, _I_M] = 0.
            continue

        cx = t_com[nd, 0]; cy = t_com[nd, 1]; cz = t_com[nd, 2]

        if t_particle[nd] >= 0:
            # Blatt: P2M
            multipoles[nd, _I_M]   = t_mass[nd]
            multipoles[nd, _I_QX]  = 0.; multipoles[nd, _I_QY]  = 0.; multipoles[nd, _I_QZ]  = 0.
            multipoles[nd, _I_QXX] = 0.; multipoles[nd, _I_QXY] = 0.; multipoles[nd, _I_QXZ] = 0.
            multipoles[nd, _I_QYY] = 0.; multipoles[nd, _I_QYZ] = 0.; multipoles[nd, _I_QZZ] = 0.
        else:
            # Innerer Knoten: M2M
            Mt = qx = qy = qz = 0.
            Qxx = Qxy = Qxz = Qyy = Qyz = Qzz = 0.

            for k in range(8):
                ch = t_children[nd, k]
                if ch < 0 or t_mass[ch] <= 0.: continue

                dx = t_com[ch, 0] - cx
                dy = t_com[ch, 1] - cy
                dz = t_com[ch, 2] - cz

                Mc   = multipoles[ch, _I_M]
                qxc  = multipoles[ch, _I_QX];  qyc  = multipoles[ch, _I_QY];  qzc  = multipoles[ch, _I_QZ]
                Qxxc = multipoles[ch, _I_QXX]; Qxyc = multipoles[ch, _I_QXY]; Qxzc = multipoles[ch, _I_QXZ]
                Qyyc = multipoles[ch, _I_QYY]; Qyzc = multipoles[ch, _I_QYZ]; Qzzc = multipoles[ch, _I_QZZ]

                Mt  += Mc
                qx  += qxc + Mc*dx
                qy  += qyc + Mc*dy
                qz  += qzc + Mc*dz
                Qxx += Qxxc + 2.*qxc*dx + Mc*dx*dx
                Qxy += Qxyc + qxc*dy + qyc*dx + Mc*dx*dy
                Qxz += Qxzc + qxc*dz + qzc*dx + Mc*dx*dz
                Qyy += Qyyc + 2.*qyc*dy + Mc*dy*dy
                Qyz += Qyzc + qyc*dz + qzc*dy + Mc*dy*dz
                Qzz += Qzzc + 2.*qzc*dz + Mc*dz*dz

            multipoles[nd, _I_M]   = Mt
            multipoles[nd, _I_QX]  = qx;  multipoles[nd, _I_QY]  = qy;  multipoles[nd, _I_QZ]  = qz
            multipoles[nd, _I_QXX] = Qxx; multipoles[nd, _I_QXY] = Qxy; multipoles[nd, _I_QXZ] = Qxz
            multipoles[nd, _I_QYY] = Qyy; multipoles[nd, _I_QYZ] = Qyz; multipoles[nd, _I_QZZ] = Qzz


# ─────────────────────────────────────────────────────────────────────────────
#  Quadrupol-Beschleunigung
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True, inline='always', fastmath=True)
def _multipole_accel_on(px, py, pz, nd, t_com, multipoles, G, eps2):
    """Beschleunigung am Punkt (px,py,pz) durch Multipol-Entwicklung von nd."""
    cx = t_com[nd, 0]; cy = t_com[nd, 1]; cz = t_com[nd, 2]
    drx = cx-px;  dry = cy-py;  drz = cz-pz
    r2  = drx*drx + dry*dry + drz*drz + eps2
    if r2 < 1e-20: return 0., 0., 0.

    r_s  = r2 ** 0.5
    inv3 = G / (r2 * r_s)
    inv5 = inv3 / r2
    inv7 = inv5 / r2

    Mt  = multipoles[nd, _I_M]
    qx  = multipoles[nd, _I_QX];  qy  = multipoles[nd, _I_QY];  qz  = multipoles[nd, _I_QZ]
    Qxx = multipoles[nd, _I_QXX]; Qxy = multipoles[nd, _I_QXY]; Qxz = multipoles[nd, _I_QXZ]
    Qyy = multipoles[nd, _I_QYY]; Qyz = multipoles[nd, _I_QYZ]; Qzz = multipoles[nd, _I_QZZ]

    # Monopol
    ax = inv3 * Mt * drx
    ay = inv3 * Mt * dry
    az = inv3 * Mt * drz

    # Dipol
    qdotr = qx*drx + qy*dry + qz*drz
    ax += inv5 * (3.*qdotr*drx - qx*r2)
    ay += inv5 * (3.*qdotr*dry - qy*r2)
    az += inv5 * (3.*qdotr*drz - qz*r2)

    # Quadrupol
    Qrx = Qxx*drx + Qxy*dry + Qxz*drz
    Qry = Qxy*drx + Qyy*dry + Qyz*drz
    Qrz = Qxz*drx + Qyz*dry + Qzz*drz
    rQr = drx*Qrx + dry*Qry + drz*Qrz

    fac  = 3.0 * inv5
    fac2 = 5.0 * rQr * inv7
    ax += fac*Qrx - fac2*drx
    ay += fac*Qry - fac2*dry
    az += fac*Qrz - fac2*drz

    return ax, ay, az


# ─────────────────────────────────────────────────────────────────────────────
#  Kraft-Traversierung
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def _fmm_forces(pos, mass, eps2_arr,
                t_mass, t_com, t_min, t_max, t_children, t_particle,
                multipoles, G, theta2):
    N = pos.shape[0]
    accel = np.zeros((N, 3), dtype=np.float64)

    for i in prange(N):
        if mass[i] <= 0.: continue

        px = pos[i, 0];  py = pos[i, 1];  pz = pos[i, 2]
        eps2 = eps2_arr[i]
        ax = ay = az = 0.

        stk = np.empty(_FORC_STACK, dtype=np.int32)
        top = 0;  stk[0] = 0

        while top >= 0:
            nd = stk[top];  top -= 1
            if t_mass[nd] == 0.: continue

            drx = t_com[nd,0]-px;  dry = t_com[nd,1]-py;  drz = t_com[nd,2]-pz
            d2  = drx*drx + dry*dry + drz*drz

            if d2 < 1e-12:
                for k in range(8):
                    ch = t_children[nd, k]
                    if ch >= 0 and top < _FORC_STACK-1:
                        top += 1;  stk[top] = ch
                continue

            s    = t_max[nd,0] - t_min[nd,0]
            leaf = True
            for k in range(8):
                if t_children[nd, k] >= 0:
                    leaf = False;  break

            if leaf:
                j = t_particle[nd]
                if j >= 0 and j != i and mass[j] > 0.:
                    eff_e2 = eps2 if eps2 >= eps2_arr[j] else eps2_arr[j]
                    d2j    = d2 + eff_e2
                    inv3   = G * mass[j] / (d2j * d2j**0.5)
                    ax += inv3*drx;  ay += inv3*dry;  az += inv3*drz
            elif s*s/d2 < theta2:
                dax, day, daz = _multipole_accel_on(px, py, pz, nd,
                                                     t_com, multipoles, G, eps2)
                ax += dax;  ay += day;  az += daz
            else:
                for k in range(8):
                    ch = t_children[nd, k]
                    if ch >= 0 and top < _FORC_STACK-1:
                        top += 1;  stk[top] = ch

        accel[i, 0] = ax;  accel[i, 1] = ay;  accel[i, 2] = az

    return accel


# ─────────────────────────────────────────────────────────────────────────────
#  Haupt-Simulations-Schritt
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, fastmath=True)
def fmm_step(pos, vel, mass, dt, G, eps2_arr, theta,
             t_mass, t_com, t_min, t_max, t_children, t_particle,
             n_count, multipoles, max_nodes):
    """
    Vollständiger FMM Leapfrog-Schritt.
    Signatur analog zu nbody_step (BarnesHutNumba).
    """
    _build_tree(pos, mass, t_mass, t_com, t_min, t_max,
                t_children, t_particle, n_count, max_nodes)

    n_nodes = int(n_count[0])
    _build_multipoles(n_nodes, pos, mass,
                      t_mass, t_com, t_children, t_particle, multipoles)

    theta2 = theta * theta
    accel  = _fmm_forces(pos, mass, eps2_arr,
                          t_mass, t_com, t_min, t_max, t_children, t_particle,
                          multipoles, G, theta2)

    N    = pos.shape[0]
    npos = np.empty_like(pos)
    nvel = np.empty_like(vel)
    for i in prange(N):
        if mass[i] <= 0.:
            npos[i,0]=pos[i,0]; npos[i,1]=pos[i,1]; npos[i,2]=pos[i,2]
            nvel[i,0]=vel[i,0]; nvel[i,1]=vel[i,1]; nvel[i,2]=vel[i,2]
            continue
        nvel[i,0] = vel[i,0] + accel[i,0]*dt
        nvel[i,1] = vel[i,1] + accel[i,1]*dt
        nvel[i,2] = vel[i,2] + accel[i,2]*dt
        npos[i,0] = pos[i,0] + nvel[i,0]*dt
        npos[i,1] = pos[i,1] + nvel[i,1]*dt
        npos[i,2] = pos[i,2] + nvel[i,2]*dt
    return npos, nvel


# ─────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def make_fmm_arrays(max_nodes):
    """Pre-allokiert alle FMM-Arrays."""
    return (
        np.zeros(max_nodes,              dtype=np.float64),   # t_mass
        np.zeros((max_nodes, 3),         dtype=np.float64),   # t_com
        np.zeros((max_nodes, 3),         dtype=np.float64),   # t_min
        np.zeros((max_nodes, 3),         dtype=np.float64),   # t_max
        np.full( (max_nodes, 8), -1,     dtype=np.int32),     # t_children
        np.full( max_nodes, -1,          dtype=np.int32),     # t_particle
        np.zeros(1,                      dtype=np.int64),     # n_count
        np.zeros((max_nodes, _N_COEFF),  dtype=np.float64),   # multipoles
    )


def warmup_fmm(N=256, max_nodes=4096):
    """JIT-Kompilierung vorab."""
    import time
    print('[FMM] Kompiliere JIT-Kernel... ', end='', flush=True)
    t0  = time.perf_counter()
    rng = np.random.default_rng(0)
    p_w = rng.uniform(-1., 1., (N, 3))
    v_w = np.zeros((N, 3))
    m_w = np.ones(N) * 0.1
    e2_w= np.full(N, 1.69)
    arrs= make_fmm_arrays(max_nodes)
    fmm_step(p_w, v_w, m_w, 0.01, 1.0, e2_w, 0.7, *arrs, max_nodes)
    print(f'{time.perf_counter()-t0:.1f}s')

