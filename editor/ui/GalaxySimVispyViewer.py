"""
GalaxySimVispyViewer.py  –  Finales Echtzeit-Rendering
════════════════════════════════════════════════════════════════════════════════
Architektur
───────────
  Physik-Backend  (automatisch gewählt):
    N ≤ 2000 : NumPy O(N²)  – vollvektorisiert, kein Overhead
    N > 2000 : Numba JIT Barnes-Hut  – O(N log N), prange-parallel

  Rendering:
    • Ein Scatter für alle Disk-Sterne: per-Partikel Farbe + Größe
      (vorberechnet, konstant – GPU-Upload nur Positionen effektiv)
    • Separater SMBH-Shadow-Scatter: dunkle Marker, Größe ∝ √M_BH
    • Separater Photonen-Ring-Glow: leuchtend orange, größer als Shadow
    • float32-Buffer für GPU-Upload (halber Speicher vs. float64)

  Merger-Integration:
    • BH–BH + BH–Stern alle 5 Physik-Schritte
    • Inaktive Partikel (mass=0) bei pos=(1e7,1e7,1e7) → GPU clippt sie

  Statistiken (alle 30 Frames):
    • FPS, ms/Frame, aktive Partikel
    • Zerlegungsrate [%] – Anteil ungebundener Partikel
    • Systemenergie

  Tastatur:
    SPACE   – Pause / Weiter
    +  /  - – Zeitschritt dt erhöhen / verringern (×1.4 / ÷1.4)
    R       – dt auf Startwert zurücksetzen
    T       – Statistik in Konsole ausgeben
    M       – Farbmodus umschalten (Standard / Metallizität)
════════════════════════════════════════════════════════════════════════════════
"""
import time
import threading
import os
import numpy as np
from vispy import scene, app  # type: ignore[import-untyped]

# ── Numba Thread-Budget: maximal halbe Kernzahl für Physik ─────────────────
# Lässt dem OS, dem Render-Thread und vispy die andere Hälfte.
import numba   # type: ignore[import-untyped]
_PHYS_CORES = max(2, (os.cpu_count() or 4) // 2)
numba.set_num_threads(_PHYS_CORES)

from engine.physics.BarnesHutNumba import nbody_step, make_tree_arrays, warmup as bh_warmup
from engine.physics.FastMultipole  import fmm_step, make_fmm_arrays, warmup_fmm
from engine.physics.ParticleMesh   import ParticleMeshSolver, GPU_AVAILABLE
from engine.physics.MergerKernel   import apply_mergers, apply_agn_jets, compute_stats
from engine.physics.DarkMatterHalo  import NFWHalo, DarkMatterSystem
from engine.physics.StellarFeedback import StellarFeedback

_NUMPY_THRESH  = 2000     # Schwelle für Backend-Wahl
_MERGE_BH_R    = 6.0      # BH–BH Verschmelzungsradius
_ACCRETE_R     = 3.0      # BH–Stern Akkretionsradius
_MERGER_EVERY  = 5        # Merger-Check alle N Physik-Schritte
_STATS_EVERY   = 30       # Statistik-Refresh alle N Frames
_JET_THRESHOLD = 18.      # Mindest-Akkretionsmasse für AGN-Jet-Auslösung
_FB_EVERY      = 5        # StellarFeedback alle N Physik-Schritte (spart ~4/5 Overhead)
_DM_EVERY      = 3        # DM-Beschleunigung alle N Physik-Schritte (Halos ändern sich langsam)
_BH_VIS_EVERY  = 10       # SMBH-Visuals alle N Frames (visuell nicht wahrnehmbar)

# Physik-Backend-Bezeichner
_BACKEND_BH    = 'BH'     # Barnes-Hut (Standard, O(N log N))
_BACKEND_FMM   = 'FMM'   # Fast Multipole Method (O(N), Quadrupol)
_BACKEND_PM    = 'PM'    # Particle Mesh (O(N + M³ log M), Fernfeld)
_BACKEND_P3M   = 'P3M'  # Particle-Particle + Particle-Mesh (PM Fernfeld + BH Nahfeld)
_BACKEND_PY    = 'PY'   # Pure-Python BH (Fallback für N < 300 / Diagnose)


class GalaxySimVispyViewer:
    """Echtzeit-Visualisierung einer N-Body-Galaxien-Simulation."""

    def __init__(self, galaxies, dm_halo_configs=None,
                 dt            = 0.7,
                 G             = 1.0,
                 eps           = 1.3,
                 theta         = 0.65,
                 steps_per_frame = 1):

        self.dt    = dt
        self._dt0  = dt          # Reset-Referenz
        self.G     = G
        self._eps_default = eps  # gespeichert für Fallback
        self.theta = theta
        self.spf   = steps_per_frame
        self.paused   = False
        self.frame_n  = 0
        self._stats   = (0., 0., 0)

        # ── Partikel-Arrays ──────────────────────────────────────────────
        all_stars = [s for gal in galaxies for s in gal]
        N = len(all_stars)
        self.N    = N
        self.pos  = np.array([s['position']         for s in all_stars], dtype=np.float64)
        self.vel  = np.array([s['velocity']         for s in all_stars], dtype=np.float64)
        self.mass = np.array([s['mass']             for s in all_stars], dtype=np.float64)
        self.bh_m = np.array([s.get('is_bh', False) for s in all_stars], dtype=bool)
        self.bh_idx = np.where(self.bh_m)[0].tolist()

        # Render-Eigenschaften (konstant über die gesamte Simulation)
        self.r_col = np.array(
            [s.get('render_color', (.7,.6,.3,.8)) for s in all_stars],
            dtype=np.float32)
        self._r_col_default = self.r_col.copy()   # Backup für Farbmodus-Reset
        self.r_sz  = np.array(
            [s.get('render_size', 0.08) for s in all_stars],
            dtype=np.float32)

        # Disk-Maske: alles außer BHs (BHs haben render_size~0.1 → unsichtbar
        #             in Star-Scatter, dafür eigener BH-Scatter)
        self._disk     = ~self.bh_m
        self._disk_idx = np.where(self._disk)[0]  # Integer-Indizes: ~4× schneller als Bool-Masken
        N_disk = len(self._disk_idx)

        # GPU-Cache: konstante Arrays für Disk-Sterne einmalig aufbauen
        # → kein boolean-Indexing mehr pro Frame
        self._disk_col = self.r_col[self._disk_idx].copy()   # (N_disk, 4) float32
        self._disk_sz  = self.r_sz [self._disk_idx].copy()   # (N_disk,)   float32
        self._disk_p32 = np.empty((N_disk, 3), dtype=np.float32)  # Pos-Buffer (wiederverwendet)
        self._disk_col_dirty = True   # True = Farben müssen erneut an GPU gesendet werden

        # Per-Partikel Softening:  SMBH stark (eps_bh), Sterne leicht (eps_star)
        eps_bh   = 6.0    # Starkes Softening für Schwarze Löcher → verhindert Singularitäten bei Passage
        eps_star = 1.2    # Leichtes Softening für Ring-/Scheiben-Partikel → erhält Orbitalstruktur
        _soft = np.where(self.bh_m, eps_bh, eps_star)
        # Lese softening-Attribut aus Preset, falls vorhanden (überschreibt Standard)
        for k, s in enumerate(all_stars):
            if 'softening' in s:
                _soft[k] = float(s['softening'])
        self.eps2_arr = (_soft * _soft).astype(np.float64)
        self.eps2     = float(eps_star * eps_star)  # Fallback für NumPy-Pfad

        # float32-Buffer für GPU-Upload (halber Transfer-Overhead)
        self._p32  = np.empty((N, 3), dtype=np.float32)

        # ── NFW-Dunkle-Materie-Halos ─────────────────────────────────────
        if dm_halo_configs:
            halo_list = []
            for k, cfg in enumerate(dm_halo_configs):
                # Initiales Halo-Zentrum = BH-Position (falls vorhanden)
                c0 = np.array(cfg.get('center', [0., 0., 0.]), dtype=np.float64)
                if k < len(self.bh_idx):
                    c0 = self.pos[self.bh_idx[k]].copy()
                halo_list.append(NFWHalo(c0, cfg['M_vir'], cfg['r_s'], cfg['c'], G))
            self._dm = DarkMatterSystem(halo_list)
            print(f'[DM] {len(halo_list)} NFW-Halos  M_vir={dm_halo_configs[0]["M_vir"]:.0f}  '
                  f'r_s={dm_halo_configs[0]["r_s"]:.0f}')
        else:
            self._dm = None

        # ── Stellares Feedback & Metallizitäts-Tracking ──────────────────
        self._feedback    = StellarFeedback(N, self.mass, self.bh_m, seed=42)
        self._total_sn    = 0
        self._color_mode  = 'default'    # 'default' oder 'metallicity'
        self._met_col_age = -1           # Frame der letzten Metallizitäts-Farb-Aktualisierung
        self._fb_step_n   = 0            # interner Schrittzähler für Feedback-Throttling
        print(f'[Feedback] SN-Feedback + Metallizitäts-Tracking aktiv  (r_fb={self._feedback.r_fb:.0f})')

        # ── Diagnostik ───────────────────────────────────────────────────
        self._stats_agn   = 0   # AGN-Jet-Kicks gesamt
        self._dm_step_n   = 0   # DM-Throttling-Zähler
        self._dm_acc_last = None  # Gecachte DM-Beschleunigung

        self.use_bh = N > _NUMPY_THRESH
        if self.use_bh:
            self._max_nodes  = int(N * 5)
            self._tree       = make_tree_arrays(self._max_nodes)
            self._fmm_arrays = make_fmm_arrays(self._max_nodes)

            # GPU verfügbar: größeres Gitter + PM als Standard-Backend
            _pm_grid = 256 if GPU_AVAILABLE else 128
            self._pm = ParticleMeshSolver(
                N_grid=_pm_grid, box_size=3000., G=G, eps=eps)

            # Standard-Backend: GPU-PM wenn verfügbar, sonst Barnes-Hut (CPU)
            if GPU_AVAILABLE:
                self.backend = _BACKEND_PM
                print('[GPU] CuPy-Beschleunigung erkannt – PM-Backend (256³) gewählt')
            else:
                self.backend = _BACKEND_BH

            # Alle Backends vorab JIT-kompilieren
            bh_warmup()
            warmup_fmm()

            print('[Viewer] JIT-Spezialisierung auf volle Partikelzahl...')
            t0 = time.perf_counter()
            self.pos, self.vel = nbody_step(
                self.pos, self.vel, self.mass,
                self.dt, self.G, self.eps2_arr, self.theta,
                *self._tree, self._max_nodes)
            print(f'[Viewer] Bereit  ({time.perf_counter()-t0:.1f}s)')
        else:
            self.backend = 'O(N²)'
            self._pm     = None
            # Leapfrog: halber Halbschritt für korrekten Startpunkt
            self.vel += 0.5 * self._accel_np() * self.dt

        # ── vispy Canvas ─────────────────────────────────────────────────
        self.canvas = scene.SceneCanvas(
            keys='interactive', show=True, bgcolor='black',
            title=f'N-Body  {N} Partikel  –  SPACE  +/-  R  T  M  B')
        self.canvas.events.key_press.connect(self._on_key)  # type: ignore[attr-defined]

        vp = self.canvas.central_widget.add_view()
        self.view = vp
        # Kamera-Distanz: automatisch aus dem Systemumfang berechnen
        _sys_span   = float(np.linalg.norm(self.pos[self.mass > 0.], axis=1).max()) if N > 0 else 500.
        _cam_dist   = max(600., _sys_span / np.tan(np.deg2rad(24)))   # 48° effektiv FOV
        self.view.camera = scene.cameras.TurntableCamera(
            fov=52, distance=_cam_dist, elevation=28, azimuth=35)

        # ── Scatter 1: alle Disk-Sterne ──────────────────────────────────
        self._upd32()
        self.sc_disk = scene.visuals.Markers()  # type: ignore[attr-defined]
        self.sc_disk.set_data(
            self._disk_p32,
            face_color=self._disk_col,
            size=self._disk_sz,
            edge_width=0)
        self._disk_col_dirty = False
        self.view.add(self.sc_disk)

        # ── Scatter 2: SMBH-Schatten (dunkle, große Marker) ─────────────
        self.sc_shadow = scene.visuals.Markers()  # type: ignore[attr-defined]
        self.view.add(self.sc_shadow)

        # ── Scatter 3: Photonen-Ring-Glow (leuchtend, noch größer) ──────
        self.sc_glow = scene.visuals.Markers()  # type: ignore[attr-defined]
        self.view.add(self.sc_glow)

        self._update_bh_visuals()

        # ── Physik-Daemon-Thread ──────────────────────────────────────────
        # Physik läuft vollständig entkoppelt vom Render-Timer.
        # Doppelbuffer: _phys_disk_p32 wird vom Physik-Thread beschrieben,
        # _render_disk_p32 wird vom Render-Thread gelesen (unter Lock getauscht).
        # _render_disk_vel32 ermöglicht lineare Positions-Interpolation zwischen
        # Physik-Frames für weiches 60fps-Rendering unabhängig von der Physikrate.
        N_disk = len(self._disk_idx)
        self._phys_disk_p32    = np.empty((N_disk, 3), dtype=np.float32)
        self._render_disk_p32  = self._disk_p32.copy()
        self._phys_disk_vel32  = np.zeros((N_disk, 3), dtype=np.float32)
        self._render_disk_vel32 = np.zeros((N_disk, 3), dtype=np.float32)
        self._disk_vel_render  = np.zeros((N_disk, 3), dtype=np.float32)
        self._disk_p32_interp  = self._disk_p32.copy()   # interpolierte Positionen für Render
        self._vel_tmp          = np.empty((N_disk, 3), dtype=np.float32)  # Scratch für vel*dt
        self._snap_wall_time   = 0.0                       # Wanduhr-Zeit des letzten Snapshots
        self._rlock      = threading.Lock()
        self._snap_ready = False          # neues Snapshot verfügbar?
        self._rsnap_bh   = None           # BH-Snapshot für Render-Thread
        self._rsnap_col  = self._disk_col.copy()   # Farb-Snapshot
        self._rsnap_dirty = False          # Farben haben sich geändert
        self._rsnap_ms   = 0.0             # letzte Physik-Zeit [ms]

        # Render-seitiger Zustand (überlebt zwischen Snapshots für Interpolation)
        self._last_snap_time = 0.0         # Wanduhr-Zeitpunkt des letzten Snapshots
        self._last_bh_snap   = None        # letztes BH-Snapshot
        self._last_phys_ms   = 0.0         # letzte Physik-Schrittzeit [ms]

        # BH-Farb-Arrays vorab allozieren (max. so viele BHs wie es gibt)
        _max_bh = max(len(self.bh_idx), 1)
        self._bh_col_s = np.tile([0.02, 0.02, 0.04, 1.00], (_max_bh, 1)).astype(np.float32)
        self._bh_col_g = np.tile([1.00, 0.68, 0.12, 0.18], (_max_bh, 1)).astype(np.float32)

        self._phys_running = True
        self._phys_thread  = threading.Thread(
            target=self._phys_loop, name='phys', daemon=True)
        self._phys_thread.start()

        self.timer = app.Timer(interval=0.016, connect=self._on_timer, start=True)

    # ── Physik ───────────────────────────────────────────────────────────

    def _accel_np(self):
        """O(N²) vollvektorisiert – für N ≤ 2000."""
        dr   = self.pos[np.newaxis,:,:] - self.pos[:,np.newaxis,:]
        d2   = (dr*dr).sum(axis=2) + self.eps2
        inv3 = self.G * self.mass[np.newaxis,:] / (d2 * np.sqrt(d2))
        np.fill_diagonal(inv3, 0.)
        return (inv3[:,:,np.newaxis] * dr).sum(axis=1)

    def _step(self):
        # 1. N-Body Gravitation (gewähltes Backend)
        if self.use_bh:
            if self.backend == _BACKEND_BH:
                self.pos, self.vel = nbody_step(
                    self.pos, self.vel, self.mass,
                    self.dt, self.G, self.eps2_arr, self.theta,
                    *self._tree, self._max_nodes)

            elif self.backend == _BACKEND_FMM:
                self.pos, self.vel = fmm_step(
                    self.pos, self.vel, self.mass,
                    self.dt, self.G, self.eps2_arr, self.theta,
                    *self._fmm_arrays, self._max_nodes)

            elif self.backend == _BACKEND_PM:
                # PM-only: gut für großskalige Dynamik, aber schlechte Auflösung nahe BHs
                assert self._pm is not None
                self.pos, self.vel = self._pm.step(
                    self.pos, self.vel, self.mass, self.dt)

            elif self.backend == _BACKEND_P3M:
                # P3M: PM für Fernfeld + BH für Nahfeld-Korrektur
                # Fernfeld: PM-Beschleunigung berechnen
                assert self._pm is not None
                pm_accel = self._pm.compute_accel(self.pos, self.mass)
                # Nahfeld-Korrektur: BH Schritt durchführen, dann PM hinzuaddieren
                self.pos, self.vel = nbody_step(
                    self.pos, self.vel, self.mass,
                    self.dt, self.G, self.eps2_arr, self.theta,
                    *self._tree, self._max_nodes)
                # PM-Korrektur zusätzlich applizieren (Kic velocities additiv)
                active = self.mass > 0.
                self.vel[active] += pm_accel[active] * self.dt * 0.5   # halber Schritt (bereits integriert)

            elif self.backend == _BACKEND_PY:
                # Pure-Python BH Fallback (kein Numba JIT, gut für N < 300 / Debugging)
                from engine.physics.BarnesHut3D import BarnesHut3D as _BH3D
                acc = _BH3D.solve(
                    self.pos, self.mass,
                    G=self.G, eps=float(np.sqrt(self.eps2_arr.mean())), theta=self.theta,
                    workers=2,
                )
                active = self.mass > 0.
                self.pos[active] += self.vel[active] * self.dt + 0.5 * acc[active] * self.dt**2
                self.vel[active] += acc[active] * self.dt
        else:
            # Velocity Verlet (Leapfrog) O(N²)
            self.pos += self.vel * self.dt
            self.vel += self._accel_np() * self.dt

        # 2. Dunkle-Materie-Halo-Beschleunigung
        # Alle _DM_EVERY Schritte (Halos ändern sich langsam – spart ~66 % DM-Overhead)
        self._dm_step_n += 1
        if self._dm is not None:
            if self._dm_step_n % _DM_EVERY == 0:
                # Halo-Zentren auf aktive BH-Positionen aktualisieren
                active_bh = [i for i in self.bh_idx if self.mass[i] > 0.]
                if active_bh:
                    bh_arr = np.asarray(active_bh)
                    self._dm.update_centers(self.pos[bh_arr])
                self._dm_acc_last = self._dm.acceleration(self.pos)
            # Gecachte Beschleunigung immer anwenden (auch wenn nicht neu berechnet)
            if self._dm_acc_last is not None:
                active = self.mass > 0.
                self.vel[active] += self._dm_acc_last[active] * self.dt

        # 3. Stellares Feedback (Supernovae + AGB-Winde + Metallizität)
        # Nur alle _FB_EVERY Schritte ausführen – spart ~80 % Overhead
        self._fb_step_n += 1
        if self._fb_step_n % _FB_EVERY == 0:
            n_sn, _n_wind = self._feedback.step(self.pos, self.vel, self.mass)
            if n_sn > 0:
                self._total_sn += n_sn
                self._disk_col_dirty = True   # Metallizitäts-Farben neu berechnen

    # ── SMBH-Visualisierung ──────────────────────────────────────────────

    def _update_bh_visuals(self):
        """
        SMBH-Darstellung in zwei Ebenen:
          Shadow : dunkler Kern  –  Größe ∝ M_BH^0.55  (Schwarzschild: r_s = 2GM/c²)
          Glow   : oranger Photonenring  –  ~1.7× Shadow-Größe, halbtransparent
        """
        abh = [i for i in self.bh_idx if self.mass[i] > 0.]
        if not abh:
            return
        p32  = self._p32[abh].copy()
        # Größe skaliert mit BH-Masse (Schwarzschild-Radius ∝ M → visuelle Näherung)
        sz_s = np.array([max(10., min(0.018 * float(self.mass[i])**.55, 32.))
                         for i in abh], dtype=np.float32)
        sz_g = sz_s * 1.75
        col_s = np.tile([0.02, 0.02, 0.04, 1.00], (len(abh), 1)).astype(np.float32)
        col_g = np.tile([1.00, 0.68, 0.12, 0.18], (len(abh), 1)).astype(np.float32)
        self.sc_shadow.set_data(p32, face_color=col_s, size=sz_s, edge_width=0)
        self.sc_glow  .set_data(p32, face_color=col_g, size=sz_g, edge_width=0)

    # ── Physik-Daemon-Thread ────────────────────────────────────────────

    def _phys_loop(self):
        """Läuft in Daemon-Thread, entkoppelt von vispy-Render-Loop."""
        phys_frame = 0
        _MIN_SLEEP = 0.002   # 2ms Pause – gibt GIL frei für Render-Thread
        while self._phys_running:
            if self.paused:
                time.sleep(0.02)
                continue

            # Warte falls Render-Thread den letzten Snapshot noch nicht gelesen hat
            if self._snap_ready:
                time.sleep(0.001)
                continue

            try:
                t0 = time.perf_counter()
                for step in range(self.spf):
                    self._step()
                    step_n = phys_frame * self.spf + step
                    if step_n % _MERGER_EVERY == 0:
                        _nbhm, _nacc, acc_mass = apply_mergers(
                            self.pos, self.vel, self.mass,
                            self.bh_idx, _MERGE_BH_R, _ACCRETE_R)
                        if acc_mass:
                            n_jet = apply_agn_jets(
                                self.bh_idx, acc_mass,
                                self.pos, self.vel, self.mass,
                                jet_threshold=_JET_THRESHOLD, G=self.G)
                            self._stats_agn += n_jet
                ms = (time.perf_counter() - t0) * 1e3
            except Exception as exc:
                import traceback
                print(f'[PhysThread] FEHLER in Schritt {phys_frame}:\n{traceback.format_exc()}')
                time.sleep(0.1)
                phys_frame += 1
                continue

            # Positionen → Physik-Buffer (kein Lock nötig, nur Physik-Thread schreibt)
            self._upd32_phys()

            # Farben bei Metallizitäts-Modus vorbereiten
            col_dirty = False
            col_snap  = None
            if self._color_mode == 'metallicity':
                if self._disk_col_dirty or (phys_frame - self._met_col_age >= 5):
                    met_col = self._feedback.metallicity_colors()
                    np.copyto(self._disk_col, met_col[self._disk_idx], casting='unsafe')
                    self._met_col_age  = phys_frame
                    col_dirty = True
            if phys_frame % _STATS_EVERY == 0:
                self._stats = compute_stats(self.pos, self.vel, self.mass, self.G)

            # BH-Snapshot vorbereiten
            abh = [i for i in self.bh_idx if self.mass[i] > 0.]
            if abh:
                bh_snap = self._p32[abh].copy()
                sz_s = np.array([max(10., min(0.018 * float(self.mass[i])**.55, 32.))
                                 for i in abh], dtype=np.float32)
            else:
                bh_snap = None
                sz_s    = None

            # Snapshot an Render-Thread übergeben (sehr kurze Lock-Zeit)
            with self._rlock:
                np.copyto(self._render_disk_p32, self._phys_disk_p32)
                np.copyto(self._render_disk_vel32, self._phys_disk_vel32)
                self._snap_wall_time = time.perf_counter()
                self._rsnap_bh    = (bh_snap, sz_s) if bh_snap is not None else None
                if col_dirty:
                    self._rsnap_col   = self._disk_col.copy()
                    self._rsnap_dirty = True
                self._snap_ready  = True
                self._rsnap_ms    = ms

            phys_frame += 1
            self._disk_col_dirty = False

            # Obligatorische Pause – gibt OS + Render-Thread Luft
            time.sleep(_MIN_SLEEP)

    def _upd32_phys(self):
        """float64 → float32, schreibt in Physik-seitigen Buffer (kein Lock)."""
        np.copyto(self._p32, self.pos, casting='unsafe')
        np.take(self._p32, self._disk_idx, axis=0, out=self._phys_disk_p32)
        # Geschwindigkeiten für Interpolation mitschreiben
        self._phys_disk_vel32[:] = self.vel[self._disk_idx]

    # ── Timer-Callback (nur Rendering, keine Physik) ─────────────────────

    def _on_timer(self, event):
        # ── Neuen Snapshot abholen (falls vorhanden) ─────────────────────
        new_snap = False
        if self._snap_ready:
            new_snap = True
            with self._rlock:
                np.copyto(self._disk_p32, self._render_disk_p32)
                np.copyto(self._disk_vel_render, self._render_disk_vel32)
                self._last_snap_time    = self._snap_wall_time
                self._last_bh_snap      = self._rsnap_bh
                if self._rsnap_dirty:
                    np.copyto(self._disk_col, self._rsnap_col)
                    self._rsnap_dirty = False
                self._last_phys_ms = self._rsnap_ms
                self._snap_ready   = False

        # Noch kein einziger Snapshot angekommen → nichts rendern
        if self._last_snap_time == 0.0:
            return

        # ── Interpolation (zero-alloc, korrekte Sim-Einheiten) ───────────
        # vel ist in sim_units/sim_time, nicht sim_units/real_seconds.
        # Korrekte Formel: pos_interp = pos_snap + vel * dt_sim * fraction
        # wobei fraction = elapsed_real / step_duration_real ∈ [0, 1]
        step_s   = self._last_phys_ms * 1e-3        # Schrittdauer in Sekunden
        elapsed  = time.perf_counter() - self._last_snap_time
        fraction = float(np.clip(elapsed / step_s, 0., 1.)) if step_s > 1e-6 else 0.
        # vel * (dt_sim * fraction) → _vel_tmp  (kein temporäres Array)
        np.multiply(self._disk_vel_render, self.dt * fraction,
                    out=self._vel_tmp)
        np.add(self._disk_p32, self._vel_tmp,
               out=self._disk_p32_interp, casting='unsafe')

        # Disk-Scatter: immer Farbe + Größe mitschicken
        # (vispy setzt bei set_data() ohne face_color den Default 'white' → weiße Partikel)
        self.sc_disk.set_data(
            self._disk_p32_interp,
            face_color=self._disk_col,
            size=self._disk_sz,
            edge_width=0)

        # SMBH-Visuals (nur bei neuem Snapshot)
        if new_snap and self._last_bh_snap is not None:
            bh_pos, sz_s = self._last_bh_snap
            assert sz_s is not None
            sz_g = sz_s * 1.75
            self.sc_shadow.set_data(bh_pos, face_color=self._bh_col_s[:len(sz_s)],
                                    size=sz_s, edge_width=0)
            self.sc_glow  .set_data(bh_pos, face_color=self._bh_col_g[:len(sz_s)],
                                    size=sz_g, edge_width=0)

        # Titelzeile (nur bei neuem Snapshot)
        if new_snap:
            fps = (1e3 / self._last_phys_ms) if self._last_phys_ms > 0. else 0.
            frag, Esys, nact = self._stats
            dm_str   = '★DM' if self._dm else ''
            mode_str = '[Z]' if self._color_mode == 'metallicity' else ''
            self.canvas.title = (
                f'N-Body {self.N}  akt:{nact} {dm_str}{mode_str}  |  '
                f'{fps:.0f} phys/s  {self._last_phys_ms:.0f}ms  [{self.backend}]  |  '
                f'SN:{self._total_sn}  '
                f'Zerlg:{frag*100:.1f}%  |  '
                f'dt={self.dt:.2f}   SPACE +/- R T M B'
            )

        self.canvas.update()
        self.frame_n += 1

    # ── Hilfsfunktionen ──────────────────────────────────────────────────

    def _upd32(self):
        """float64 → float32 in-place + Disk-Positionen gecacht (kein Heap-Alloc)."""
        np.copyto(self._p32, self.pos, casting='unsafe')
        # Disk-Positions-Unter-Buffer aktualisieren (Integer-Indizierung ist ~4× schneller
        # als Boolean-Masken-Indizierung für Teilmengen)
        np.take(self._p32, self._disk_idx, axis=0, out=self._disk_p32)

    # ── Tastatur-Events ──────────────────────────────────────────────────

    def _on_key(self, event):
        k = event.key.name if hasattr(event.key, 'name') else str(event.key)
        if k == 'Space':
            self.paused = not self.paused
            print('■ PAUSIERT' if self.paused else '▶ LÄUFT')
        elif k == '+':
            self.dt = min(self.dt * 1.4, 6.)
            print(f'  dt → {self.dt:.3f}')
        elif k == '-':
            self.dt = max(self.dt / 1.4, 0.005)
            print(f'  dt → {self.dt:.3f}')
        elif k.upper() == 'R':
            self.dt = self._dt0
            print(f'  dt → {self.dt:.3f}  (Reset)')
        elif k.upper() == 'T':
            frag, E, n = compute_stats(self.pos, self.vel, self.mass, self.G)
            Z_mean = float(self._feedback.Z[~self.bh_m & (self.mass > 0.)].mean())
            print(f'  [Stats] aktiv={n}  Zerlegung={frag*100:.2f}%  E={E:.1f}  '
                  f'SN_gesamt={self._total_sn}  <Z>={Z_mean:.4f}  '
                  f'AGN-Jets={self._stats_agn}  DM={self._dm is not None}')
        elif k.upper() == 'M':
            # Farbmodus umschalten: Standard ↔ Metallizität
            if self._color_mode == 'default':
                self._color_mode = 'metallicity'
                self._met_col_age    = -1     # Sofortige Aktualisierung erzwingen
                self._disk_col_dirty = True   # erste Farb-Berechnung garantieren
                print('  Farbmodus: METALLIZITÄT  (blau=arm, gelb=solar, weiß=reich)')
            else:
                self._color_mode = 'default'
                # Originalfarben in Disk-Cache wiederherstellen
                self.r_col = self._r_col_default.copy()
                np.copyto(self._disk_col, self._r_col_default[self._disk_idx], casting='unsafe')
                self._disk_col_dirty = True
                print('  Farbmodus: STANDARD')
        elif k.upper() == 'B':
            # Backend-Selektor durchschalten: BH → FMM → PM → P3M → BH
            if not self.use_bh:
                print('  Backend-Wechsel nicht verfügbar (N ≤ NUMPY_THRESH)')
                return
            cycle = [_BACKEND_BH, _BACKEND_FMM, _BACKEND_PM, _BACKEND_P3M, _BACKEND_PY]
            idx   = cycle.index(self.backend) if self.backend in cycle else 0
            self.backend = cycle[(idx + 1) % len(cycle)]
            desc = {
                _BACKEND_BH:  'Barnes-Hut     O(N log N)  – Genauigkeit: hoch',
                _BACKEND_FMM: 'Fast Multipole O(N)        – Quadrupol-Ordnung p=2',
                _BACKEND_PM:  'Particle Mesh  O(N+M³logM) – Fernfeld-only, Grid 128³',
                _BACKEND_P3M: 'P3M Hybrid     BH+PM       – Nahfeld BH + Fernfeld PM',
                _BACKEND_PY:  'Pure-Python BH O(N log N)  – kein JIT, nur N < 300',
            }
            print(f'  Backend: {desc[self.backend]}')
        elif k.upper() == 'S':
            # Snapshot als PNG speichern
            import time as _time
            from editor.ui.GalaxyViewer import GalaxyViewer as _GV
            snap_path = f'snapshot_{int(_time.time())}.png'
            active = self.mass > 0.
            _GV.save_snapshot(
                self.pos, self.r_col, self.mass,
                path=snap_path, dpi=200,
                title=f'N-Body Snapshot  –  {int(active.sum())} Partikel',
            )
            print(f'  [Snapshot] gespeichert: {snap_path}')

    def show(self):
        app.run()
        self._phys_running = False
