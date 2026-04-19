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
import numpy as np
from vispy import scene, app

from engine.physics.BarnesHutNumba import nbody_step, make_tree_arrays, warmup as bh_warmup
from engine.physics.FastMultipole  import fmm_step, make_fmm_arrays, warmup_fmm
from engine.physics.ParticleMesh   import ParticleMeshSolver
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
            [s.get('render_size', 1.5) for s in all_stars],
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

        # ── Tiefenskalierung + additiver Glow-Schicht Puffer ─────────────────
        self._disk_sz_base    = self._disk_sz.copy()            # unveränderl. Basis für Tiefenmod.
        self._disk_col_render = np.empty_like(self._disk_col)   # Render-Puffer (kein Alloc/Frame)
        self._glow_col        = np.empty((N_disk, 4), dtype=np.float32)
        self._glow_sz_base    = np.maximum(4.0, self._disk_sz * 4.0).astype(np.float32)
        self._glow_sz         = self._glow_sz_base.copy()

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
            self._pm         = ParticleMeshSolver(
                N_grid=128, box_size=3000., G=G, eps=eps)
            # Standard-Backend
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
        self.canvas.events.key_press.connect(self._on_key)

        vp = self.canvas.central_widget.add_view()
        self.view = vp
        # Kamera-Distanz: automatisch aus dem Systemumfang berechnen
        _sys_span   = float(np.linalg.norm(self.pos[self.mass > 0.], axis=1).max()) if N > 0 else 500.
        _cam_dist   = max(600., _sys_span / np.tan(np.deg2rad(24)))   # 48° effektiv FOV
        self.view.camera = scene.cameras.TurntableCamera(
            fov=52, distance=_cam_dist, elevation=28, azimuth=35)

        # ── Scatter 0: Disk-Stern-Glow (additives Blending → Nebel/Bloom-Effekt) ──
        self._upd32()
        self._rebuild_glow_colors()
        self.sc_glow_disk = scene.visuals.Markers()
        self.sc_glow_disk.set_data(
            self._disk_p32,
            face_color=self._glow_col,
            size=self._glow_sz,
            edge_width=0)
        self.sc_glow_disk.set_gl_state('additive', depth_test=False)
        self.view.add(self.sc_glow_disk)

        # ── Scatter 1: alle Disk-Sterne ──────────────────────────────────
        self.sc_disk = scene.visuals.Markers()
        self.sc_disk.set_data(
            self._disk_p32,
            face_color=self._disk_col,
            size=self._disk_sz,
            edge_width=0)
        self.sc_disk.antialias = 1.5
        self._disk_col_dirty = False
        self.view.add(self.sc_disk)

        # ── Scatter 2: SMBH-Schatten (dunkle, große Marker) ─────────────
        self.sc_shadow = scene.visuals.Markers()
        self.view.add(self.sc_shadow)

        # ── Scatter 3: Photonen-Ring-Glow (leuchtend, noch größer) ──────
        self.sc_glow = scene.visuals.Markers()
        self.sc_glow.set_gl_state('additive', depth_test=False)
        self.view.add(self.sc_glow)

        self._update_bh_visuals()

        # ── Physik-Thread: Render (~60 FPS) vollständig von Physik entkoppelt ──
        # Numba-JIT-Kernel (BH, FMM) geben die Python-GIL frei → echter Parallelismus.
        _nd = len(self._disk_idx)
        self._phys_disk_p32   = np.empty((_nd, 3), dtype=np.float32)  # Physik schreibt hier
        self._render_disk_p32 = self._disk_p32.copy()                  # Puffer unter Lock geteilt
        self._rlock       = threading.Lock()
        self._rsnap_new   = False
        self._rsnap_bh    = None                  # (pos32, sz_s, sz_g) oder None
        self._rsnap_stats = self._stats           # (frag, E, nact)
        self._rsnap_zval  = 0.0
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
                self.pos, self.vel = self._pm.step(
                    self.pos, self.vel, self.mass, self.dt)

            elif self.backend == _BACKEND_P3M:
                # P3M: PM für Fernfeld + BH für Nahfeld-Korrektur
                # Fernfeld: PM-Beschleunigung berechnen
                pm_accel = self._pm.compute_accel(self.pos, self.mass)
                # Nahfeld-Korrektur: BH Schritt durchführen, dann PM hinzuaddieren
                self.pos, self.vel = nbody_step(
                    self.pos, self.vel, self.mass,
                    self.dt, self.G, self.eps2_arr, self.theta,
                    *self._tree, self._max_nodes)
                # PM-Korrektur zusätzlich applizieren (Kic velocities additiv)
                active = self.mass > 0.
                self.vel[active] += pm_accel[active] * self.dt * 0.5   # halber Schritt (bereits integriert)
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

    # ── Physik-Hintergrundthread ──────────────────────────────────────────
    # Numba-JIT-Kernel (BH, FMM) geben die Python-GIL frei → echter Parallelismus
    # zwischen Physik-Thread und Render-Thread (vispy event loop).

    def _phys_loop(self):
        """Physik kontinuierlich im Daemon-Thread; Render läuft unabhängig bei ~60 FPS."""
        phys_fn = 0
        while self._phys_running:
            if self.paused:
                time.sleep(0.005)
                continue

            # ── N-Body-Schritte + Merger ──────────────────────────────────
            for step in range(self.spf):
                self._step()
                if (phys_fn * self.spf + step) % _MERGER_EVERY == 0:
                    _nbhm, _nacc, acc_mass = apply_mergers(
                        self.pos, self.vel, self.mass,
                        self.bh_idx, _MERGE_BH_R, _ACCRETE_R)
                    if acc_mass:
                        n_jet = apply_agn_jets(
                            self.bh_idx, acc_mass,
                            self.pos, self.vel, self.mass,
                            jet_threshold=_JET_THRESHOLD, G=self.G)
                        self._stats_agn += n_jet

            # ── Statistiken ────────────────────────────────────────────────
            if phys_fn % _STATS_EVERY == 0:
                self._stats = compute_stats(self.pos, self.vel, self.mass, self.G)
            frag, Esys, nact = self._stats
            zval = float(self._feedback.Z[~self.bh_m & (self.mass > 0.)].mean()) \
                   if nact > 3 else 0.

            # ── BH-Visualdaten vorbereiten ─────────────────────────────────
            bh_data = None
            if phys_fn % _BH_VIS_EVERY == 0:
                abh = [i for i in self.bh_idx if self.mass[i] > 0.]
                if abh:
                    sz_s = np.array(
                        [max(10., min(0.018 * float(self.mass[i])**.55, 32.))
                         for i in abh], dtype=np.float32)
                    bh_data = (self.pos[abh].astype(np.float32).copy(),
                               sz_s, sz_s * 1.75)

            # ── Render-Snapshot: float64→float32, dann kurz locken ─────────
            np.copyto(self._p32, self.pos, casting='unsafe')
            np.take(self._p32, self._disk_idx, axis=0, out=self._phys_disk_p32)
            with self._rlock:
                np.copyto(self._render_disk_p32, self._phys_disk_p32)
                if bh_data is not None:
                    self._rsnap_bh = bh_data
                self._rsnap_stats = (frag, Esys, nact)
                self._rsnap_zval  = zval
                self._rsnap_new   = True
            phys_fn += 1

    # ── Timer-Callback (Render-Thread, ~60 FPS) ───────────────────────────

    def _on_timer(self, event):
        # Render-Snapshot vom Physik-Thread übernehmen (Lock nur für kurzen copyto)
        with self._rlock:
            if self._rsnap_new:
                np.copyto(self._disk_p32, self._render_disk_p32)
                self._rsnap_new = False
            bh_data = self._rsnap_bh
            stats   = self._rsnap_stats
            zval    = self._rsnap_zval

        # Metallizitäts-Farbmodus
        if self._color_mode == 'metallicity':
            if self._disk_col_dirty or (self.frame_n - self._met_col_age >= 5):
                met_col = self._feedback.metallicity_colors()
                np.copyto(self._disk_col, met_col[self._disk_idx], casting='unsafe')
                self._met_col_age    = self.frame_n
                self._disk_col_dirty = True

        # ── Tiefenskalierung (Kamera-Raumprojektion) ─────────────────────
        _az  = np.radians(float(self.view.camera.azimuth))
        _el  = np.radians(float(self.view.camera.elevation))
        _cam = np.array([np.cos(_el) * np.cos(_az),
                         np.cos(_el) * np.sin(_az),
                         np.sin(_el)], dtype=np.float32)
        _d  = self._disk_p32 @ _cam
        _dt = (_d - _d.min()) / max(float(_d.max() - _d.min()), 1e-6)

        np.multiply(self._disk_sz_base, 0.85 + 0.30 * _dt, out=self._disk_sz)
        np.copyto(self._disk_col_render, self._disk_col)
        self._disk_col_render[:, 3] *= 0.60 + 0.40 * _dt

        if self._disk_col_dirty:
            self._rebuild_glow_colors()
        np.multiply(self._glow_sz_base, 0.85 + 0.30 * _dt, out=self._glow_sz)
        self.sc_glow_disk.set_data(
            self._disk_p32, face_color=self._glow_col,
            size=self._glow_sz, edge_width=0)
        self.sc_disk.set_data(
            self._disk_p32,
            face_color=self._disk_col_render,
            size=self._disk_sz,
            edge_width=0)
        self._disk_col_dirty = False

        # ── SMBH-Visuals (aus Physik-Snapshot) ────────────────────────────
        if bh_data is not None:
            pos32, sz_s, sz_g = bh_data
            col_s = np.tile([0.02, 0.02, 0.04, 1.00], (len(sz_s), 1)).astype(np.float32)
            col_g = np.tile([1.00, 0.68, 0.12, 0.18], (len(sz_s), 1)).astype(np.float32)
            self.sc_shadow.set_data(pos32, face_color=col_s, size=sz_s, edge_width=0)
            self.sc_glow  .set_data(pos32, face_color=col_g, size=sz_g, edge_width=0)

        # ── Fenstertitel ───────────────────────────────────────────────────
        frag, Esys, nact = stats
        dm_str   = '★DM' if self._dm else ''
        mode_str = '[Z]'  if self._color_mode == 'metallicity' else ''
        self.canvas.title = (
            f'N-Body {self.N}  akt:{nact} {dm_str}{mode_str}  |  '
            f'[{self.backend}]  |  '
            f'SN:{self._total_sn}  <Z>:{zval:.3f}  '
            f'Zerlg:{frag*100:.1f}%  |  '
            f'dt={self.dt:.2f}   SPACE +/- R T M B'
        )
        self.canvas.update()
        self.frame_n += 1

    # ── Hilfsfunktionen ──────────────────────────────────────────────────

    def _rebuild_glow_colors(self):
        """Glow-Farben aus aktuellen Disk-Farben ableiten (aufgehellt, sehr transparent).

        Mit additivem Blending akkumuliert sich das Glow-Licht in dichten Regionen
        (Galaxienkerne, Cluster) zu leuchtenden Flächen – sparse Gebiete bleiben dunkel.
        """
        np.clip(self._disk_col[:, :3] * 1.6, 0., 1., out=self._glow_col[:, :3])
        self._glow_col[:, 3] = 0.045

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
                self._met_col_age = -1   # Sofortige Aktualisierung erzwingen
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
            cycle = [_BACKEND_BH, _BACKEND_FMM, _BACKEND_PM, _BACKEND_P3M]
            idx   = cycle.index(self.backend) if self.backend in cycle else 0
            self.backend = cycle[(idx + 1) % len(cycle)]
            desc = {
                _BACKEND_BH:  'Barnes-Hut     O(N log N)  – Genauigkeit: hoch',
                _BACKEND_FMM: 'Fast Multipole O(N)        – Quadrupol-Ordnung p=2',
                _BACKEND_PM:  'Particle Mesh  O(N+M³logM) – Fernfeld-only, Grid 128³',
                _BACKEND_P3M: 'P3M Hybrid     BH+PM       – Nahfeld BH + Fernfeld PM',
            }
            print(f'  Backend: {desc[self.backend]}')

    def show(self):
        app.run()
        self._phys_running = False
