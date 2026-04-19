"""
Config – Konfigurationsklasse für Engine-Parameter
"""


class Config:
    def __init__(self):
        self.target_fps = 60
        self.window_width = 1280
        self.window_height = 720
        self.window_title = "N-Body Simulation Engine"
        self.theta = 0.5        # Barnes-Hut threshold
        self.G = 6.674e-11      # Gravitationskonstante
        self.num_threads = 8
        self.timestep = 0.01
