"""
Engine Core – Hauptklasse, Lebenszyklus, Hauptschleife
"""
import time
from engine.core.EventBus import EventBus
from engine.core.Config import Config


class Engine:
    _instance = None

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.event_bus = EventBus()
        self.systems = []
        self.running = False
        Engine._instance = self

    @staticmethod
    def get():
        return Engine._instance

    def add_system(self, system):
        self.systems.append(system)

    def run(self):
        self.running = True
        self._on_start()
        last = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            dt = now - last
            last = now
            self._update(dt)
        self._on_stop()

    def stop(self):
        self.running = False

    def _on_start(self):
        for s in self.systems:
            if hasattr(s, 'on_start'):
                s.on_start()

    def _update(self, dt: float):
        for s in self.systems:
            if hasattr(s, 'update'):
                s.update(dt)

    def _on_stop(self):
        for s in self.systems:
            if hasattr(s, 'on_stop'):
                s.on_stop()
