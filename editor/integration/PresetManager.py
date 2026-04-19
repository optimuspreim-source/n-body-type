"""
PresetManager – Lädt und verwaltet Presets/Templates für Szenen
"""
import importlib.util
import os

class PresetManager:
    def __init__(self, preset_dir='assets/presets'):
        self.preset_dir = preset_dir
        self.presets = self._discover_presets()

    def _discover_presets(self):
        presets = []
        for fname in os.listdir(self.preset_dir):
            if fname.endswith('.py'):
                presets.append(fname[:-3])
        return presets

    def load_preset(self, name):
        path = os.path.join(self.preset_dir, name + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
