"""
DragDropPanel – Ermöglicht Drag & Drop von Preset-Objekten in die Szene
"""
class DragDropPanel:
    def __init__(self, preset_manager):
        self.preset_manager = preset_manager
        self.selected_preset = None

    def show_presets(self):
        # Listet verfügbare Presets auf
        for name in self.preset_manager.presets:
            print(f"Preset: {name}")

    def select_preset(self, name):
        self.selected_preset = self.preset_manager.load_preset(name)

    def drag_to_scene(self, scene, position=(0,0,0)):
        if self.selected_preset:
            stars = self.selected_preset.generate_spiral_galaxy()
            for star in stars:
                # Füge Sternobjekt an Position hinzu (Platzhalter)
                scene.add_object(star, position)
