"""
TransformTool – Werkzeug zum Verschieben, Rotieren, Skalieren
"""

class TransformTool:
    def __init__(self):
        self.mode = 'translate'  # oder 'rotate', 'scale'
        self.selected_object = None

    def set_mode(self, mode):
        self.mode = mode

    def apply(self, value):
        # Platzhalter für Transformation
        pass
