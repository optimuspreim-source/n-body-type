"""
MainWindow – Hauptfenster des Editors
"""

class MainWindow:
    def __init__(self):
        self.title = "3D Simulationseditor"
        self.width = 1280
        self.height = 720
        self.panels = []

    def add_panel(self, panel):
        self.panels.append(panel)

    def show(self):
        # Platzhalter für GUI-Start
        pass
