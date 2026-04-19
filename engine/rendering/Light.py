"""
Light – Lichtquelle für die Szene
"""

class Light:
    def __init__(self, position=(0,10,0), color=(1,1,1), intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity
