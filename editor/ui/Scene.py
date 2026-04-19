"""
Scene – Repräsentiert die aktuelle Szene im Editor
"""
class Scene:
    def __init__(self):
        self.objects = []

    def add_object(self, obj, position=(0,0,0)):
        # Objekt mit Position speichern (Platzhalter)
        obj['position'] = position
        self.objects.append(obj)

    def list_objects(self):
        for obj in self.objects:
            print(obj)
