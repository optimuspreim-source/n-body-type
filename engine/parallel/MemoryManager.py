"""
MemoryManager – Platzhalter für Speicherverwaltung
"""

class MemoryManager:
    def __init__(self):
        self.allocated = 0

    def allocate(self, size):
        self.allocated += size
        # Platzhalter für echten Speicherzugriff
        return bytearray(size)

    def free(self, size):
        self.allocated -= size
