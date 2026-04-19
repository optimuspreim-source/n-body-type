"""
SelectionTool – Werkzeug zur Objektauswahl
"""

class SelectionTool:
    def __init__(self):
        self.selected_object = None

    def select(self, obj):
        self.selected_object = obj
