"""
IScriptEngine – Interface für Skripting-Engines
"""

class IScriptEngine:
    def execute(self, script):
        raise NotImplementedError
