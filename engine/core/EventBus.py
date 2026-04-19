"""
EventBus – Publisher/Subscriber-Eventsystem
"""
from collections import defaultdict


class EventBus:
    def __init__(self):
        self._listeners = defaultdict(list)

    def subscribe(self, event_type: str, callback):
        self._listeners[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback):
        self._listeners[event_type].remove(callback)

    def emit(self, event_type: str, data=None):
        for cb in self._listeners[event_type]:
            cb(data)
