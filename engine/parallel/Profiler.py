"""
Profiler – Einfaches Zeit- und Speicherprofiling
"""
import time
import tracemalloc

class Profiler:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.snapshots = []

    def start(self):
        self.start_time = time.perf_counter()
        tracemalloc.start()

    def stop(self):
        self.end_time = time.perf_counter()
        self.snapshots.append(tracemalloc.take_snapshot())
        tracemalloc.stop()

    def elapsed(self):
        return self.end_time - self.start_time if self.end_time and self.start_time else 0

    def memory_stats(self):
        if self.snapshots:
            return self.snapshots[-1].statistics('filename')
        return []
