"""
ThreadPool – Einfache Thread-Pool-Implementierung für parallele Aufgaben
"""
import threading
from queue import Queue

class ThreadPool:
    def __init__(self, num_threads):
        self.tasks = Queue()
        self.threads = []
        self.active = True
        for _ in range(num_threads):
            t = threading.Thread(target=self.worker)
            t.daemon = True
            t.start()
            self.threads.append(t)

    def worker(self):
        while self.active:
            try:
                func, args, kwargs = self.tasks.get(timeout=1)
                func(*args, **kwargs)
                self.tasks.task_done()
            except Exception:
                continue

    def submit(self, func, *args, **kwargs):
        self.tasks.put((func, args, kwargs))

    def shutdown(self, wait=True):
        self.active = False
        if wait:
            for t in self.threads:
                t.join()
