"""Medición de tiempo de pared y memoria pico (incluye procesos hijos, p. ej. la JVM de Spark)."""
import json
import threading
import time

import psutil


class Peak:
    """Muestrea el RSS del proceso y sus hijos en un hilo; se usa con `with Peak() as m:`."""

    def __init__(self, every=0.25):
        self.every, self.peak, self.sec = every, 0, None
        self._stop = threading.Event()

    def _rss(self):
        p = psutil.Process()
        tot = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                tot += c.memory_info().rss
            except psutil.Error:
                pass
        return tot

    def _loop(self):
        while not self._stop.wait(self.every):
            self.peak = max(self.peak, self._rss())

    def __enter__(self):
        self.base = self._rss()
        self.peak = self.base
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.sec = time.perf_counter() - self.t0
        self._stop.set()
        self._t.join()
        self.gb = self.peak / 1024**3


def append_jsonl(path, row):
    """Agrega una fila a un .jsonl; así una corrida interrumpida conserva lo ya calculado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(row, default=float) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf8") as f:
        return [json.loads(x) for x in f if x.strip()]
