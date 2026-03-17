# timer.py
import time
from contextlib import ContextDecorator

class _TimerBlock(ContextDecorator):
    def __init__(self, label: str, printer=print):
        self.label = label
        self.printer = printer
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        self.printer(f"[TIMER] {self.label}: {dt:.3f} s")
        # don't suppress exceptions
        return False


class Timer:
    """
    Usage:
        import timer
        tim = timer.Timer()

        with tim("step name"):
            ...
    """
    def __init__(self, printer=print):
        self.printer = printer

    def __call__(self, label: str):
        return _TimerBlock(label, printer=self.printer)
