import time
from contextlib import contextmanager

import cv2


def resize_width(frame, width):
    """Resize keeping aspect ratio."""
    h, w = frame.shape[:2]
    new_h = int(h * width / w)
    return cv2.resize(frame, (width, new_h))


class StageTimer:
    """Measures how many milliseconds each pipeline stage takes.

    Usage:
        timer = StageTimer()
        with timer.stage("blur"):
            ...
        timer.times  ->  {"blur": 0.8}
    """

    def __init__(self):
        self.times = {}

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        yield
        self.times[name] = (time.perf_counter() - start) * 1000  # ms

    def total(self):
        return sum(self.times.values())