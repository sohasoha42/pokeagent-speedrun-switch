from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys

import cv2


@dataclass(frozen=True)
class CameraProbe:
    index: int
    opened: bool
    readable: bool
    shape: tuple[int, ...] | None

    @property
    def usable(self) -> bool:
        return self.opened and self.readable


@contextmanager
def suppress_native_stderr():
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def camera_backends() -> list[int]:
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_video_capture(index: int) -> cv2.VideoCapture:
    with suppress_native_stderr():
        for backend in camera_backends():
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap
            cap.release()
    return cv2.VideoCapture()


def probe_camera(index: int) -> CameraProbe:
    cap = open_video_capture(index)
    opened = cap.isOpened()
    readable = False
    shape: tuple[int, ...] | None = None

    if opened:
        ret, frame = cap.read()
        readable = bool(ret and frame is not None)
        if frame is not None:
            shape = tuple(frame.shape)

    cap.release()
    return CameraProbe(index=index, opened=opened, readable=readable, shape=shape)


def detect_cameras(max_index: int = 10) -> list[CameraProbe]:
    return [probe_camera(index) for index in range(max_index)]


def usable_camera_indices(max_index: int = 10) -> list[int]:
    return [probe.index for probe in detect_cameras(max_index) if probe.usable]


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture | None:
    cap = open_video_capture(index)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
