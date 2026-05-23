from __future__ import annotations

from dataclasses import dataclass

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


def probe_camera(index: int) -> CameraProbe:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
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
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
