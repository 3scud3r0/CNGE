import asyncio
from collections import deque

import torch


class ContinuousSensorStream:
    """Bounded telemetry buffer. Camera frames require an explicit frame-to-coordinate extractor."""

    def __init__(self, source_id, target_fps=60, capacity=4096):
        if target_fps <= 0 or capacity < 1:
            raise ValueError("invalid stream controls")
        self.source_id, self.target_fps = source_id, target_fps
        self.buffer = deque(maxlen=capacity)

    def push(self, coordinates):
        value = torch.as_tensor(coordinates).detach().clone()
        if value.ndim != 1 or not torch.isfinite(value).all():
            raise ValueError("coordinates must be a finite vector")
        if self.buffer and value.shape != self.buffer[-1].shape:
            raise ValueError("coordinate dimension changed")
        self.buffer.append(value)

    def get_state_buffer(self, window_size):
        if not 0 < window_size <= len(self.buffer):
            raise ValueError("insufficient samples or invalid window")
        return torch.stack(list(self.buffer)[-window_size:])

    async def capture(self, coordinate_extractor):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("install cnge-engine[camera] for camera capture") from exc
        capture = cv2.VideoCapture(self.source_id)
        try:
            if not capture.isOpened():
                raise RuntimeError("camera source unavailable")
            while True:
                ok, frame = await asyncio.to_thread(capture.read)
                if not ok:
                    break
                self.push(coordinate_extractor(frame))
                await asyncio.sleep(1 / self.target_fps)
        finally:
            capture.release()
