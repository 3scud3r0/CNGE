from collections import deque

import torch


class ObservationHistory:
    """Bounded delay coordinates for sensors whose instantaneous frame is not observable."""

    def __init__(self, frame_dim, length=4):
        if frame_dim < 1 or length < 2:
            raise ValueError("positive frame dimension and at least two history frames required")
        self.frame_dim, self.length = frame_dim, length
        self.frames = deque(maxlen=length)

    def update(self, frame):
        value = torch.as_tensor(frame)
        if value.shape != (self.frame_dim,) or not torch.isfinite(value).all():
            raise ValueError("one finite frame vector is required")
        if self.frames and (
            value.dtype != self.frames[0].dtype or value.device != self.frames[0].device
        ):
            raise ValueError("reset history before changing dtype/device")
        self.frames.append(value.detach().clone())
        return torch.cat(tuple(self.frames)) if len(self.frames) == self.length else None

    def reset(self):
        self.frames.clear()
