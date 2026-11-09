import torch


class OnlineSymplecticStateFilter:
    """EMA finite differences with explicit constant mass; canonical only for that model."""

    def __init__(self, spatial_dim, dt, mass=1.0, smoothing=1.0):
        if spatial_dim < 1 or dt <= 0 or mass <= 0 or not 0 < smoothing <= 1:
            raise ValueError("invalid filter configuration")
        self.spatial_dim, self.dt, self.mass, self.smoothing = spatial_dim, dt, mass, smoothing
        self.previous = self.velocity = None

    def update_and_extract_phase_point(self, q_observed):
        if (
            q_observed.ndim != 2
            or q_observed.shape[-1] != self.spatial_dim
            or not q_observed.is_floating_point()
            or not torch.isfinite(q_observed).all()
        ):
            raise ValueError("expected finite floating [B,d] coordinates")
        q = q_observed.detach()
        if self.previous is not None and (
            self.previous.shape != q.shape
            or self.previous.device != q.device
            or self.previous.dtype != q.dtype
        ):
            raise ValueError("reset filter before changing batch/device/dtype")
        velocity = torch.zeros_like(q) if self.previous is None else (q - self.previous) / self.dt
        self.velocity = (
            velocity
            if self.velocity is None
            else self.smoothing * velocity + (1 - self.smoothing) * self.velocity
        )
        self.previous = q.clone()
        return torch.cat((q, self.mass * self.velocity), -1)

    def reset(self):
        self.previous = self.velocity = None
