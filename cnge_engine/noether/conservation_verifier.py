import math

import torch


class ConservationVerifier:
    """Empirical relative trajectory drift; confidence bounds require independent bounded samples."""

    def __init__(self, manifold, tolerance=1e-4, confidence=0.999):
        if tolerance <= 0 or not 0 < confidence < 1:
            raise ValueError("invalid verification settings")
        self.manifold, self.tolerance, self.confidence = manifold, tolerance, confidence

    def verify_trajectory_invariance(self, Q_fn, trajectories):
        if trajectories.ndim != 3 or trajectories.shape[1] < 2 or not trajectories.numel():
            raise ValueError("expected nonempty [B,T,2d], T>=2")
        values = Q_fn(trajectories.reshape(-1, trajectories.shape[-1])).reshape(
            *trajectories.shape[:2], -1
        )
        initial = values[:, :1]
        drift = ((values - initial).abs() / (initial.abs() + torch.finfo(values.dtype).eps)).max()
        metric = float(drift.detach())
        return math.isfinite(metric) and metric < self.tolerance, metric

    def bounded_failure_upper_bound(self, failures, sample_count):
        """Hoeffding upper bound on Bernoulli failure rate for IID trajectories, not timesteps."""
        if not 0 <= failures <= sample_count or sample_count < 1:
            raise ValueError("invalid sample counts")
        return min(
            1.0,
            failures / sample_count
            + math.sqrt(math.log(1 / (1 - self.confidence)) / (2 * sample_count)),
        )
