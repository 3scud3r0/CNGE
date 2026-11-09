import torch


class PhaseTopologyAnalyzer:
    def __init__(self, manifold):
        self.manifold = manifold

    def compute_poincare_map(self, trajectory, surface_fn, chart_fn=None):
        """Positive transverse crossings, linearly interpolated in ambient coordinates.

        A user-supplied surface chart is required to return intrinsic 2d-1 coordinates.
        A general surface has no canonical coordinate that can safely be dropped.
        """
        if trajectory.ndim != 2 or len(trajectory) < 2:
            raise ValueError("expected [T,2d], T>=2")
        s = surface_fn(trajectory).reshape(-1)
        if len(s) != len(trajectory):
            raise ValueError("surface must return one value per point")
        indices = torch.where((s[:-1] < 0) & (s[1:] >= 0))[0]
        fraction = -s[indices] / (s[indices + 1] - s[indices])
        points = trajectory[indices] + fraction[:, None] * (
            trajectory[indices + 1] - trajectory[indices]
        )
        return chart_fn(points) if chart_fn else points

    def detect_attractor_dimensionality(self, trajectory):
        """Finite-sample correlation dimension, with bounded O(2048^2) memory."""
        if trajectory.ndim != 2 or len(trajectory) < 20 or not torch.isfinite(trajectory).all():
            raise ValueError("need at least 20 finite trajectory points")
        stride = max(1, (len(trajectory) + 2047) // 2048)
        distances = torch.pdist(trajectory[::stride]).double()
        distances = distances[distances > 0]
        if len(distances) < 10:
            raise ValueError("degenerate trajectory")
        lo, hi = torch.quantile(
            distances, torch.tensor([0.05, 0.5], device=distances.device, dtype=distances.dtype)
        )
        if hi <= lo:
            raise ValueError("no usable scale interval")
        radii = torch.exp(torch.linspace(lo.log(), hi.log(), 16, device=lo.device))
        counts = (distances[:, None] < radii).double().mean(0)
        valid = (counts > 0) & (counts < 1)
        if valid.sum() < 3:
            raise ValueError("insufficient correlation scales")
        x, y = radii[valid].log(), counts[valid].log()
        return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())

    @staticmethod
    def pullback_metric(flow_fn, z):
        jacobians = torch.stack([torch.func.jacrev(flow_fn)(point) for point in z])
        return jacobians.transpose(-1, -2) @ jacobians
