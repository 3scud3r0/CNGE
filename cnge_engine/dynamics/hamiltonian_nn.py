import torch
from torch import nn

from ..math_core.symplectic import scalar_gradient


class SmoothSoftplus(nn.Module):
    """Stable softplus without the piecewise linear threshold of nn.Softplus."""

    def __init__(self, beta=5.0):
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = beta

    def forward(self, x):
        return torch.logaddexp(torch.zeros_like(x), self.beta * x) / self.beta


def smooth_mlp(input_dim, hidden_dims, beta=5.0):
    layers = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), SmoothSoftplus(beta)))
        previous = width
    return nn.Sequential(*layers, nn.Linear(previous, 1))


class HamiltonianNN(nn.Module):
    def __init__(self, manifold, hidden_dims=(128, 128, 128), beta=5.0):
        super().__init__()
        self.manifold = manifold
        self.network = smooth_mlp(2 * manifold.spatial_dim, hidden_dims, beta).to(manifold.J)

    def forward(self, z):
        return self.network(z)

    def predict_vector_field(self, z):
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            return self.manifold.hamiltonian_vector_field(scalar_gradient(self(z), z))

    def trajectory_loss(self, z_observed, dt):
        if dt <= 0 or z_observed.ndim != 3 or z_observed.shape[1] != 2:
            raise ValueError("expected [B,2,2d] and positive dt")
        midpoint = z_observed.mean(1)
        observed = (z_observed[:, 1] - z_observed[:, 0]) / dt
        return ((self.predict_vector_field(midpoint) - observed) ** 2).sum(-1).mean()
