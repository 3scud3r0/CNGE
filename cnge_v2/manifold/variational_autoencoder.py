import math

import torch
from torch import nn

from cnge_engine.dynamics.hamiltonian_nn import SmoothSoftplus

from ..solvers.implicit_midpoint import ImplicitMidpointSolver
from .canonical_symplectic import SymplecticStructure


class SymplecticAutoencoder(nn.Module):
    """Differential autoencoder; latent Hamiltonian dynamics, not an identifiable canonical encoder.

    `variational` refers to the dynamical objective, not a probabilistic ELBO.
    A single position image cannot determine an unobserved velocity; supply history features.
    """

    def __init__(
        self,
        obs_dim,
        spatial_dim,
        hidden_dim=64,
        tangent_weight=0.01,
        prediction_weight=1.0,
        observation_noise_variance=0.0,
        solver=None,
    ):
        super().__init__()
        if min(obs_dim, spatial_dim, hidden_dim) < 1 or min(tangent_weight, prediction_weight) < 0:
            raise ValueError("invalid autoencoder configuration")
        if not math.isfinite(observation_noise_variance) or observation_noise_variance < 0:
            raise ValueError("observation noise variance must be finite and nonnegative")
        self.observation_noise_variance = observation_noise_variance
        self.solver = solver or ImplicitMidpointSolver()
        self.obs_dim, self.spatial_dim = obs_dim, spatial_dim
        self.tangent_weight, self.prediction_weight = tangent_weight, prediction_weight
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), SmoothSoftplus(), nn.Linear(hidden_dim, 2 * spatial_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * spatial_dim, hidden_dim), SmoothSoftplus(), nn.Linear(hidden_dim, obs_dim)
        )
        self.structure = SymplecticStructure(spatial_dim)

    def forward(self, x):
        if x.shape[-1] != self.obs_dim:
            raise ValueError("wrong observation dimension")
        z = self.encoder(x)
        return z, self.decoder(z)

    def compute_loss(self, x_seq, dt, H_fn):
        if (
            x_seq.ndim != 3
            or x_seq.shape[1:] != (2, self.obs_dim)
            or not torch.isfinite(x_seq).all()
            or not math.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError("expected finite [B,2,obs_dim] and positive dt")
        with torch.enable_grad():
            z0, r0 = self(x_seq[:, 0])
            _z1, r1 = self(x_seq[:, 1])
            if not z0.requires_grad:
                z0 = z0.requires_grad_(True)
            field = self.structure.field(H_fn, z0)
            _, tangent = torch.autograd.functional.jvp(self.decoder, z0, field, create_graph=True)
            rec = (r0 - x_seq[:, 0]).square().mean() + (r1 - x_seq[:, 1]).square().mean()
            tangent_loss = ((x_seq[:, 1] - x_seq[:, 0]) / dt - tangent).square().mean()
            # Integrated observation residual avoids the dt^-2 amplification of sensor noise.
            # The actual temporal prediction always uses implicit midpoint, never an Euler fallback.
            predicted = self.solver.integrate_step(z0, lambda z: self.structure.field(H_fn, z), dt)
            prediction = (self.decoder(predicted) - x_seq[:, 1]).square().mean()
            noise_scale = 1 + 2 * self.observation_noise_variance / (dt * dt)
            total = (
                rec
                + self.tangent_weight * tangent_loss / noise_scale
                + self.prediction_weight * prediction
            )
        return {
            "total_loss": total,
            "rec_loss": rec,
            "tangent_loss": tangent_loss,
            "prediction_loss": prediction,
        }
