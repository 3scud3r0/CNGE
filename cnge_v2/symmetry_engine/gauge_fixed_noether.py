import weakref

import torch
from torch import nn

from cnge_engine.dynamics.hamiltonian_nn import smooth_mlp
from cnge_engine.math_core.symplectic import scalar_gradient

from ..manifold.canonical_symplectic import SymplecticStructure


class DegenerateGaugeError(RuntimeError):
    """No normalized independent scalar generator exists for this candidate/reference set."""


class GaugeFixedNoetherMiner(nn.Module):
    """Sobolev Gram-Schmidt of scalar functions; never divides by u^T J u.

    Gauge coefficients are constant with respect to query z, preserving integrability.
    Guarantees concern the fixed reference set, not every point of phase space.
    """

    def __init__(self, spatial_dim, hidden_dim=64, orth_weight=5.0, norm_weight=10.0):
        super().__init__()
        self.structure = SymplecticStructure(spatial_dim)
        self.network = smooth_mlp(2 * spatial_dim, (hidden_dim, hidden_dim))
        # The gauge removes additive constants exactly; do not optimize an unidentifiable bias.
        self.network[-1].register_parameter("bias", None)
        self.orth_weight, self.norm_weight = orth_weight, norm_weight
        self.register_buffer("reference", torch.empty(0, 2 * spatial_dim))
        self._hamiltonian = None

    def set_reference(self, points, H_model):
        if (
            points.ndim != 2
            or len(points) < 4
            or points.shape[-1] != self.structure.J.shape[0]
            or not torch.isfinite(points).all()
        ):
            raise ValueError("need at least four finite phase-space reference points")
        self.reference = points.detach().clone()
        self._hamiltonian = weakref.ref(H_model)

    def _gauge(self):
        H_model = self._hamiltonian() if self._hamiltonian else None
        if H_model is None or not len(self.reference):
            raise RuntimeError("call set_reference(points, H_model) before evaluating the miner")
        with torch.enable_grad():
            reference = self.reference.detach().requires_grad_(True)
            f, h = self.network(reference), H_model(reference)
            df, dh = scalar_gradient(f, reference), scalar_gradient(h, reference)
            h_norm = dh.square().sum(-1).mean()
            eps = 100 * torch.finfo(reference.dtype).eps
            if not torch.isfinite(h_norm) or float(h_norm.detach()) <= eps:
                raise DegenerateGaugeError("Hamiltonian is locally constant on reference set")
            coefficient = (df * dh).sum(-1).mean() / h_norm
            residual_gradient = df - coefficient * dh
            squared_norm = residual_gradient.square().sum(-1).mean()
            perpendicular = (
                residual_gradient
                - (
                    (residual_gradient * dh).sum(-1, keepdim=True)
                    / dh.square().sum(-1, keepdim=True).clamp_min(eps)
                )
                * dh
            )
            independent = perpendicular.square().sum(-1).mean()
            if not torch.isfinite(squared_norm) or float(squared_norm.detach()) <= eps:
                raise DegenerateGaugeError("constant or energy-alias candidate rejected")
            if float(independent.detach() / squared_norm.detach()) <= eps:
                raise DegenerateGaugeError("candidate is functionally dependent on energy")
            scale = squared_norm.sqrt()
        return H_model, coefficient, f.mean(), h.mean(), scale

    def forward(self, z):
        H, a, f_mean, h_mean, scale = self._gauge()
        return (self.network(z) - f_mean - a * (H(z) - h_mean)) / scale

    def compute_orthogonalized_loss(self, z, H_model):
        if not len(self.reference):
            self.set_reference(z, H_model)
        elif self._hamiltonian() is not H_model:
            self.set_reference(self.reference, H_model)
        with torch.enable_grad():
            point = z.detach().requires_grad_(True)
            Q, H = self(point), H_model(point)
            dq, dh = scalar_gradient(Q, point), scalar_gradient(H, point)
            bracket = self.structure.pairing(dq, dh)
            comm = bracket.square().mean()
            orth = (dq * dh).sum(-1).mean().square() / (
                dq.square().sum(-1).mean() * dh.square().sum(-1).mean()
            ).clamp_min(torch.finfo(point.dtype).eps)
            norm = (dq.square().sum(-1).mean() - 1).square()
            total = comm + self.orth_weight * orth + self.norm_weight * norm
            metrics = {
                "commutator": float(comm.detach()),
                "orthogonality": float(orth.detach()),
                "unit_norm_loss": float(norm.detach()),
            }
            del Q, H, dq, dh, bracket
        return total, metrics
