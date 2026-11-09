import torch

from cnge_engine.math_core.symplectic import SymplecticManifold, scalar_gradient


class SymplecticStructure(SymplecticManifold):
    """A skew pairing, not an inner product: u^T J u is always zero."""

    def symplectic_gradient(self, grad_H):
        return self.hamiltonian_vector_field(grad_H)

    def pairing(self, u, v):
        return (u * (v @ self.J.T)).sum(-1, keepdim=True)

    def field(self, H_fn, z):
        with torch.enable_grad():
            point = z if z.requires_grad else z.detach().requires_grad_(True)
            return self.symplectic_gradient(scalar_gradient(H_fn(point), point))

    def pullback_defect(self, jacobian):
        if jacobian.shape[-2:] != self.J.shape:
            raise ValueError("a square canonical-to-canonical Jacobian is required")
        return jacobian.transpose(-1, -2) @ self.J @ jacobian - self.J
