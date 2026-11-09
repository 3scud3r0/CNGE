import torch
from torch import nn


def scalar_gradient(value, z):
    """Gradient of independent scalar evaluations, including constant functions."""
    if not value.requires_grad:
        return z * 0
    grad = torch.autograd.grad(
        value.sum(), z, create_graph=True, retain_graph=True, allow_unused=True
    )[0]
    return z * 0 if grad is None else grad


class SymplecticManifold(nn.Module):
    def __init__(self, spatial_dim, device="cpu", dtype=torch.float32):
        super().__init__()
        if spatial_dim < 1:
            raise ValueError("spatial_dim must be positive")
        self.spatial_dim = spatial_dim
        eye = torch.eye(spatial_dim, device=device, dtype=dtype)
        zero = torch.zeros_like(eye)
        self.register_buffer(
            "J", torch.cat((torch.cat((zero, eye), 1), torch.cat((-eye, zero), 1)), 0)
        )

    def canonical_matrix(self):
        return self.J

    def hamiltonian_vector_field(self, grad_H):
        if grad_H.shape[-1] != self.J.shape[0]:
            raise ValueError("incorrect phase dimension")
        return grad_H @ self.J.T

    def poisson_bracket(self, f, g, z):
        if not z.requires_grad:
            raise ValueError("z must require gradients")
        return (scalar_gradient(f, z) * self.hamiltonian_vector_field(scalar_gradient(g, z))).sum(
            -1, keepdim=True
        )

    def check_symplectomorphism(self, jacobian_map, tol=1e-6):
        error = jacobian_map.transpose(-1, -2) @ self.J @ jacobian_map - self.J
        return bool(torch.linalg.matrix_norm(error).max() < tol), error
