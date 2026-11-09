import torch
from torch import nn

from .hamiltonian_nn import smooth_mlp


class LagrangianNN(nn.Module):
    def __init__(self, spatial_dim, hidden_dims=(128, 128)):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.network = smooth_mlp(2 * spatial_dim, hidden_dims)

    def forward(self, q, q_dot):
        return self.network(torch.cat((q, q_dot), -1))

    def compute_acceleration(self, q, q_dot):
        def scalar(qi, vi):
            return self(qi[None], vi[None]).sum()

        rows = []
        for qi, vi in zip(q, q_dot):
            dLq = torch.func.grad(scalar, argnums=0)(qi, vi)
            momentum = torch.func.grad(scalar, argnums=1)
            mass = torch.func.jacrev(momentum, argnums=1)(qi, vi)
            mixed = torch.func.jacrev(momentum, argnums=0)(qi, vi)
            # Singular/ill-conditioned learned mass matrices must not be silently inverted.
            if not torch.isfinite(mass).all() or torch.linalg.cond(mass).detach() > 1e10:
                raise RuntimeError("singular or ill-conditioned Lagrangian mass matrix")
            rows.append(torch.linalg.solve(mass, dLq - mixed @ vi))
        return torch.stack(rows)
