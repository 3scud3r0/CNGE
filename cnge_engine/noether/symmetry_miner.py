import torch
from torch import nn

from ..dynamics.hamiltonian_nn import smooth_mlp
from ..math_core.symplectic import scalar_gradient


class LieSymmetryMiner(nn.Module):
    def __init__(self, manifold, hidden_dims=(64, 64, 64)):
        super().__init__()
        self.manifold = manifold
        self.network = smooth_mlp(2 * manifold.spatial_dim, hidden_dims).to(manifold.J)

    def forward(self, z):
        return self.network(z)

    def compute_loss(self, z_batch, H_model):
        z = z_batch.detach().requires_grad_(True)
        Q, H = self(z), H_model(z)
        dq, dh = scalar_gradient(Q, z), scalar_gradient(H, z)
        comm = self.manifold.poisson_bracket(Q, H, z).square().mean()
        nontrivial = torch.exp(-Q.var(unbiased=False))
        denominator = dq.square().sum(-1).mean() * dh.square().sum(-1).mean()
        orth = (dq * dh).sum(-1).mean().square() / denominator.clamp_min(torch.finfo(z.dtype).eps)
        loss = comm + nontrivial + orth
        return loss, {
            "commutator": float(comm.detach()),
            "nontrivial": float(nontrivial.detach()),
            "orthogonality": float(orth.detach()),
        }
