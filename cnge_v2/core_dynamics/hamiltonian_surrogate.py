from torch import nn

from cnge_engine.dynamics.hamiltonian_nn import smooth_mlp

from ..manifold.canonical_symplectic import SymplecticStructure


class HamiltonianSurrogate(nn.Module):
    def __init__(self, spatial_dim, hidden_dim=128):
        super().__init__()
        self.structure = SymplecticStructure(spatial_dim)
        self.network = smooth_mlp(2 * spatial_dim, (hidden_dim, hidden_dim))

    def energy(self, z):
        return self.network(z)

    def forward(self, z):
        return self.energy(z)

    def vector_field(self, z):
        return self.structure.field(self.energy, z)
