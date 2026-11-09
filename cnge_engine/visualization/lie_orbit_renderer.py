import torch

from ..math_core.integrators import ImplicitMidpointIntegrator
from ..math_core.symplectic import scalar_gradient
from .phase_space_projector import SymplecticPhaseProjector


class LieOrbitVisualizer:
    def __init__(self, manifold, steps=32, dt=0.01):
        self.manifold, self.steps, self.dt = manifold, steps, dt

    def generate_streamlines(self, invariant_fn, bounds, grid_res=32):
        if not 2 <= grid_res <= 128 or bounds[0] >= bounds[1]:
            raise ValueError("invalid streamline bounds or grid size")
        axis = torch.linspace(
            *bounds, grid_res, device=self.manifold.J.device, dtype=self.manifold.J.dtype
        )
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        z = torch.zeros(grid_res**2, self.manifold.J.shape[0], device=axis.device, dtype=axis.dtype)
        z[:, 0], z[:, self.manifold.spatial_dim] = x.reshape(-1), y.reshape(-1)

        def field(point):
            point = point.requires_grad_(True)
            return self.manifold.hamiltonian_vector_field(
                scalar_gradient(invariant_fn(point), point)
            )

        integrator = ImplicitMidpointIntegrator()
        trajectory = [z]
        for _ in range(self.steps):
            z = integrator.step(z, field, self.dt).detach()
            trajectory.append(z)
        return {
            "vertices": SymplecticPhaseProjector(self.manifold.spatial_dim).project_to_3d(
                torch.stack(trajectory, 1)
            )
        }
