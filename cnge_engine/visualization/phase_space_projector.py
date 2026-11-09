import torch


class SymplecticPhaseProjector:
    """Display slice only: odd-dimensional 3D space cannot preserve a symplectic form."""

    def __init__(self, spatial_dim):
        self.spatial_dim = spatial_dim

    def project_to_3d(self, z_trajectories, projection_plane="canonical_canonical"):
        if (
            projection_plane != "canonical_canonical"
            or z_trajectories.shape[-1] != 2 * self.spatial_dim
        ):
            raise ValueError("unsupported projection or wrong phase dimension")
        d = self.spatial_dim
        if d == 1:
            points = torch.stack(
                (
                    z_trajectories[..., 0],
                    z_trajectories[..., 1],
                    torch.zeros_like(z_trajectories[..., 0]),
                ),
                -1,
            )
        else:
            points = z_trajectories[..., [0, 1, d]]
        return points.detach().cpu().numpy()
