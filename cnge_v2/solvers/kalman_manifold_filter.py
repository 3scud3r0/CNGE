import math

import torch

from .implicit_midpoint import ImplicitMidpointSolver


class SymplecticEKF:
    """Euclidean EKF in canonical latent coordinates; measurement updates are not symplectic maps."""

    def __init__(self, spatial_dim, Q_cov=1e-5, R_cov=1e-2, *, solver=None):
        if spatial_dim < 1 or not all(math.isfinite(v) and v > 0 for v in (Q_cov, R_cov)):
            raise ValueError("invalid EKF configuration")
        self.n = 2 * spatial_dim
        self.Q_cov, self.R_cov = Q_cov, R_cov
        self.solver = solver or ImplicitMidpointSolver()

    def _validate(self, z, covariance):
        if z.ndim != 2 or z.shape[-1] != self.n or covariance.shape != (len(z), self.n, self.n):
            raise ValueError("expected z[B,2d] and P[B,2d,2d]")
        if not torch.isfinite(z).all() or not torch.isfinite(covariance).all():
            raise ValueError("nonfinite EKF input")
        if not torch.allclose(covariance, covariance.transpose(-1, -2), rtol=1e-5, atol=1e-7):
            raise ValueError("covariance must be symmetric")
        if torch.linalg.eigvalsh(covariance).min() < -1e-7:
            raise ValueError("covariance must be positive semidefinite")

    def predict(self, z_post, P_post, vector_field_fn, dt):
        self._validate(z_post, P_post)
        states, covariances = [], []
        for z, P in zip(z_post, P_post):

            def flow(point):
                return self.solver.integrate_step(point[None], vector_field_fn, dt)[0]

            with torch.enable_grad():
                point = z.detach().requires_grad_(True)
                prior = flow(point)
                M = torch.autograd.functional.jacobian(flow, point)
            covariance = M @ P @ M.T + self.Q_cov * torch.eye(
                self.n, device=P.device, dtype=P.dtype
            )
            states.append(prior.detach())
            covariances.append(((covariance + covariance.T) / 2).detach())
        return torch.stack(states), torch.stack(covariances)

    def update(self, z_prior, P_prior, x_observed, decoder_fn):
        self._validate(z_prior, P_prior)
        if (
            x_observed.ndim != 2
            or len(x_observed) != len(z_prior)
            or not torch.isfinite(x_observed).all()
        ):
            raise ValueError("invalid observations")
        states, covariances = [], []
        for z, P, x in zip(z_prior, P_prior, x_observed):
            with torch.enable_grad():
                point = z.detach().requires_grad_(True)
                predicted = decoder_fn(point[None])[0]
                C = torch.autograd.functional.jacobian(lambda p: decoder_fn(p[None])[0], point)
            if predicted.shape != x.shape:
                raise ValueError("decoder and observation dimensions differ")
            R = self.R_cov * torch.eye(len(x), device=P.device, dtype=P.dtype)
            S = C @ P @ C.T + R
            # SPD solve, not an explicitly formed inverse.
            chol = torch.linalg.cholesky((S + S.T) / 2)
            K = torch.cholesky_solve(C @ P, chol).T
            state = z + K @ (x - predicted.detach())
            residual = torch.eye(self.n, device=P.device, dtype=P.dtype) - K @ C
            covariance = residual @ P @ residual.T + K @ R @ K.T  # Joseph form
            states.append(state.detach())
            covariances.append(((covariance + covariance.T) / 2).detach())
        return torch.stack(states), torch.stack(covariances)
