import math
from abc import ABC, abstractmethod

import torch


class SymplecticIntegrator(ABC):
    @abstractmethod
    def step(self, z_t, vector_field_fn, dt):
        raise NotImplementedError


def _validate_dt(dt):
    if not math.isfinite(dt):
        raise ValueError("dt must be finite")


class ImplicitMidpointIntegrator(SymplecticIntegrator):
    """Differentiable fixed-point solve; raises instead of accepting nonconvergence."""

    def __init__(self, max_iters=20, tol=1e-12):
        if max_iters < 1 or tol <= 0:
            raise ValueError("invalid solver controls")
        self.max_iters, self.tol = max_iters, tol

    def step(self, z_t, vector_field_fn, dt):
        _validate_dt(dt)
        estimate = z_t
        for _ in range(self.max_iters):
            updated = z_t + dt * vector_field_fn((z_t + estimate) / 2)
            residual = (updated - estimate).detach().abs().max()
            if not torch.isfinite(updated).all():
                raise RuntimeError("nonfinite midpoint iterate")
            # A precision floor is necessary for float32; float64 retains the requested tolerance.
            threshold = max(
                self.tol,
                4 * torch.finfo(z_t.dtype).eps * max(1.0, float(updated.detach().abs().max())),
            )
            estimate = updated
            if residual <= threshold:
                return estimate
        raise RuntimeError("implicit midpoint did not converge; reduce dt or increase max_iters")


class Yoshida4thIntegrator(SymplecticIntegrator):
    """Fourth order composition for explicitly separable H=T(p)+V(q)."""

    def step(self, z_t, vector_field_fn, dt):
        raise ValueError("Yoshida requires explicit grad_V_fn and grad_T_fn; use step_separable")

    def step_separable(self, q_t, p_t, grad_V_fn, grad_T_fn, dt):
        _validate_dt(dt)
        w1 = 1 / (2 - 2 ** (1 / 3))
        w0 = -(2 ** (1 / 3)) * w1
        q, p = q_t, p_t
        for weight in (w1, w0, w1):
            p = p - dt * weight / 2 * grad_V_fn(q)
            q = q + dt * weight * grad_T_fn(p)
            p = p - dt * weight / 2 * grad_V_fn(q)
        return q, p
