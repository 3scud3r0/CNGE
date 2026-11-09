import pytest
import torch

from cnge_engine.math_core.integrators import ImplicitMidpointIntegrator, Yoshida4thIntegrator
from cnge_engine.math_core.symplectic import SymplecticManifold, scalar_gradient


def henon_heiles(z):
    q, p = z.chunk(2, -1)
    return (
        (q.square().sum(-1) + p.square().sum(-1)) / 2
        + q[..., 0] ** 2 * q[..., 1]
        - q[..., 1] ** 3 / 3
    )


def double_pendulum(z):
    # Canonical equal-mass/equal-length double pendulum; velocities are NOT momenta.
    q, p = z.chunk(2, -1)
    cosine = torch.cos(q[..., 0] - q[..., 1])
    kinetic = (p[..., 0] ** 2 + 2 * p[..., 1] ** 2 - 2 * p[..., 0] * p[..., 1] * cosine) / (
        2 * (2 - cosine**2)
    )
    return kinetic - 2 * torch.cos(q[..., 0]) - torch.cos(q[..., 1])


@pytest.mark.parametrize("H", [henon_heiles, double_pendulum])
def test_midpoint_nonlinear_symplectic_jacobian(H):
    m = SymplecticManifold(2, dtype=torch.float64)
    z = torch.tensor([[0.1, 0.2, 0.05, -0.1]], dtype=torch.float64, requires_grad=True)
    field = lambda x: m.hamiltonian_vector_field(scalar_gradient(H(x), x))
    step = lambda x: ImplicitMidpointIntegrator(max_iters=40, tol=1e-14).step(x, field, 0.01)
    jac = torch.autograd.functional.jacobian(step, z).reshape(4, 4)
    assert m.check_symplectomorphism(jac, 1e-10)[0]
    torch.testing.assert_close(
        ImplicitMidpointIntegrator(40, 1e-14).step(step(z), field, -0.01), z, atol=1e-12, rtol=1e-12
    )


def test_nonconvergence_and_nonseparable_rejection():
    with pytest.raises(RuntimeError, match="converge"):
        ImplicitMidpointIntegrator(2).step(torch.ones(1, 2), lambda z: 100 * z, 1.0)
    with pytest.raises(ValueError, match="separable"):
        Yoshida4thIntegrator().step(torch.ones(1, 2), lambda z: z, 0.01)


@pytest.mark.parametrize("steps", [200, pytest.param(50000, marks=pytest.mark.long)])
def test_yoshida_henon_heiles_cumulative_tangent(steps):
    # Propagate the full tangent map; checking only individual step Jacobians is insufficient.
    dtype = torch.float64
    m = SymplecticManifold(2, dtype=dtype)
    q = torch.tensor([[0.1, 0.0]], dtype=dtype)
    p = torch.tensor([[0.0, 0.2]], dtype=dtype)
    tangent = torch.eye(4, dtype=dtype)
    integrator = Yoshida4thIntegrator()

    def grad_v(q):
        return torch.stack(
            (q[..., 0] + 2 * q[..., 0] * q[..., 1], q[..., 1] + q[..., 0] ** 2 - q[..., 1] ** 2), -1
        )

    def mapping(z):
        return torch.cat(integrator.step_separable(*z.chunk(2, -1), grad_v, lambda p: p, 0.01), -1)

    z = torch.cat((q, p), -1)
    maximum = 0.0
    for _ in range(steps):
        local = torch.func.jacrev(mapping)(z).reshape(4, 4)
        tangent = local @ tangent
        z = mapping(z)
        error = float(torch.linalg.matrix_norm(tangent.T @ m.J @ tangent - m.J))
        maximum = max(maximum, error)
    assert maximum < 1e-6


@pytest.mark.parametrize("H", [henon_heiles, double_pendulum])
@pytest.mark.long
def test_midpoint_50000_cumulative_tangent(H):
    m = SymplecticManifold(2, dtype=torch.float64)
    z = torch.tensor([[0.1, 0.0, 0.0, 0.2]], dtype=torch.float64)
    tangent = torch.eye(4, dtype=z.dtype)
    solver = ImplicitMidpointIntegrator(40, 1e-14)
    field = torch.func.grad(lambda x: H(x).sum())

    def vector(x):
        return m.hamiltonian_vector_field(field(x))

    for _ in range(50000):
        next_z = solver.step(z, vector, 0.01).detach()
        middle = (z + next_z) / 2
        derivative = torch.func.jacrev(vector)(middle).reshape(4, 4)
        identity = torch.eye(4, dtype=z.dtype)
        local = torch.linalg.solve(identity - 0.005 * derivative, identity + 0.005 * derivative)
        tangent = local @ tangent
        assert torch.linalg.matrix_norm(tangent.T @ m.J @ tangent - m.J) < 1e-6
        z = next_z
