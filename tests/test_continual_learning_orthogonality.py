import torch

from cnge_engine.dynamics.hamiltonian_nn import HamiltonianNN
from cnge_engine.dynamics.lagrangian_nn import LagrangianNN
from cnge_engine.dynamics.stability_memory import SymplecticSubspaceProtector
from cnge_engine.math_core.symplectic import SymplecticManifold


def test_exact_and_damped_projection_with_dependent_constraints():
    gradient = torch.randn(20, dtype=torch.float64)
    A = torch.randn(3, 20, dtype=torch.float64)
    A[2] = A[0]
    exact = SymplecticSubspaceProtector(20, damping=0)
    exact.rows = list(A)
    projected = exact.project_gradient(gradient)
    assert (A @ projected).abs().max() < 1e-12
    torch.testing.assert_close(exact.project_gradient(projected), projected)
    damped = SymplecticSubspaceProtector(20)
    damped.rows = list(A)
    assert torch.linalg.norm(A @ damped.project_gradient(gradient)) < torch.linalg.norm(
        A @ gradient
    )


def test_invariant_registration_sensitivity():
    m = SymplecticManifold(2, dtype=torch.float64)
    H = HamiltonianNN(m, (8,))
    protector = SymplecticSubspaceProtector(sum(p.numel() for p in H.parameters()), damping=0)
    Q = lambda z: z[:, :1] * z[:, 3:4] - z[:, 1:2] * z[:, 2:3]
    protector.register_invariant(Q, H, torch.randn(32, 4, dtype=torch.float64))
    assert len(protector.rows) == 1
    g = torch.randn(protector.parameter_dim, dtype=torch.float64)
    assert (protector.rows[0] @ protector.project_gradient(g)).abs() < 1e-12


def test_hamiltonian_learning_backpropagates():
    m = SymplecticManifold(1, dtype=torch.float64)
    H = HamiltonianNN(m, (16,))
    z = torch.randn(64, 2, dtype=torch.float64)
    dt = 0.01
    field = z @ m.J.T
    pairs = torch.stack((z - dt * field / 2, z + dt * field / 2), 1)
    optimizer = torch.optim.Adam(H.parameters(), lr=0.01)
    initial = float(H.trajectory_loss(pairs, dt).detach())
    for _ in range(60):
        optimizer.zero_grad()
        loss = H.trajectory_loss(pairs, dt)
        loss.backward()
        optimizer.step()
    assert float(H.trajectory_loss(pairs, dt).detach()) < initial * 0.5
    H.eval()
    with torch.no_grad():
        assert H.predict_vector_field(z).shape == z.shape


def test_lagrangian_acceleration_and_parameter_gradients():
    class Oscillator(LagrangianNN):
        def forward(self, q, q_dot):
            return (q_dot.square() - q.square()).sum(-1, keepdim=True) / 2

    q = torch.randn(4, 2, dtype=torch.float64)
    v = torch.randn_like(q)
    torch.testing.assert_close(Oscillator(2).compute_acceleration(q, v), -q)
    learned = LagrangianNN(2, (8,)).double()
    a = learned.compute_acceleration(q, v)
    a.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in learned.parameters())
