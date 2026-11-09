import io

import pytest
import torch

from cnge_v2.core_dynamics.hamiltonian_surrogate import HamiltonianSurrogate
from cnge_v2.core_dynamics.symplectic_lora import ActivationCapacityError, SymplecticLoRAEngine
from cnge_v2.solvers.implicit_midpoint import ImplicitMidpointSolver


def test_overlapping_activation_subspaces_use_union_not_sum():
    engine = SymplecticLoRAEngine(torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False)), rank=2)
    layer = engine.layers["0"]
    basis = torch.eye(4)[:, :1]
    engine.register_frozen_invariant_adapter("first", engine.active_adapter_state(), basis)
    engine.register_frozen_invariant_adapter("second", engine.active_adapter_state(), basis)
    assert layer.U.shape == (4, 1)
    raw = torch.randn(2, 4)
    projected = engine.get_orthogonalized_gradient("0", raw)
    assert (projected @ basis).abs().max() < 1e-6
    with torch.no_grad():
        layer.A.fill_(1)
        layer.B.fill_(1)
    x = torch.eye(4)[:1]
    with engine.task("second"):
        previous = engine(x)
    torch.testing.assert_close(engine(x), previous)
    assert (engine(torch.eye(4)[1:2]) - layer.base(torch.eye(4)[1:2])).abs().max() > 0


def test_capacity_exhaustion_is_explicit():
    engine = SymplecticLoRAEngine(torch.nn.Sequential(torch.nn.Linear(2, 2)), rank=2)
    engine.register_frozen_invariant_adapter("full", engine.active_adapter_state(), torch.eye(2))
    with pytest.raises(ActivationCapacityError):
        engine.assert_plastic_capacity()


def test_task_routed_sequential_dynamics_learning_and_lz_retention(record_property):
    class PolynomialEnergy(HamiltonianSurrogate):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.linear = torch.nn.Linear(3, 1, bias=False, dtype=torch.float64)
            with torch.no_grad():
                self.linear.weight[:] = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
            from cnge_v2.manifold.canonical_symplectic import SymplecticStructure

            self.structure = SymplecticStructure(2, dtype=torch.float64)

        def energy(self, z):
            q, p = z.chunk(2, -1)
            features = torch.stack(
                (
                    q.square().sum(-1) / 2,
                    q.pow(4).sum(-1),
                    q[:, 0] ** 2 * q[:, 1] - q[:, 1] ** 3 / 3,
                ),
                -1,
            )
            return p.square().sum(-1, keepdim=True) / 2 + self.linear(features)

    model = PolynomialEnergy()
    engine = SymplecticLoRAEngine(model, rank=1, protection_mode="task")
    z = torch.randn(128, 4, dtype=torch.float64) * 0.3
    baseline = engine(z).detach().clone()
    engine.register_frozen_invariant_adapter(
        "oscillator", engine.active_adapter_state(), engine.capture_activation_subspaces(z)
    )
    for name, coefficients in [("quartic", [0.0, 0.2, 0.0]), ("henon_heiles", [1.0, 0.0, 0.15])]:
        optimizer = torch.optim.Adam(engine.active_parameters(), lr=0.04)
        desired = torch.tensor(coefficients, dtype=z.dtype)
        q, p = z.chunk(2, -1)
        features = torch.stack(
            (q.square().sum(-1) / 2, q.pow(4).sum(-1), q[:, 0] ** 2 * q[:, 1] - q[:, 1] ** 3 / 3),
            -1,
        )
        target = p.square().sum(-1, keepdim=True) / 2 + (features @ desired)[:, None]
        initial = (engine(z) - target).square().mean().item()
        for _ in range(250):
            optimizer.zero_grad()
            loss = (engine(z) - target).square().mean()
            loss.backward()
            optimizer.step()
        assert (engine(z) - target).square().mean().item() < initial * 0.01
        with engine.task("oscillator"):
            assert torch.equal(engine(z), baseline)
        engine.register_frozen_invariant_adapter(
            name, engine.active_adapter_state(), engine.capture_activation_subspaces(z)
        )
    point = torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=z.dtype)
    solver = ImplicitMidpointSolver(tol=1e-13, max_iter=30)
    max_drift = 0.0
    with engine.task("oscillator"):
        for _ in range(500):
            point = solver.integrate_step(point, model.vector_field, 0.01).detach()
            L = point[:, 0] * point[:, 3] - point[:, 1] * point[:, 2]
            max_drift = max(max_drift, abs(float(L) - 1))
            assert abs(float(L) - 1) < 1e-4
    # Explicitly task-conditioned retention; the active quartic/HH fields need not conserve Lz.
    record_property("routed_oscillator_max_lz_drift", max_drift)
    assert len(engine.task_ids) == 3


def test_adapter_state_dict_roundtrip_and_transactional_validation():
    engine = SymplecticLoRAEngine(torch.nn.Sequential(torch.nn.Linear(4, 3)), rank=2)
    x = torch.randn(20, 4)
    with torch.no_grad():
        engine.layers["0"].B.normal_()
    before = engine(x).detach()
    engine.register_frozen_invariant_adapter(
        "saved", engine.active_adapter_state(), torch.eye(4)[:, :2]
    )
    storage = io.BytesIO()
    torch.save(engine.state_dict(), storage)
    storage.seek(0)
    restored = SymplecticLoRAEngine(torch.nn.Sequential(torch.nn.Linear(4, 3)), rank=2)
    restored.load_state_dict(torch.load(storage, weights_only=True))
    with restored.task("saved"):
        torch.testing.assert_close(restored(x), before)
    with pytest.raises(ValueError):
        restored.register_frozen_invariant_adapter(
            "bad", restored.active_adapter_state(), torch.randn(6, 2)
        )
    assert restored.task_ids == ["saved"]
