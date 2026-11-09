import pytest
import torch

from cnge_v2.manifold.canonical_symplectic import SymplecticStructure
from cnge_v2.manifold.variational_autoencoder import SymplecticAutoencoder
from cnge_v2.solvers.implicit_midpoint import ImplicitMidpointSolver
from cnge_v2.solvers.kalman_manifold_filter import SymplecticEKF


def test_autoencoder_jvp_and_hamiltonian_backward():
    model = SymplecticAutoencoder(8, 2, hidden_dim=16).double()
    energy = torch.nn.Linear(4, 1).double()
    observations = torch.randn(16, 2, 8, dtype=torch.float64)
    losses = model.compute_loss(observations, 0.1, energy)
    losses["total_loss"].backward()
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert model.encoder[0].weight.grad.abs().max() > 0
    assert model.decoder[0].weight.grad.abs().max() > 0
    assert energy.weight.grad.abs().max() > 0
    with pytest.raises(ValueError):
        model.compute_loss(observations, 0, energy)


def test_noisy_128d_full_state_observation_denoising(record_property):
    # Identifiable full-state telemetry, explicitly NOT a single position-only pendulum image.
    model = SymplecticAutoencoder(
        128, 1, hidden_dim=32, tangent_weight=0, prediction_weight=0
    ).double()
    projection, _ = torch.linalg.qr(torch.randn(128, 2, dtype=torch.float64))
    state = torch.randn(256, 2, dtype=torch.float64) * 0.5
    clean = state @ projection.T
    noise_scale = 0.15 * clean.std()
    noisy = clean + noise_scale * torch.randn_like(clean)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = float((model(noisy)[1] - clean).square().mean().detach())
    # Independent noisy reconstructions require no privileged physical-state targets in training.
    for _ in range(350):
        optimizer.zero_grad()
        _, reconstructed = model(noisy)
        loss = (reconstructed - noisy).square().mean()
        loss.backward()
        optimizer.step()
    test_state = torch.randn(256, 2, dtype=torch.float64) * 0.5
    test_clean = test_state @ projection.T
    test_noisy = test_clean + noise_scale * torch.randn_like(test_clean)
    with torch.no_grad():
        latent, reconstructed = model(test_noisy)
        r2 = 1 - (reconstructed - test_clean).square().sum() / test_clean.square().sum()
        training_latent = model.encoder(noisy)
        alignment = torch.linalg.lstsq(
            torch.cat((training_latent, torch.ones(256, 1, dtype=latent.dtype)), 1), state
        ).solution
        aligned = torch.cat((latent, torch.ones(256, 1, dtype=latent.dtype)), 1) @ alignment
        aligned_r2 = 1 - (aligned - test_state).square().sum() / test_state.square().sum()
    assert initial > float((reconstructed - test_clean).square().mean()) * 10
    record_property("observation_r2", float(r2))
    record_property("affine_aligned_latent_r2", float(aligned_r2))
    assert r2 > 0.98
    assert aligned_r2 > 0.98  # Diagnostic affine alignment, not canonical-identifiability proof.


def test_position_only_pixels_do_not_identify_momentum():
    model = SymplecticAutoencoder(128, 1)
    same_image = torch.randn(1, 128).expand(2, -1)
    latent, _ = model(same_image)
    assert torch.equal(latent[0], latent[1])
    momenta = torch.tensor([-1.0, 1.0])
    assert (latent[:, 1] - momenta).square().mean() >= 1  # An information-theoretic obstruction.


def test_latent_flow_is_symplectic_but_rectangular_encoder_has_no_canonical_defect():
    structure = SymplecticStructure(2, dtype=torch.float64)
    z = torch.randn(4, dtype=torch.float64, requires_grad=True)
    solver = ImplicitMidpointSolver(tol=1e-14, max_iter=30)

    def flow(point):
        return solver.integrate_step(point[None], lambda x: structure.symplectic_gradient(x), 0.03)[
            0
        ]

    jacobian = torch.autograd.functional.jacobian(flow, z)
    assert structure.pullback_defect(jacobian).norm() < 1e-10
    with pytest.raises(ValueError, match="square"):
        structure.pullback_defect(torch.randn(4, 128, dtype=z.dtype))
    u = torch.randn(16, 4, dtype=z.dtype)
    assert structure.pairing(u, u).abs().max() < 1e-14


def test_ekf_reduces_noise_and_joseph_covariance_is_psd():
    dtype = torch.float64
    structure = SymplecticStructure(1, dtype=dtype)
    ekf = SymplecticEKF(1, Q_cov=1e-6, R_cov=0.15**2)
    solver = ImplicitMidpointSolver()
    true = torch.tensor([[1.0, 0.2]], dtype=dtype)
    state = true.clone()
    P = torch.eye(2, dtype=dtype)[None] * 0.01
    raw_errors, filtered_errors = [], []
    for _ in range(100):
        true = solver.integrate_step(true, lambda z: structure.symplectic_gradient(z), 0.03)
        observed = true + 0.15 * torch.randn_like(true)
        prior, covariance = ekf.predict(state, P, lambda z: structure.symplectic_gradient(z), 0.03)
        state, P = ekf.update(prior, covariance, observed, lambda z: z)
        raw_errors.append((observed - true).square().mean())
        filtered_errors.append((state - true).square().mean())
        assert torch.linalg.eigvalsh(P).min() >= -1e-12
        torch.testing.assert_close(P, P.transpose(-1, -2), atol=1e-12, rtol=1e-12)
    assert torch.stack(filtered_errors).mean() < 0.2 * torch.stack(raw_errors).mean()


def test_tangent_noise_weight_does_not_explode_as_dt_shrinks():
    model = SymplecticAutoencoder(2, 1, hidden_dim=8, observation_noise_variance=0.1).double()
    H = lambda z: z.square().sum(-1, keepdim=True) / 2
    x = torch.randn(8, 2, 2, dtype=torch.float64)
    first = model.compute_loss(x, 0.01, H)
    second = model.compute_loss(x, 0.001, H)
    assert second["total_loss"] < 2 * first["total_loss"]


def test_history_distinguishes_opposite_motion_at_same_position():
    from cnge_v2.execution.observation_history import ObservationHistory

    left = ObservationHistory(1, length=3)
    right = ObservationHistory(1, length=3)
    for a, b in zip((-0.2, -0.1, 0.0), (0.2, 0.1, 0.0)):
        previous = left.update([a])
        current = right.update([b])
    assert previous[-1] == current[-1]
    assert not torch.equal(previous, current)
    assert len(left.frames) == 3
    left.update([0.1])
    assert len(left.frames) == 3
    left.reset()
    assert left.update([0.0]) is None
