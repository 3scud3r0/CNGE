import pytest
import torch

from cnge_engine.math_core.symplectic import scalar_gradient
from cnge_v2 import V2EngineConfig
from cnge_v2.execution.live_orchestrator import LiveOrchestrator
from cnge_v2.execution.streaming_pipeline import StreamingEpistemicPipeline
from cnge_v2.symmetry_engine.gauge_fixed_noether import DegenerateGaugeError, GaugeFixedNoetherMiner


class Oscillator(torch.nn.Module):
    def forward(self, z):
        return z.square().sum(-1, keepdim=True) / 2


@pytest.mark.parametrize("alias", ["constant", "energy", "energy_squared"])
def test_degenerate_generators_are_rejected(alias):
    class Candidate(torch.nn.Module):
        def forward(self, z):
            energy = Oscillator()(z)
            return (
                torch.ones_like(energy)
                if alias == "constant"
                else energy
                if alias == "energy"
                else energy.square()
            )

    H = Oscillator()
    miner = GaugeFixedNoetherMiner(2).double()
    miner.network = Candidate()
    miner.set_reference(torch.randn(48, 4, dtype=torch.float64), H)
    with pytest.raises(DegenerateGaugeError):
        miner(torch.randn(8, 4, dtype=torch.float64))


def test_normalized_scalar_gauge_exact_reference_conditions_and_backward():
    H = Oscillator()
    miner = GaugeFixedNoetherMiner(2, hidden_dim=16).double()
    reference = torch.randn(64, 4, dtype=torch.float64)
    miner.set_reference(reference, H)
    z = reference.detach().requires_grad_(True)
    Q = miner(z)
    gradient = scalar_gradient(Q, z)
    assert abs(float(gradient.square().sum(-1).mean().detach()) - 1) < 1e-10
    assert abs(float((gradient * z).sum(-1).mean().detach())) < 1e-10
    loss, metrics = miner.compute_orthogonalized_loss(z, H)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in miner.parameters())
    assert metrics["unit_norm_loss"] < 1e-20
    # A Hessian of the actual scalar remains symmetric; no non-integrable pointwise projection.
    hessian = torch.autograd.functional.hessian(
        lambda point: miner(point[None]).sum(), reference[0]
    )
    torch.testing.assert_close(hessian, hessian.T, atol=1e-10, rtol=1e-10)


def calibration_pairs():
    angles = torch.linspace(0, 6, 40, dtype=torch.float64)
    q0 = torch.stack((angles.cos(), angles.sin()), -1)
    q1 = torch.stack((torch.cos(angles + 0.01), torch.sin(angles + 0.01)), -1)
    return torch.stack((q0, q1), 1)


async def test_pipeline_calibration_ingestion_checkpoint_and_real_learning(tmp_path):
    config = V2EngineConfig(
        obs_dim=2, latent_spatial_dim=1, hidden_dim=8, dtype=torch.float64, protection_mode="task"
    )
    pipeline = StreamingEpistemicPipeline(config)
    with pytest.raises(RuntimeError, match="calibrate"):
        await pipeline.ingest_sensor_tick([1.0, 0.0])
    history = pipeline.calibrate(calibration_pairs(), steps=5)
    assert len(history) == 5
    before = [p.detach().clone() for p in pipeline.adapters.active_parameters()]
    for i in range(5):
        telemetry = await pipeline.ingest_sensor_tick(calibration_pairs()[i, 0])
    assert telemetry["covariance_trace"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, pipeline.adapters.active_parameters()))
    assert (
        pipeline.trigger_symbolic_distillation() is None
    )  # d=1 has no independent generic Noether generator.
    pipeline.consolidate_task("first")
    path = tmp_path / "model.pt"
    pipeline.save_checkpoint(path)
    restored = StreamingEpistemicPipeline.load_checkpoint(path)
    assert restored.adapters.task_ids == ["first"]
    samples = pipeline.reference
    with pipeline.adapters.task("first"), restored.adapters.task("first"):
        torch.testing.assert_close(pipeline.hamiltonian(samples), restored.hamiltonian(samples))
    assert all(not p.requires_grad for p in restored.autoencoder.parameters())
    with pytest.raises(RuntimeError, match="chart"):
        restored.calibrate(calibration_pairs(), steps=1)
    messages = []

    async def source():
        for pair in calibration_pairs()[:3]:
            yield pair[0]

    orchestrator = LiveOrchestrator(restored, capacity=2, callback=messages.append)
    await orchestrator.run(source())
    assert len(messages) == 3
    assert not orchestrator._running


async def test_orchestrator_producer_failure_cancels_worker():
    pipeline = StreamingEpistemicPipeline(V2EngineConfig(2, 1, hidden_dim=8))
    pipeline.calibrate(calibration_pairs(), steps=1)

    async def broken():
        yield [1.0, 0.0]
        raise ValueError("sensor disconnected")

    orchestrator = LiveOrchestrator(pipeline, capacity=1)
    with pytest.raises(ExceptionGroup) as caught:
        await orchestrator.run(broken())
    assert any(isinstance(error, ValueError) for error in caught.value.exceptions)
    assert not orchestrator._running


def test_distillation_consolidates_only_validated_scalar():
    from cnge_v2.core_dynamics.hamiltonian_surrogate import HamiltonianSurrogate
    from cnge_v2.core_dynamics.symplectic_lora import SymplecticLoRAEngine
    from cnge_v2.manifold.canonical_symplectic import SymplecticStructure

    class Squared(torch.nn.Module):
        def forward(self, z):
            return z.square()

    class ExactEnergy(HamiltonianSurrogate):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.structure = SymplecticStructure(2, dtype=torch.float64)
            self.network = torch.nn.Sequential(
                Squared(), torch.nn.Linear(4, 1, bias=False, dtype=torch.float64)
            )
            with torch.no_grad():
                self.network[-1].weight.fill_(0.5)

    class Angular(torch.nn.Module):
        def forward(self, z):
            return z[:, :1] * z[:, 3:4] - z[:, 1:2] * z[:, 2:3]

    config = V2EngineConfig(
        4, 2, hidden_dim=8, dtype=torch.float64, stlsq_threshold=0.001, protection_mode="task"
    )
    pipeline = StreamingEpistemicPipeline(config)
    samples = torch.randn(128, 4, dtype=torch.float64)
    pipeline.calibrate(torch.stack((samples, samples), 1), steps=1)
    pipeline.reference = samples
    pipeline.hamiltonian = ExactEnergy()
    pipeline.adapters = SymplecticLoRAEngine(pipeline.hamiltonian, rank=1, protection_mode="task")
    pipeline.miner.network = Angular()
    pipeline.miner.set_reference(samples, pipeline.hamiltonian)
    result = pipeline.trigger_symbolic_distillation()
    assert result is not None
    assert result["r2"] > 0.999999
    assert pipeline.adapters.task_ids == ["discovery_1"]
    assert len(pipeline.discoveries) == 1
    # No consolidation if the variance gate fails; adapter state remains unchanged.
    pipeline.config = V2EngineConfig(
        4, 2, hidden_dim=8, dtype=torch.float64, min_discovery_variance=1e9, protection_mode="task"
    )
    assert pipeline.trigger_symbolic_distillation() is None
    assert pipeline.adapters.task_ids == ["discovery_1"]
