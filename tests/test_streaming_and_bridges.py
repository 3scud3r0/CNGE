import asyncio
import json

import numpy as np
import pytest
import torch
from websockets.asyncio.client import connect

from cnge_engine import EngineConfig, SymplecticManifold
from cnge_engine.crystallization.ast_grammar import ASTGrammar, ASTNode
from cnge_engine.crystallization.dynamic_operator_registry import DynamicOperatorRegistry
from cnge_engine.crystallization.topology_analyzer import PhaseTopologyAnalyzer
from cnge_engine.semantic_bridge.bidirectional_translator import BidirectionalTranslator
from cnge_engine.semantic_bridge.foundation_embedding import FoundationEmbeddingBridge
from cnge_engine.semantic_bridge.manifold_transport import ManifoldOptimalTransportBridge
from cnge_engine.streaming.asynchronous_orchestrator import EpistemicContinuousOrchestrator
from cnge_engine.streaming.camera_sensor_feed import ContinuousSensorStream
from cnge_engine.streaming.online_symplectic_filter import OnlineSymplecticStateFilter
from cnge_engine.visualization.interactive_canvas_bridge import InteractiveCanvasServer
from cnge_engine.visualization.lie_orbit_renderer import LieOrbitVisualizer
from cnge_engine.visualization.phase_space_projector import SymplecticPhaseProjector


def test_filter_stream_and_projector():
    stream = ContinuousSensorStream("telemetry", capacity=3)
    filt = OnlineSymplecticStateFilter(2, 0.1, mass=2)
    for i in range(5):
        stream.push([i, 2 * i])
        state = filt.update_and_extract_phase_point(torch.tensor([[i, 2 * i]], dtype=torch.float64))
    assert stream.get_state_buffer(3).shape == (3, 2)
    torch.testing.assert_close(state[:, 2:], torch.tensor([[20.0, 40.0]], dtype=torch.float64))
    assert SymplecticPhaseProjector(2).project_to_3d(state[:, None]).shape == (1, 1, 3)
    with pytest.raises(ValueError):
        filt.update_and_extract_phase_point(torch.ones(2, 2))


def test_poincare_correlation_and_metric():
    m = SymplecticManifold(1, dtype=torch.float64)
    analyzer = PhaseTopologyAnalyzer(m)
    t = torch.linspace(0, 12 * torch.pi, 1000, dtype=torch.float64)
    trajectory = torch.stack((torch.sin(t), torch.cos(t)), -1)
    intersections = analyzer.compute_poincare_map(trajectory, lambda z: z[:, 0], lambda z: z[:, 1:])
    assert intersections.shape == (5, 1)
    assert (intersections - 1).abs().max() < 0.001
    assert 0.8 < analyzer.detect_attractor_dimensionality(trajectory) < 1.3
    metric = analyzer.pullback_metric(lambda x: 2 * x, trajectory[:2])
    torch.testing.assert_close(metric, 4 * torch.eye(2, dtype=t.dtype).expand(2, 2, 2))


def test_semantic_bridge_with_injected_backend():
    class Backend:
        def encode(self, texts, **kwargs):
            return torch.eye(len(texts), 8)

    embedder = FoundationEmbeddingBridge(backend=Backend())
    embeddings = embedder.embed_text(["energy", "angular momentum"])
    assert embeddings.shape == (2, 8)
    transport = ManifoldOptimalTransportBridge(4, 8)
    loss = transport.alignment_loss(torch.randn(2, 4), embeddings)
    loss.backward()
    assert transport.projection[0].weight.grad.abs().max() > 0
    registry = DynamicOperatorRegistry(ASTGrammar(4))
    translator = BidirectionalTranslator(embedder, transport, registry)
    constraint = translator.prompt_to_geometric_constraint(
        "Assume rotational symmetry in the xy-plane"
    )
    m = SymplecticManifold(2)
    assert (constraint.generator.T @ m.J + m.J @ constraint.generator).abs().max() == 0
    assert translator.discovery_to_natural_explanation(ASTNode("var", value=0))["proof"] is None
    with pytest.raises(ValueError):
        translator.prompt_to_geometric_constraint("invent an arbitrary force")


async def test_websocket_roundtrip_and_cleanup():
    server = await InteractiveCanvasServer(port=0).start()
    event = asyncio.Event()
    packets = []

    def callback(packet):
        packets.append(packet)
        event.set()

    server.on_user_interaction(callback)
    try:
        async with connect(f"ws://127.0.0.1:{server.port}") as socket:
            await socket.send(json.dumps({"type": "perturbation", "delta": [1, 0]}))
            await asyncio.wait_for(event.wait(), 2)
            assert packets[0]["delta"] == [1, 0]
            await server.broadcast_phase_frame({"vertices": [[1, 2, 3]]})
            assert json.loads(await asyncio.wait_for(socket.recv(), 2))["vertices"] == [[1, 2, 3]]
            await socket.send("invalid json")
            assert "error" in json.loads(await socket.recv())
    finally:
        await server.close()
    assert not server.clients


async def test_runtime_learning_interventions_and_cancellation():
    config = EngineConfig(
        spatial_dim=2, dtype=torch.float64, discovery_interval=4, memory_replay_interval_secs=0.001
    )
    runtime = EpistemicContinuousOrchestrator(config)
    before = next(runtime.hamiltonian.parameters()).detach().clone()

    async def source():
        for i in range(10):
            yield [np.cos(i * 0.01), np.sin(i * 0.01)]

    task = runtime.start_epistemic_runtime(source())
    await task
    assert runtime.updates == 10
    assert not torch.equal(before, next(runtime.hamiltonian.parameters()))
    assert np.isfinite(runtime.metrics["dynamics_loss"])
    runtime.inject_user_intervention({"type": "constraint", "prompt": "rotational symmetry xy"})
    runtime.inject_user_intervention({"type": "perturbation", "delta": [1, 0, 0, 0]})
    await runtime.process_observation([1.0, 0.0])
    assert runtime.rotation_constraint

    async def infinite():
        while True:
            yield [1.0, 0.0]
            await asyncio.sleep(0.01)

    task = runtime.start_epistemic_runtime(infinite())
    await asyncio.sleep(0.02)
    await runtime.stop()
    assert task.done()


def test_orbit_generator():
    m = SymplecticManifold(1, dtype=torch.float64)
    result = LieOrbitVisualizer(m, steps=4).generate_streamlines(
        lambda z: z.square().sum(-1, keepdim=True) / 2, (-1, 1), grid_res=4
    )
    assert result["vertices"].shape == (16, 5, 3)
    assert np.isfinite(result["vertices"]).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spatial_dim": 0},
        {"integrator_dt": -1},
        {"dtype": torch.int64},
        {"integrator_type": "euler"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        EngineConfig(**kwargs)


async def test_runtime_yoshida_and_error_propagation():
    with pytest.raises(ValueError, match="separable"):
        EpistemicContinuousOrchestrator(EngineConfig(integrator_type="yoshida4"))
    engine = EpistemicContinuousOrchestrator(
        EngineConfig(spatial_dim=1, integrator_type="yoshida4"),
        separable_gradients=(lambda q: q, lambda p: p),
    )
    await engine.process_observation([1.0])
    result = await engine.process_observation([0.99995])
    assert result.shape == (1, 2)

    async def broken():
        yield [1.0, 2.0]  # invalid dimension for d=1

    task = engine.start_epistemic_runtime(broken())
    with pytest.raises(ValueError, match="sensor observation"):
        await task
    with pytest.raises(ValueError, match="sensor observation"):
        await engine.stop()


async def test_slow_client_frame_queue_is_bounded():
    server = InteractiveCanvasServer()
    queue = asyncio.Queue(maxsize=1)
    server.clients["fake-client"] = queue
    for i in range(100):
        await server.broadcast_phase_frame({"index": i})
    assert queue.qsize() == 1
    assert json.loads(queue.get_nowait())["index"] == 99
