import asyncio
import time
from collections import deque

import numpy as np
import torch

from ..config.system_params import EngineConfig
from ..crystallization.ast_grammar import ASTGrammar
from ..crystallization.dynamic_operator_registry import DynamicOperatorRegistry
from ..crystallization.pareto_compiler import ParetoSymbolicCompiler
from ..dynamics.hamiltonian_nn import HamiltonianNN
from ..dynamics.stability_memory import SymplecticSubspaceProtector
from ..math_core.integrators import ImplicitMidpointIntegrator, Yoshida4thIntegrator
from ..math_core.symplectic import SymplecticManifold
from ..noether.conservation_verifier import ConservationVerifier
from ..noether.symmetry_miner import LieSymmetryMiner
from ..semantic_bridge.bidirectional_translator import BidirectionalTranslator
from ..visualization.phase_space_projector import SymplecticPhaseProjector
from .online_symplectic_filter import OnlineSymplecticStateFilter


class EpistemicContinuousOrchestrator:
    """Bounded, cancellable online training loop. Discoveries are empirical model candidates."""

    def __init__(self, config: EngineConfig, *, separable_gradients=None, canvas=None):
        self.config = config
        self.manifold = SymplecticManifold(config.spatial_dim, config.device, config.dtype)
        self.hamiltonian = HamiltonianNN(self.manifold)
        self.miner = LieSymmetryMiner(self.manifold)
        self.protector = SymplecticSubspaceProtector(
            sum(p.numel() for p in self.hamiltonian.parameters())
        )
        self.h_optimizer = torch.optim.SGD(self.hamiltonian.parameters(), lr=config.learning_rate)
        self.q_optimizer = torch.optim.Adam(self.miner.parameters(), lr=config.learning_rate)
        self.filter = OnlineSymplecticStateFilter(config.spatial_dim, config.integrator_dt)
        self.registry = DynamicOperatorRegistry(ASTGrammar(2 * config.spatial_dim))
        self.compiler = ParetoSymbolicCompiler(
            self.registry.grammar, mdl_weight=config.pareto_mdl_weight
        )
        self.verifier = ConservationVerifier(self.manifold, config.poisson_threshold)
        self.integrator = ImplicitMidpointIntegrator(
            config.max_fixed_point_iters, config.integrator_tol
        )
        self.separable_gradients = separable_gradients
        if config.integrator_type == "yoshida4" and separable_gradients is None:
            raise ValueError(
                "Yoshida runtime requires explicit separable_gradients=(grad_V,grad_T)"
            )
        self.canvas = canvas
        self.buffer = deque(maxlen=config.buffer_size)
        self.interventions = asyncio.Queue(maxsize=32)
        self.tasks = []
        self.metrics = {}
        self.discovery_reports = []
        self.discovery_counter = 0
        self.updates = 0
        self.previous = None
        self.rotation_constraint = False
        self.offset = torch.zeros(
            1, 2 * config.spatial_dim, device=config.device, dtype=config.dtype
        )
        self.next_replay = time.monotonic() + config.memory_replay_interval_secs
        self._last_frame = 0.0

    def inject_user_intervention(self, intervention_packet):
        kind = intervention_packet.get("type")
        if kind == "perturbation":
            delta = torch.as_tensor(
                intervention_packet["delta"], device=self.config.device, dtype=self.config.dtype
            )
            if delta.shape != (2 * self.config.spatial_dim,) or not torch.isfinite(delta).all():
                raise ValueError("delta must be a finite phase vector")
        elif kind == "constraint":
            BidirectionalTranslator(None, None, self.registry).prompt_to_geometric_constraint(
                intervention_packet["prompt"]
            )
        else:
            raise ValueError("unknown intervention type")
        self.interventions.put_nowait(dict(intervention_packet))

    def _apply_interventions(self):
        while not self.interventions.empty():
            packet = self.interventions.get_nowait()
            if packet["type"] == "perturbation":
                self.offset += torch.as_tensor(
                    packet["delta"], device=self.offset.device, dtype=self.offset.dtype
                )
                # A discontinuity must not be treated as a dynamical training pair.
                self.previous = None
                self.buffer.clear()
            else:
                self.rotation_constraint = True

    def _learn(self, pairs):
        self.h_optimizer.zero_grad(set_to_none=True)
        loss = self.hamiltonian.trajectory_loss(pairs, self.config.integrator_dt)
        if self.rotation_constraint:
            z = pairs[:, 0].detach().requires_grad_(True)
            d = self.config.spatial_dim
            L = z[:, :1] * z[:, d + 1 : d + 2] - z[:, 1:2] * z[:, d : d + 1]
            loss = loss + self.manifold.poisson_bracket(L, self.hamiltonian(z), z).square().mean()
        loss.backward()
        params = list(self.hamiltonian.parameters())
        flat = torch.cat(
            [(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1) for p in params]
        )
        projected = self.protector.project_gradient(flat)
        offset = 0
        for p in params:
            p.grad = projected[offset : offset + p.numel()].reshape_as(p).clone()
            offset += p.numel()
        self.h_optimizer.step()
        self.q_optimizer.zero_grad(set_to_none=True)
        mining_loss, metrics = self.miner.compute_loss(pairs[:, 0], self.hamiltonian)
        gradients = torch.autograd.grad(mining_loss, self.miner.parameters())
        for p, grad in zip(self.miner.parameters(), gradients):
            p.grad = grad
        self.q_optimizer.step()
        self.metrics = {
            "dynamics_loss": float(loss.detach()),
            "mining_loss": float(mining_loss.detach()),
            **metrics,
        }

    def _try_discovery(self):
        if len(self.buffer) < 32:
            return
        samples = torch.cat(list(self.buffer))
        training, heldout = samples[::2], samples[1::2]
        z = heldout.detach().requires_grad_(True)
        Q = self.miner(z)
        bracket = self.manifold.poisson_bracket(Q, self.hamiltonian(z), z)
        if (
            float(Q.detach().var()) < self.config.min_invariant_variance
            or float(bracket.detach().abs().max()) >= self.config.poisson_threshold
        ):
            return
        trajectory = samples.unsqueeze(0)
        valid, drift = self.verifier.verify_trajectory_invariance(self.miner, trajectory)
        if not valid:
            return

        def target(array):
            with torch.no_grad():
                tensor = torch.as_tensor(array, device=self.config.device, dtype=self.config.dtype)
                return self.miner(tensor).cpu().numpy().reshape(-1)

        node = self.compiler.compile(target, training.cpu().numpy())
        actual = target(heldout.cpu().numpy())
        predicted = node.evaluate(heldout.cpu().numpy())
        variance = np.var(actual)
        r2 = 1 - np.mean((predicted - actual) ** 2) / max(float(variance), 1e-24)
        if not np.isfinite(r2) or r2 < 0.9999 or node in self.registry.grammar.registered.values():
            return
        self.discovery_counter += 1
        name = f"Q_{self.discovery_counter}"
        self.registry.register_crystallized_invariant(node, name)
        self.protector.register_invariant(self.miner, self.hamiltonian, heldout)
        self.discovery_reports.append(
            {
                "name": name,
                "latex": node.to_latex(),
                "empirical_drift": drift,
                "heldout_r2": float(r2),
                "status": "empirical candidate; not a physical proof",
            }
        )
        # Bound metadata and protected subspace growth in a continuous process.
        self.discovery_reports = self.discovery_reports[-32:]
        self.protector.rows = self.protector.rows[-32:]
        if len(self.registry.grammar.registered) > 32:
            self.registry.grammar.registered.pop(next(iter(self.registry.grammar.registered)))

    async def process_observation(self, coordinates):
        self._apply_interventions()
        q = torch.as_tensor(coordinates, device=self.config.device, dtype=self.config.dtype)
        if q.ndim == 1:
            q = q[None]
        if q.shape != (1, self.config.spatial_dim):
            raise ValueError("runtime accepts one [d] or [1,d] sensor observation at a time")
        state = self.filter.update_and_extract_phase_point(q) + self.offset
        # First finite-difference velocity is undefined; skip it for training.
        if self.filter.previous is not None and self.updates == 0 and self.previous is None:
            self.updates += 1
            return state
        if self.previous is not None:
            pair = torch.stack((self.previous, state), 1)
            recent = list(self.buffer)[-31:] + [state.detach()]
            if len(recent) > 1:
                replay = torch.cat(recent)
                self._learn(torch.stack((replay[:-1], replay[1:]), 1))
            else:
                self._learn(pair)
        self.previous = state.detach()
        self.buffer.append(state.detach())
        self.updates += 1
        if self.config.integrator_type == "yoshida4":
            q1, p1 = Yoshida4thIntegrator().step_separable(
                *state.chunk(2, -1), *self.separable_gradients, self.config.integrator_dt
            )
            prediction = torch.cat((q1, p1), -1)
        else:
            prediction = self.integrator.step(
                state, self.hamiltonian.predict_vector_field, self.config.integrator_dt
            )
        if self.updates % self.config.discovery_interval == 0:
            self._try_discovery()
        if time.monotonic() >= self.next_replay:
            states = torch.cat(list(self.buffer))
            if len(states) >= 2:
                self._learn(torch.stack((states[:-1], states[1:]), 1))
            self.next_replay = time.monotonic() + self.config.memory_replay_interval_secs
        if self.canvas is not None and time.monotonic() - self._last_frame >= 1 / 60:
            vertices = SymplecticPhaseProjector(self.config.spatial_dim).project_to_3d(
                state[:, None]
            )
            await self.canvas.broadcast_phase_frame(
                {
                    "vertices": vertices.tolist(),
                    "metrics": self.metrics,
                    "discoveries": self.discovery_reports,
                }
            )
            self._last_frame = time.monotonic()
        return prediction.detach()

    async def _consume(self, source):
        async for coordinates in source:
            await self.process_observation(coordinates)
            await asyncio.sleep(0)

    def start_epistemic_runtime(self, source):
        if any(not task.done() for task in self.tasks):
            raise RuntimeError("runtime already active")
        self.tasks = [asyncio.create_task(self._consume(source), name="cnge-assimilation")]
        return self.tasks[0]

    async def stop(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        results = await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
        for result in results:
            if isinstance(result, Exception):
                raise result
