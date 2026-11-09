import asyncio
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..config.engine_config import V2EngineConfig
from ..core_dynamics.hamiltonian_surrogate import HamiltonianSurrogate
from ..core_dynamics.symplectic_lora import SymplecticLoRAEngine
from ..manifold.variational_autoencoder import SymplecticAutoencoder
from ..solvers.implicit_midpoint import ImplicitMidpointSolver
from ..solvers.kalman_manifold_filter import SymplecticEKF
from ..symbolic_distiller.constrained_stlsq import ConstrainedSTLSQ
from ..symbolic_distiller.functional_dictionary import FunctionalDictionary
from ..symmetry_engine.gauge_fixed_noether import DegenerateGaugeError, GaugeFixedNoetherMiner


class StreamingEpistemicPipeline:
    """Single-worker, serialized calibration/EKF/adapter/mining/distillation lifecycle."""

    def __init__(self, config: V2EngineConfig):
        self.config = config
        self.autoencoder = SymplecticAutoencoder(
            config.obs_dim,
            config.latent_spatial_dim,
            config.hidden_dim,
            observation_noise_variance=config.observation_covariance,
        ).to(device=config.device, dtype=config.dtype)
        self.hamiltonian = HamiltonianSurrogate(config.latent_spatial_dim, config.hidden_dim).to(
            device=config.device, dtype=config.dtype
        )
        self.adapters = SymplecticLoRAEngine(
            self.hamiltonian, config.lora_rank, config.protection_mode
        )
        self.miner = GaugeFixedNoetherMiner(
            config.latent_spatial_dim,
            config.hidden_dim,
            config.noether_orth_weight,
            config.noether_norm_weight,
        ).to(device=config.device, dtype=config.dtype)
        self.solver = ImplicitMidpointSolver(config.integrator_tol, config.integrator_max_iter)
        self.autoencoder.solver = self.solver
        self.filter = SymplecticEKF(
            config.latent_spatial_dim,
            config.process_covariance,
            config.observation_covariance,
            solver=self.solver,
        )
        self.optimizer = torch.optim.Adam(
            self.adapters.active_parameters(), lr=config.learning_rate
        )
        self.miner_optimizer = torch.optim.Adam(self.miner.parameters(), lr=config.learning_rate)
        d = config.latent_spatial_dim
        self.dictionary = FunctionalDictionary(
            [f"q_{i}" for i in range(d)] + [f"p_{i}" for i in range(d)],
            config.dictionary_degree,
            max_features=config.dictionary_max_features,
        )
        self.distiller = ConstrainedSTLSQ(config.stlsq_threshold, max_iter=config.stlsq_max_iter)
        self.buffer = deque(maxlen=config.buffer_size)
        self.reference = None
        self.posterior = self.covariance = self.previous = None
        self.ready = False
        self.last_metrics = {}
        self.discoveries = []
        self._lock = asyncio.Lock()
        self._sequence = 0

    def calibrate(self, x_seq, steps=100):
        if self.adapters.task_ids:
            raise RuntimeError("cannot change the shared coordinate chart after freezing a task")
        samples = torch.as_tensor(x_seq, device=self.config.device, dtype=self.config.dtype)
        if (
            samples.ndim != 3
            or samples.shape[1:] != (2, self.config.obs_dim)
            or len(samples) < 8
            or not torch.isfinite(samples).all()
            or steps < 1
        ):
            raise ValueError(
                "calibration needs at least 8 finite observation pairs [B,2,D] and steps>=1"
            )
        self.autoencoder.requires_grad_(True)
        optimizer = torch.optim.Adam(
            [*self.autoencoder.parameters(), *self.adapters.active_parameters()],
            lr=self.config.learning_rate,
        )
        history = []
        try:
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                losses = self.autoencoder.compute_loss(
                    samples, self.config.integrator_dt, self.hamiltonian.energy
                )
                if not all(torch.isfinite(value) for value in losses.values()):
                    raise RuntimeError("nonfinite calibration loss")
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [*self.autoencoder.parameters(), *self.adapters.active_parameters()],
                    10.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                history.append({key: float(value.detach()) for key, value in losses.items()})
        finally:
            self.autoencoder.requires_grad_(False)
        with torch.no_grad():
            self.reference = self.autoencoder.encoder(samples[:, 0]).detach()
        self.miner.set_reference(self.reference, self.hamiltonian)
        self.optimizer = torch.optim.Adam(
            self.adapters.active_parameters(), lr=self.config.learning_rate
        )
        self.reset_stream()
        self.ready = True
        return history

    def reset_stream(self):
        self.posterior = self.covariance = self.previous = None
        self.buffer.clear()

    def _learn(self):
        self.adapters.assert_plastic_capacity()
        if len(self.buffer) < 2:
            return {}
        recent = list(self.buffer)[-32:]
        z0 = torch.cat([entry[0] for entry in recent[:-1]])
        z1 = torch.cat([entry[0] for entry in recent[1:]])
        self.optimizer.zero_grad(set_to_none=True)
        predicted = self.solver.integrate_step(
            z0, self.hamiltonian.vector_field, self.config.integrator_dt
        )
        # Integrated latent and decoded-observation residuals; no q finite-difference momentum.
        observed = torch.cat([entry[1] for entry in recent[1:]])
        dynamics = (predicted - z1).square().mean() + (
            self.autoencoder.decoder(predicted) - observed
        ).square().mean()
        if not torch.isfinite(dynamics):
            raise RuntimeError("nonfinite dynamics loss")
        dynamics.backward()
        torch.nn.utils.clip_grad_norm_(
            self.adapters.active_parameters(), 10.0, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.miner_optimizer.zero_grad(set_to_none=True)
        flags = [(p, p.requires_grad) for p in self.adapters.active_parameters()]
        try:
            for p, _ in flags:
                p.requires_grad_(False)
            loss, metrics = self.miner.compute_orthogonalized_loss(self.reference, self.hamiltonian)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite Noether loss")
            gradients = torch.autograd.grad(loss, tuple(self.miner.parameters()))
            for p, gradient in zip(self.miner.parameters(), gradients):
                p.grad = gradient
            torch.nn.utils.clip_grad_norm_(self.miner.parameters(), 10.0, error_if_nonfinite=True)
            self.miner_optimizer.step()
            del gradients, loss
        except DegenerateGaugeError as error:
            metrics = {"gauge_rejection": str(error)}
        finally:
            for p, flag in flags:
                p.requires_grad_(flag)
        return {"dynamics_loss": float(dynamics.detach()), **metrics}

    async def ingest_sensor_tick(self, x_raw):
        async with self._lock:
            if not self.ready:
                raise RuntimeError("calibrate(x_seq) is required before ingesting observations")
            x = torch.as_tensor(x_raw, device=self.config.device, dtype=self.config.dtype)
            if x.ndim == 1:
                x = x[None]
            if x.shape != (1, self.config.obs_dim) or not torch.isfinite(x).all():
                raise ValueError("one finite raw observation [obs_dim] is required")
            with torch.no_grad():
                encoded = self.autoencoder.encoder(x)
            if self.posterior is None:
                prior = encoded
                covariance = torch.eye(
                    encoded.shape[-1], device=encoded.device, dtype=encoded.dtype
                )[None]
            else:
                prior, covariance = self.filter.predict(
                    self.posterior,
                    self.covariance,
                    self.hamiltonian.vector_field,
                    self.config.integrator_dt,
                )
            posterior, covariance = self.filter.update(
                prior, covariance, x, self.autoencoder.decoder
            )
            self.posterior, self.covariance = posterior, covariance
            self.buffer.append((posterior.detach(), x.detach()))
            self.last_metrics = self._learn()
            with torch.no_grad():
                energy = float(self.hamiltonian.energy(posterior).item())
            self._sequence += 1
            return {
                "sequence": self._sequence,
                "phase_coordinates": posterior[0].cpu().tolist(),
                "energy": energy,
                "metrics": self.last_metrics,
                "covariance_trace": float(torch.diagonal(covariance, dim1=-2, dim2=-1).sum()),
            }

    def consolidate_task(self, task_id):
        if not self.ready:
            raise RuntimeError("pipeline is not calibrated")
        subspaces = self.adapters.capture_activation_subspaces(self.reference)
        self.adapters.register_frozen_invariant_adapter(
            task_id, self.adapters.active_adapter_state(), subspaces
        )
        # Reset Adam moments when starting a new adapter; momentum must not leak across tasks.
        self.optimizer = torch.optim.Adam(
            self.adapters.active_parameters(), lr=self.config.learning_rate
        )
        self.reset_stream()

    def trigger_symbolic_distillation(self):
        if not self.ready or len(self.reference) < 32:
            return None
        training, heldout = self.reference[::2], self.reference[1::2]
        try:
            z = heldout.detach().requires_grad_(True)
            values = self.miner(z)
            bracket = self.hamiltonian.structure.poisson_bracket(values, self.hamiltonian(z), z)
            if (
                float(values.detach().var(unbiased=False)) <= self.config.min_discovery_variance
                or float(bracket.detach().square().mean()) >= self.config.max_commutator
            ):
                return None
            with torch.enable_grad():
                target = self.miner(training).detach().cpu().numpy()
            features, names = self.dictionary.build_library(training)
            constraints = (
                self.dictionary.conservation_constraints(training, self.hamiltonian.vector_field)
                .detach()
                .cpu()
                .numpy()
            )
            formula, coefficients, train_r2 = self.distiller.fit(
                features.detach().cpu().numpy(), target, names, constraints=constraints
            )
            heldout_features, _ = self.dictionary.build_library(heldout)
            predictions = heldout_features.detach().cpu().numpy() @ coefficients
            actual = values.detach().cpu().numpy().reshape(-1)
            r2 = 1 - np.mean((predictions - actual) ** 2) / max(float(np.var(actual)), 1e-24)
            if not np.isfinite(r2) or r2 < self.config.min_discovery_r2:
                return None
            # Check the synthesized scalar's own Poisson residual, not only the neural target.
            gradients, _ = self.dictionary.gradient_library(heldout)
            compiled_gradient = torch.einsum(
                "bmn,m->bn",
                gradients,
                torch.as_tensor(coefficients, device=z.device, dtype=z.dtype),
            )
            residual = (compiled_gradient * self.hamiltonian.vector_field(heldout)).sum(-1)
            if float(residual.detach().square().mean()) >= self.config.max_commutator:
                return None
        except DegenerateGaugeError:
            return None
        task_id = f"discovery_{len(self.adapters.task_ids) + 1}"
        report = {
            "task_id": task_id,
            "latex": formula,
            "r2": float(r2),
            "training_r2": train_r2,
            "coefficients": coefficients.tolist(),
            "features": names,
            "status": "empirically validated candidate on a finite latent reference set",
        }
        self.consolidate_task(task_id)
        self.discoveries.append(report)
        return report

    def save_checkpoint(self, path):
        if not self.ready:
            raise RuntimeError("cannot save an uncalibrated pipeline")
        config = asdict(self.config)
        config["dtype"] = str(self.config.dtype).split(".")[-1]
        payload = {
            "version": 2,
            "config": config,
            "autoencoder": self.autoencoder.state_dict(),
            "adapters": self.adapters.state_dict(),
            "miner_network": self.miner.network.state_dict(),
            "reference": self.reference,
            "optimizer": self.optimizer.state_dict(),
            "miner_optimizer": self.miner_optimizer.state_dict(),
            "discoveries": self.discoveries,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)

    @classmethod
    def load_checkpoint(cls, path, *, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload["version"] != 2:
            raise ValueError("unsupported checkpoint")
        config = payload["config"]
        config["device"] = device
        config["dtype"] = {"float32": torch.float32, "float64": torch.float64}[config["dtype"]]
        pipeline = cls(V2EngineConfig(**config))
        pipeline.autoencoder.load_state_dict(payload["autoencoder"])
        pipeline.autoencoder.requires_grad_(False)
        pipeline.adapters.load_state_dict(payload["adapters"])
        pipeline.miner.network.load_state_dict(payload["miner_network"])
        pipeline.reference = payload["reference"]
        pipeline.miner.set_reference(pipeline.reference, pipeline.hamiltonian)
        pipeline.optimizer.load_state_dict(payload["optimizer"])
        pipeline.miner_optimizer.load_state_dict(payload["miner_optimizer"])
        pipeline.discoveries = payload["discoveries"]
        pipeline.ready = True
        return pipeline
