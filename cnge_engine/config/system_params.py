import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EngineConfig:
    spatial_dim: int = 2
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    integrator_type: str = "implicit_midpoint"
    integrator_dt: float = 0.01
    integrator_tol: float = 1e-12
    max_fixed_point_iters: int = 20
    poisson_threshold: float = 1e-4
    pareto_mdl_weight: float = 0.05
    memory_replay_interval_secs: float = 3600
    embedding_dim: int = 1024
    learning_rate: float = 1e-3
    buffer_size: int = 256
    discovery_interval: int = 100
    min_invariant_variance: float = 1e-3

    def __post_init__(self):
        for name in (
            "spatial_dim",
            "embedding_dim",
            "max_fixed_point_iters",
            "buffer_size",
            "discovery_interval",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "integrator_dt",
            "integrator_tol",
            "poisson_threshold",
            "learning_rate",
            "memory_replay_interval_secs",
            "min_invariant_variance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.pareto_mdl_weight < 0 or not math.isfinite(self.pareto_mdl_weight):
            raise ValueError("pareto_mdl_weight must be finite and nonnegative")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("use float32 or float64")
        if self.integrator_type not in ("implicit_midpoint", "yoshida4"):
            raise ValueError("unknown integrator")
        if torch.device(self.device).type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
