import math
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class V2EngineConfig:
    obs_dim: int
    latent_spatial_dim: int
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32
    integrator_dt: float = 0.01
    integrator_tol: float = 1e-10
    integrator_max_iter: int = 15
    stlsq_threshold: float = 0.05
    stlsq_max_iter: int = 20
    lora_rank: int = 4
    noether_orth_weight: float = 5.0
    noether_norm_weight: float = 10.0
    learning_rate: float = 1e-3
    buffer_size: int = 256
    hidden_dim: int = 64
    dictionary_degree: int = 3
    dictionary_max_features: int = 2048
    process_covariance: float = 1e-5
    observation_covariance: float = 1e-2
    min_discovery_variance: float = 0.05
    max_commutator: float = 1e-4
    min_discovery_r2: float = 0.999
    protection_mode: str = "activation"

    def __post_init__(self):
        for name in (
            "obs_dim",
            "latent_spatial_dim",
            "integrator_max_iter",
            "stlsq_max_iter",
            "lora_rank",
            "buffer_size",
            "hidden_dim",
            "dictionary_degree",
            "dictionary_max_features",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "integrator_dt",
            "integrator_tol",
            "learning_rate",
            "process_covariance",
            "observation_covariance",
            "min_discovery_variance",
            "max_commutator",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("stlsq_threshold", "noether_orth_weight", "noether_norm_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"invalid {name}")
        if self.dtype not in (torch.float32, torch.float64) or self.protection_mode not in (
            "activation",
            "task",
        ):
            raise ValueError("unsupported dtype or protection mode")
        if not 0 < self.min_discovery_r2 <= 1:
            raise ValueError("min_discovery_r2 must be in (0,1]")
        if torch.device(self.device).type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
