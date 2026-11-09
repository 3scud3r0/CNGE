"""Noise-aware canonical dynamics, sparse distillation and task-aware adapters."""

from .config.engine_config import V2EngineConfig
from .manifold.canonical_symplectic import SymplecticStructure

__all__ = ["SymplecticStructure", "V2EngineConfig"]

__version__ = "0.2.0"
