"""Run a noisy-observation v2 pipeline and an independent sparse Kepler benchmark."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import V2EngineConfig
from .execution.live_orchestrator import LiveOrchestrator
from .execution.streaming_pipeline import StreamingEpistemicPipeline
from .symbolic_distiller.constrained_stlsq import ConstrainedSTLSQ
from .symbolic_distiller.functional_dictionary import FunctionalDictionary


def sparse_kepler(seed=9):
    rng = np.random.default_rng(seed)
    library = FunctionalDictionary(
        [f"q_{i}" for i in range(3)] + [f"p_{i}" for i in range(3)], include_kepler=True
    )

    def sample():
        z = rng.normal(size=(512, 6))
        z[:, :3] += [2.0, 0.0, 0.0]
        q, p = z[:, :3], z[:, 3:]
        target = (np.cross(p, np.cross(q, p)) - q / np.linalg.norm(q, axis=1, keepdims=True))[:, 0]
        features, names = library.build_library(torch.from_numpy(z))
        return features.numpy(), target, names

    features, target, names = sample()
    start = time.perf_counter()
    formula, coefficients, r2 = ConstrainedSTLSQ().fit(features, target, names)
    seconds = time.perf_counter() - start
    test_features, test_target, _ = sample()
    test_r2 = 1 - np.mean((test_features @ coefficients - test_target) ** 2) / np.var(test_target)
    return {
        "formula": formula,
        "active_terms": int(np.count_nonzero(coefficients)),
        "training_r2": r2,
        "heldout_r2": float(test_r2),
        "fit_seconds": seconds,
        "scope": "Sparse distillation of analytical Runge-Lenz evaluations, not autonomous discovery.",
    }


async def run_demo(steps=20, calibration_steps=40):
    if steps < 2 or calibration_steps < 1:
        raise ValueError("steps >=2 and calibration_steps >=1 required")
    torch.manual_seed(12)
    torch.set_num_threads(1)
    config = V2EngineConfig(
        obs_dim=8,
        latent_spatial_dim=1,
        hidden_dim=16,
        dtype=torch.float64,
        device="cpu",
        protection_mode="task",
        observation_covariance=0.015**2,
    )
    pipeline = StreamingEpistemicPipeline(config)
    projection, _ = torch.linalg.qr(torch.randn(8, 2, dtype=config.dtype))
    phase = torch.linspace(0, 8, 64, dtype=config.dtype)

    def observe(angle):
        return torch.stack((angle.cos(), angle.sin()), -1) @ projection.T

    pairs = torch.stack((observe(phase), observe(phase + 0.01)), 1)
    pairs += 0.015 * torch.randn_like(pairs)
    history = pipeline.calibrate(pairs, steps=calibration_steps)
    telemetry = []

    async def source():
        for i in range(steps):
            yield observe(torch.tensor(i * 0.01, dtype=config.dtype)) + 0.015 * torch.randn(
                config.obs_dim, dtype=config.dtype
            )

    await LiveOrchestrator(pipeline, callback=telemetry.append).run(source())
    return {
        "calibration_initial_loss": history[0]["total_loss"],
        "calibration_final_loss": history[-1]["total_loss"],
        "ticks": len(telemetry),
        "last_tick": telemetry[-1],
        "kepler": sparse_kepler(),
        "scope": "Differential coordinate learning + EKF + online adapters. No claim of uniquely recovered physical canonical coordinates.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--calibration-steps", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(run_demo(args.steps, args.calibration_steps))
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if report["kepler"]["heldout_r2"] < 0.999 or report["kepler"]["active_terms"] >= 6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
