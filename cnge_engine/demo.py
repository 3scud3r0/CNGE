"""Reproducible oscillator -> conservation -> symbolic registry demo, no model downloads."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .crystallization.ast_grammar import ASTGrammar
from .crystallization.dynamic_operator_registry import DynamicOperatorRegistry
from .crystallization.pareto_compiler import ParetoSymbolicCompiler
from .math_core.integrators import Yoshida4thIntegrator
from .math_core.symplectic import SymplecticManifold
from .noether.conservation_verifier import ConservationVerifier


def run_demo(steps=1000, seed=7):
    if steps < 2:
        raise ValueError("steps must be >=2")
    generator = torch.Generator().manual_seed(seed)
    manifold = SymplecticManifold(3, dtype=torch.float64)
    z = torch.randn(8, 6, generator=generator, dtype=torch.float64)
    # Keep normalization away from zero for this relative-drift demonstration.
    z[:, 0], z[:, 1], z[:, 3], z[:, 4] = 1.0, 0.0, 0.0, 1.0
    integrator = Yoshida4thIntegrator()
    history = [z]
    for _ in range(steps):
        q, p = integrator.step_separable(*z.chunk(2, -1), lambda q: q, lambda p: p, 0.01)
        z = torch.cat((q, p), -1)
        history.append(z)

    def angular(t):
        return t[:, :1] * t[:, 4:5] - t[:, 1:2] * t[:, 3:4]

    valid, drift = ConservationVerifier(manifold, tolerance=1e-8).verify_trajectory_invariance(
        angular, torch.stack(history, 1)
    )
    samples = np.random.default_rng(seed).normal(size=(256, 6))
    target = lambda x: x[:, 0] * x[:, 4] - x[:, 1] * x[:, 3]
    grammar = ASTGrammar(6)
    compiler = ParetoSymbolicCompiler(grammar)
    node = compiler.compile(target, samples)
    registry = DynamicOperatorRegistry(grammar)
    registry.register_crystallized_invariant(node, "L_z")
    heldout = np.random.default_rng(seed + 1).normal(size=(256, 6))
    mse = np.mean((node.evaluate(heldout) - target(heldout)) ** 2)
    r2 = 1 - mse / np.var(target(heldout))
    return {
        "system": "3D isotropic harmonic oscillator",
        "steps": steps,
        "conserved": valid,
        "max_relative_drift": drift,
        "latex": node.to_latex(),
        "heldout_r2": float(r2),
        "operator_count": len(grammar.registered),
        "scope": "Analytical reference invariant; this demo does not claim autonomous neural discovery.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_demo(args.steps)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if not report["conserved"] or report["heldout_r2"] < 0.9999:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
