import numpy as np
import pytest
import torch

from cnge_engine.crystallization.ast_grammar import ASTGrammar, ASTNode
from cnge_engine.crystallization.dynamic_operator_registry import DynamicOperatorRegistry
from cnge_engine.crystallization.pareto_compiler import ParetoSymbolicCompiler
from cnge_engine.demo import run_demo
from cnge_engine.math_core.symplectic import SymplecticManifold
from cnge_engine.noether.conservation_verifier import ConservationVerifier
from cnge_engine.noether.symmetry_miner import LieSymmetryMiner


def test_reference_crystallization_end_to_end():
    report = run_demo(1000)
    assert report["conserved"]
    assert report["heldout_r2"] >= 0.9999
    assert report["operator_count"] == 1
    assert "z_{0}" in report["latex"]


def test_compiler_angular_momentum_structure_and_registry():
    z = np.random.default_rng(1).normal(size=(200, 6))
    grammar = ASTGrammar(6)
    compiler = ParetoSymbolicCompiler(grammar)
    target = lambda x: x[:, 0] * x[:, 4] - x[:, 1] * x[:, 3]
    result = compiler.compile(target, z)
    expected = ASTNode(
        "-",
        (
            ASTNode("*", (ASTNode("var", value=0), ASTNode("var", value=4))),
            ASTNode("*", (ASTNode("var", value=1), ASTNode("var", value=3))),
        ),
    )
    assert result == expected
    registry = DynamicOperatorRegistry(grammar)
    registry.register_crystallized_invariant(result, "L_z")
    assert registry.get_active_grammar().registered["L_z"].complexity() == 7
    assert all(
        not (a.node.complexity() <= b.node.complexity() and a.mse <= b.mse)
        for a, b in zip(compiler.frontier, compiler.frontier[1:])
    )
    with pytest.raises(ValueError):
        registry.register_crystallized_invariant(ASTNode("const", value=1.0), "L_z")


@pytest.mark.parametrize("operation", ["sin", "cos", "exp", "ln", "square", "sqrt"])
def test_grammar_unary(operation):
    values = np.array([[0.2], [0.5]])
    node = ASTNode(operation, (ASTNode("var", value=0),))
    assert np.isfinite(node.evaluate(values)).all()
    assert node.to_latex()
    assert node.complexity() == 2


def test_miner_gradients_and_loss_reduction():
    m = SymplecticManifold(2, dtype=torch.float64)
    miner = LieSymmetryMiner(m, (16, 16))
    H = lambda z: z.square().sum(-1, keepdim=True) / 2
    z = torch.randn(64, 4, dtype=torch.float64)
    optimizer = torch.optim.Adam(miner.parameters(), lr=0.003)
    initial = float(miner.compute_loss(z, H)[0].detach())
    for _ in range(60):
        optimizer.zero_grad()
        loss, metrics = miner.compute_loss(z, H)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in miner.parameters())
        optimizer.step()
    assert float(miner.compute_loss(z, H)[0].detach()) < initial
    assert metrics["commutator"] >= 0


def test_verifier_rejects_drift_and_nonfinite():
    m = SymplecticManifold(1)
    verifier = ConservationVerifier(m)
    trajectory = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    assert not verifier.verify_trajectory_invariance(lambda z: z[:, :1], trajectory)[0]
    assert not verifier.verify_trajectory_invariance(lambda z: z[:, :1] * float("nan"), trajectory)[
        0
    ]
    assert verifier.bounded_failure_upper_bound(0, 1000) < 0.1


def test_symbolic_nonlinear_search_and_invalid_domains():
    domain = np.linspace(0.1, 2.0, 128)[:, None]
    compiler = ParetoSymbolicCompiler(ASTGrammar(1), max_depth=3, beam_width=16)
    result = compiler.compile(lambda x: np.sin(x[:, 0]), domain)
    np.testing.assert_allclose(result.evaluate(domain), np.sin(domain[:, 0]), atol=1e-10)
    invalid = ASTNode("ln", (ASTNode("const", value=-1.0),))
    assert not np.isfinite(invalid.evaluate(domain)).any()
    with pytest.raises(ValueError, match="target"):
        compiler.compile(lambda x: np.full(len(x), np.nan), domain)
    with pytest.raises(ValueError, match="arity"):
        ASTNode("exec", ())
