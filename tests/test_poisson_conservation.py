import pytest
import torch

from cnge_engine.math_core.lie_algebra import LieAlgebraEngine
from cnge_engine.math_core.symplectic import SymplecticManifold


@pytest.mark.parametrize("d", [1, 2, 3])
def test_canonical_poisson_and_matrix(d):
    m = SymplecticManifold(d, dtype=torch.float64)
    z = torch.randn(12, 2 * d, dtype=torch.float64, requires_grad=True)
    assert torch.equal(m.J.T, -m.J)
    assert torch.equal(m.J @ m.J, -torch.eye(2 * d, dtype=z.dtype))
    for i in range(2 * d):
        for j in range(2 * d):
            actual = m.poisson_bracket(z[:, i : i + 1], z[:, j : j + 1], z)
            torch.testing.assert_close(actual, m.J[i, j].expand_as(actual))
    torch.testing.assert_close(
        m.poisson_bracket(torch.ones(12, 1), z[:, :1], z), torch.zeros(12, 1, dtype=z.dtype)
    )


def test_central_force_angular_momentum():
    m = SymplecticManifold(2, dtype=torch.float64)
    z = torch.randn(128, 4, dtype=torch.float64, requires_grad=True)
    q, p = z.chunk(2, -1)
    H = p.square().sum(-1, keepdim=True) / 2 + q.square().sum(-1, keepdim=True).square()
    L = z[:, :1] * z[:, 3:4] - z[:, 1:2] * z[:, 2:3]
    assert m.poisson_bracket(L, H, z).abs().max() < 1e-12


def test_lie_bracket_structure_and_killing():
    engine = LieAlgebraEngine()
    z = torch.randn(24, 2, dtype=torch.float64)
    translation = lambda x: torch.ones_like(x)
    dilation = lambda x: x
    torch.testing.assert_close(
        engine.lie_bracket_fields(translation, dilation, z), torch.ones_like(z)
    )
    constants = engine.structure_constants([translation, dilation], z)
    assert abs(constants[0, 1, 0] - 1) < 1e-12
    rotation = lambda x: torch.stack((-x[:, 1], x[:, 0]), -1)
    metric = lambda x: torch.eye(2, dtype=x.dtype).expand(len(x), 2, 2)
    assert engine.killing_residual(rotation, metric, z).abs().max() == 0
