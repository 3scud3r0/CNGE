import time

import numpy as np
import pytest
import torch

from cnge_v2.symbolic_distiller.constrained_stlsq import ConstrainedSTLSQ
from cnge_v2.symbolic_distiller.functional_dictionary import FunctionalDictionary


def kepler_data(seed, count=512):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(count, 6))
    z[:, :3] += np.array([2.0, 0, 0])
    q, p = z[:, :3], z[:, 3:]
    angular = np.cross(q, p)
    A = np.cross(p, angular) - q / np.linalg.norm(q, axis=1, keepdims=True)
    return z, A[:, 0]


def test_kepler_runge_lenz_five_term_recovery_and_timing(record_property):
    z, target = kepler_data(3)
    library = FunctionalDictionary(
        ["q_0", "q_1", "q_2", "p_0", "p_1", "p_2"], include_trig=True, include_kepler=True
    )
    X, names = library.build_library(torch.from_numpy(z))
    solver = ConstrainedSTLSQ(threshold=0.05)
    start = time.perf_counter()
    formula, coefficients, r2 = solver.fit(X.numpy(), target, names)
    elapsed = time.perf_counter() - start
    record_property("fit_seconds", elapsed)
    record_property("r2", r2)
    assert elapsed < 0.2, f"STLSQ took {elapsed:.3f}s on this CPU"
    assert r2 > 0.999
    assert np.count_nonzero(coefficients) == 5
    other, actual = kepler_data(19)
    values, _ = library.build_library(torch.from_numpy(other))
    predicted = values.numpy() @ coefficients
    assert 1 - np.mean((predicted - actual) ** 2) / np.var(actual) > 0.999
    assert "q_0/r" in formula
    assert solver.n_iter_ <= solver.max_iter


def test_scaling_and_unit_filters_are_applied_before_regression():
    library = FunctionalDictionary(
        ["q_0", "p_0"], include_trig=True, scaling_weights=[1, 1], target_scaling=2
    )
    _, names = library.build_library(torch.randn(10, 2))
    assert set(names) == {"q_0 q_0", "q_0 p_0", "p_0 p_0"}
    units = FunctionalDictionary(
        ["q_0", "p_0"], variable_units=[(1, 0), (0, 1)], target_units=(1, 1)
    )
    _, names = units.build_library(torch.randn(4, 2))
    assert names == ["q_0 p_0"]
    with pytest.raises(ValueError, match="budget"):
        FunctionalDictionary([str(i) for i in range(100)], max_poly_order=5)


def test_homogeneous_symmetry_constraints_and_gradient_distillation():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(128, 4))
    target = X @ np.array([1.0, 1.0, 0.0, 0.0])
    _, coef, r2 = ConstrainedSTLSQ().fit(
        X, target, ["a", "b", "c", "d"], constraints=np.array([[1.0, -1.0, 0.0, 0.0]])
    )
    assert r2 > 0.999999
    assert abs(coef[0] - coef[1]) < 1e-12
    z = torch.randn(128, 4, dtype=torch.float64)
    library = FunctionalDictionary(
        ["q_0", "q_1", "p_0", "p_1"], max_poly_order=2, include_trig=False
    )
    gradient = torch.stack((z[:, 3], -z[:, 2], -z[:, 1], z[:, 0]), -1)
    _, coefficients, r2 = ConstrainedSTLSQ().fit_gradient(library, z, gradient)
    assert r2 > 0.999999
    assert np.count_nonzero(coefficients) == 2


def test_empty_support_rank_deficiency_and_invalid_inputs():
    X = np.ones((32, 3))
    _, coefficients, r2 = ConstrainedSTLSQ(threshold=100).fit(X, np.zeros(32), ["a", "b", "c"])
    assert not np.any(coefficients)
    assert r2 == 1
    with pytest.raises(ValueError):
        ConstrainedSTLSQ().fit(X, np.full(32, np.nan), ["a", "b", "c"])
    with pytest.raises(ValueError, match="collision"):
        FunctionalDictionary(["q_0", "p_0"], include_kepler=True).build_library(torch.zeros(1, 2))
