from itertools import combinations_with_replacement
from math import comb

import torch


class FunctionalDictionary:
    """Fixed-degree explicit dictionary with a hard feature budget, units and scaling filters."""

    def __init__(
        self,
        variable_names,
        max_poly_order=3,
        include_trig=True,
        *,
        include_kepler=False,
        max_features=2048,
        scaling_weights=None,
        target_scaling=None,
        variable_units=None,
        target_units=None,
    ):
        n = len(variable_names)
        if n < 1 or len(set(variable_names)) != n or max_poly_order < 1:
            raise ValueError("invalid dictionary variables or degree")
        count = (
            comb(n + max_poly_order, max_poly_order)
            + (2 * n if include_trig else 0)
            + (n // 2 if include_kepler else 0)
        )
        if count > max_features:
            raise ValueError(f"dictionary would contain {count} features, budget is {max_features}")
        if include_kepler and n % 2:
            raise ValueError("Kepler terms require canonical q,p coordinates")
        if scaling_weights is not None and len(scaling_weights) != n:
            raise ValueError("one scaling weight per variable is required")
        if variable_units is not None and (
            len(variable_units) != n or len({len(u) for u in variable_units}) != 1
        ):
            raise ValueError("invalid variable units")
        if include_kepler and (scaling_weights is not None or variable_units is not None):
            raise ValueError(
                "supply radial units/scaling explicitly in a custom library; automatic Kepler filtering is unsupported"
            )
        if (
            variable_units is not None
            and target_units is not None
            and len(variable_units[0]) != len(target_units)
        ):
            raise ValueError("target and variable unit dimensions differ")
        self.variable_names = list(variable_names)
        self.features = []
        for degree in range(max_poly_order + 1):
            for indices in combinations_with_replacement(range(n), degree):
                if scaling_weights is not None and target_scaling is not None:
                    weight = sum(scaling_weights[i] for i in indices)
                    if abs(weight - target_scaling) > 1e-12:
                        continue
                if variable_units is not None and target_units is not None:
                    units = tuple(
                        sum(variable_units[i][k] for i in indices) for k in range(len(target_units))
                    )
                    if units != tuple(target_units):
                        continue
                name = " ".join(variable_names[i] for i in indices) or "1"
                self.features.append(("poly", indices, name))
        if include_trig:
            for i, name in enumerate(variable_names):
                dimensionless = variable_units is None or not any(variable_units[i])
                target_dimensionless = target_units is None or not any(target_units)
                scale_ok = scaling_weights is None or (
                    scaling_weights[i] == 0 and target_scaling in (None, 0)
                )
                if dimensionless and target_dimensionless and scale_ok:
                    for op in ("sin", "cos"):
                        self.features.append((op, (i,), f"\\{op}({name})"))
        if include_kepler:
            for i in range(n // 2):
                self.features.append(("radial", (i,), f"{variable_names[i]}/r"))
        if not self.features:
            raise ValueError("dimensional/scaling filters eliminated all features")

    def build_library(self, z):
        if z.ndim != 2 or z.shape[1] != len(self.variable_names) or not torch.isfinite(z).all():
            raise ValueError("expected finite [B,variable_count]")
        values = []
        for op, indices, _ in self.features:
            if op == "poly":
                value = z[:, list(indices)].prod(-1) if indices else torch.ones_like(z[:, 0])
            elif op == "sin":
                value = z[:, indices[0]].sin()
            elif op == "cos":
                value = z[:, indices[0]].cos()
            else:
                radius = torch.linalg.vector_norm(z[:, : z.shape[1] // 2], dim=-1)
                if (radius <= torch.finfo(z.dtype).eps).any():
                    raise ValueError("Kepler dictionary undefined at collision r=0")
                value = z[:, indices[0]] / radius
            values.append(value)
        return torch.stack(values, -1), [f[2] for f in self.features]

    def gradient_library(self, z):
        point = z.detach().requires_grad_(True)
        values, names = self.build_library(point)
        gradients = []
        for column in values.unbind(-1):
            gradient = (
                torch.autograd.grad(column.sum(), point, retain_graph=True)[0]
                if column.requires_grad
                else torch.zeros_like(point)
            )
            gradients.append(gradient)
        return torch.stack(gradients, 1), names

    def conservation_constraints(self, z, vector_field_fn):
        gradients, _ = self.gradient_library(z)
        field = vector_field_fn(z).detach()
        return torch.einsum("bmn,bn->bm", gradients, field)
