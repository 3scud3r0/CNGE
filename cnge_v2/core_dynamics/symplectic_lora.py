import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class ActivationCapacityError(RuntimeError):
    """The protected activation span exhausts a layer's finite input dimension."""


def orthonormal_union(existing, incoming):
    if (
        incoming.ndim != 2
        or incoming.shape[0] != existing.shape[0]
        or not torch.isfinite(incoming).all()
    ):
        raise ValueError("activation bases must be finite [input_dim, rank] matrices")
    joined = torch.cat((existing, incoming.to(existing)), 1)
    if joined.shape[1] == 0:
        return joined
    U, singular, _ = torch.linalg.svd(joined, full_matrices=False)
    tol = max(joined.shape) * torch.finfo(joined.dtype).eps * singular.max()
    return U[:, singular > tol].detach()


class FrozenAdapter(nn.Module):
    def __init__(self, A, B):
        super().__init__()
        self.register_buffer("A", A.detach().clone())
        self.register_buffer("B", B.detach().clone())

    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B)


class OrthogonalLoRALinear(nn.Module):
    def __init__(self, base, rank, protection_mode):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank = min(rank, base.in_features, base.out_features)
        self.mode = protection_mode
        self.A = nn.Parameter(base.weight.new_empty(self.rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, self.rank))
        self.register_buffer("U", base.weight.new_empty(base.in_features, 0))
        self.frozen = nn.ModuleList()
        self.selected_task = None
        self.reset_active()

    def reset_active(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def effective_A(self):
        if self.mode == "task":
            return self.A
        return self.A - (self.A @ self.U) @ self.U.T

    def forward(self, x):
        output = self.base(x)
        count = len(self.frozen) if self.selected_task is None else self.selected_task + 1
        for adapter in self.frozen[:count]:
            output = output + adapter(x)
        if self.selected_task is None:
            output = output + F.linear(F.linear(x, self.effective_A()), self.B)
        return output


class SymplecticLoRAEngine(nn.Module):
    """Low-rank energy adapters preserve Hamiltonian form through the scalar energy.

    Activation mode constrains new changes on a finite reference span. Task mode
    permits unrestricted new adapters and exact replay of named frozen snapshots.
    Neither mode proves context-free preservation of all physical invariants.
    """

    def __init__(self, base_model, rank=4, protection_mode="activation"):
        super().__init__()
        if rank < 1 or protection_mode not in ("activation", "task"):
            raise ValueError("invalid adapter configuration")
        self.base_model = base_model.requires_grad_(False).eval()
        self.protection_mode = protection_mode
        self.task_ids = []
        self.layers = {}
        for name, module in list(base_model.named_modules()):
            if isinstance(module, nn.Linear):
                parent_name, _, leaf = name.rpartition(".")
                parent = base_model.get_submodule(parent_name) if parent_name else base_model
                replacement = OrthogonalLoRALinear(module, rank, protection_mode)
                setattr(parent, leaf, replacement)
                self.layers[name] = replacement
        if not self.layers:
            raise ValueError("base_model must contain at least one Linear layer")

    def train(self, mode=True):
        super().train(mode)
        # Frozen running statistics and disabled dropout are part of snapshot reproducibility.
        self.base_model.eval()
        return self

    def forward(self, *args, **kwargs):
        self.base_model.eval()
        return self.base_model(*args, **kwargs)

    def active_parameters(self):
        return [p for layer in self.layers.values() for p in (layer.A, layer.B)]

    def assert_plastic_capacity(self):
        if self.protection_mode == "activation":
            saturated = [
                name
                for name, layer in self.layers.items()
                if layer.U.shape[1] == layer.base.in_features
            ]
            if saturated:
                raise ActivationCapacityError(
                    f"activation capacity exhausted in {saturated}; explicit task routing or feature expansion is required"
                )

    def active_adapter_state(self):
        return {
            f"{name}.{suffix}": getattr(layer, suffix).detach().clone()
            for name, layer in self.layers.items()
            for suffix in ("A", "B")
        }

    def capture_activation_subspaces(self, samples):
        activations = {name: [] for name in self.layers}
        handles = []
        for name, layer in self.layers.items():

            def hook(module, args, key=name):
                activations[key].append(args[0].detach().reshape(-1, module.base.in_features))

            handles.append(layer.register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                self(samples)
        finally:
            for handle in handles:
                handle.remove()
        result = {}
        for name, entries in activations.items():
            if not entries:
                raise ValueError(f"layer {name} was not evaluated")
            values = torch.cat(entries).T
            empty = values.new_empty(values.shape[0], 0)
            result[name] = orthonormal_union(empty, values)
        return result

    def register_frozen_invariant_adapter(self, task_id, adapter_weights, activation_subspace):
        if not isinstance(task_id, str) or not task_id or task_id in self.task_ids:
            raise ValueError("task_id must be unique and nonempty")
        if len(self.task_ids) >= 128:
            raise ActivationCapacityError("snapshot budget of 128 tasks exhausted")
        # Validate every layer before changing any model state.
        prepared = []
        for name, layer in self.layers.items():
            A = adapter_weights[f"{name}.A"].to(layer.A)
            B = adapter_weights[f"{name}.B"].to(layer.B)
            if (
                A.shape != layer.A.shape
                or B.shape != layer.B.shape
                or not torch.isfinite(A).all()
                or not torch.isfinite(B).all()
            ):
                raise ValueError("invalid adapter weights")
            incoming = (
                activation_subspace[name]
                if isinstance(activation_subspace, dict)
                else activation_subspace
            )
            union = orthonormal_union(layer.U, incoming)
            effective = A if layer.mode == "task" else A - (A @ layer.U) @ layer.U.T
            prepared.append((layer, effective, B, union))
        with torch.no_grad():
            for layer, A, B, union in prepared:
                layer.frozen.append(FrozenAdapter(A, B))
                layer.U = union
                layer.reset_active()
        self.task_ids.append(task_id)

    def get_orthogonalized_gradient(self, layer_name, raw_gradient):
        layer = self.layers[layer_name]
        if raw_gradient.shape != layer.base.weight.shape:
            raise ValueError("expected a dense [output_dim,input_dim] layer gradient")
        U = layer.U.to(raw_gradient)
        return raw_gradient - (raw_gradient @ U) @ U.T

    @contextmanager
    def task(self, task_id):
        if task_id is not None and task_id not in self.task_ids:
            raise KeyError(f"unknown task: {task_id}")
        previous = [layer.selected_task for layer in self.layers.values()]
        index = None if task_id is None else self.task_ids.index(task_id)
        try:
            for layer in self.layers.values():
                layer.selected_task = index
            yield self
        finally:
            for layer, value in zip(self.layers.values(), previous):
                layer.selected_task = value

    def get_extra_state(self):
        return {"task_ids": self.task_ids.copy(), "protection_mode": self.protection_mode}

    def set_extra_state(self, state):
        if state["protection_mode"] != self.protection_mode:
            raise ValueError("checkpoint protection mode differs")
        self.task_ids = list(state["task_ids"])

    def load_state_dict(self, state_dict, strict=True, assign=False):
        metadata = state_dict.get("_extra_state", {})
        tasks = metadata.get("task_ids", [])
        for name, layer in self.layers.items():
            prefix = f"base_model.{name}."
            layer.U = torch.empty_like(state_dict[prefix + "U"], device=layer.A.device)
            layer.frozen = nn.ModuleList(
                [
                    FrozenAdapter(
                        state_dict[prefix + f"frozen.{i}.A"].to(layer.A),
                        state_dict[prefix + f"frozen.{i}.B"].to(layer.B),
                    )
                    for i in range(len(tasks))
                ]
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
