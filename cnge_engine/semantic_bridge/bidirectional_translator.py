from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GeometricConstraint:
    name: str
    generator: torch.Tensor
    description: str


class BidirectionalTranslator:
    """Explicit supported constraint vocabulary; no fabricated scientific citations."""

    def __init__(self, embedder, transport, registry):
        self.embedder, self.transport, self.registry = embedder, transport, registry

    def prompt_to_geometric_constraint(self, user_prompt):
        prompt = user_prompt.casefold()
        if (
            not any(word in prompt for word in ("rotational", "rotation", "rotação", "rotacional"))
            or "xy" not in prompt
        ):
            raise ValueError("supported constraint: rotational symmetry in the xy-plane")
        n = self.registry.grammar.variable_count
        if n < 4 or n % 2:
            raise ValueError("xy rotation requires at least two canonical coordinates")
        generator = torch.zeros(n, n)
        d = n // 2
        generator[0, 1], generator[1, 0] = -1, 1
        generator[d, d + 1], generator[d + 1, d] = -1, 1
        return GeometricConstraint(
            "rotation_xy",
            generator,
            "Simultaneous rotation of q and p in the xy-plane; enforce via the Lz Poisson residual.",
        )

    def discovery_to_natural_explanation(self, symbolic_node):
        names = [
            name for name, node in self.registry.grammar.registered.items() if node == symbolic_node
        ]
        return {
            "latex": symbolic_node.to_latex(),
            "registered_names": names,
            "interpretation": "Candidate scalar invariant; interpretation requires independent physical validation.",
            "literature_matches": [],
            "proof": None,
        }
