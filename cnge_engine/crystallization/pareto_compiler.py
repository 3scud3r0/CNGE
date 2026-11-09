import itertools
from dataclasses import dataclass

import numpy as np

from .ast_grammar import ASTNode


@dataclass(frozen=True)
class Candidate:
    node: ASTNode
    mse: float
    r2: float


class ParetoSymbolicCompiler:
    """Bounded search plus quadratic least squares. No global-optimum guarantee."""

    def __init__(self, grammar, max_depth=6, beam_width=64, mdl_weight=0.05):
        if max_depth < 1 or beam_width < 1 or mdl_weight < 0:
            raise ValueError("invalid search budget")
        self.grammar, self.max_depth, self.beam_width = grammar, max_depth, beam_width
        self.mdl_weight = mdl_weight
        self.frontier = []

    def compile(self, target_eval_fn, z_domain, r2_target=0.9999):
        z = np.asarray(z_domain, dtype=float)
        if (
            z.ndim != 2
            or z.shape[1] != self.grammar.variable_count
            or len(z) < 2
            or not np.isfinite(z).all()
        ):
            raise ValueError("invalid symbolic domain")
        y = np.asarray(target_eval_fn(z), dtype=float).reshape(-1)
        if len(y) != len(z) or not np.isfinite(y).all() or not 0 <= r2_target <= 1:
            raise ValueError("invalid target evaluations or R2 target")
        variance = np.mean((y - y.mean()) ** 2)
        seen, candidates = set(), []

        def add(node):
            if node.depth() > self.max_depth or node in seen:
                return
            seen.add(node)
            values = node.evaluate(z)
            if not np.isfinite(values).all():
                return
            with np.errstate(over="ignore"):
                mse = float(np.mean((values - y) ** 2))
            if not np.isfinite(mse):
                return
            r2 = 1 - mse / variance if variance > 1e-24 else float(mse < 1e-24)
            candidates.append(Candidate(node, mse, r2))

        terminals = self.grammar.terminals()
        for node in terminals:
            add(node)
        # Fit sparse quadratic signatures; all retained coefficients are explicit AST constants.
        variables = [ASTNode("var", value=i) for i in range(z.shape[1])]
        features = [ASTNode("const", value=1.0), *variables]
        features += [
            ASTNode("*", (a, b)) for a, b in itertools.combinations_with_replacement(variables, 2)
        ]
        coefficients = np.linalg.lstsq(
            np.stack([f.evaluate(z) for f in features], 1), y, rcond=None
        )[0]
        terms = []
        for coefficient, feature in zip(coefficients, features):
            if abs(coefficient) < 1e-9:
                continue
            rounded = round(float(coefficient), 10)
            term = (
                feature
                if abs(rounded - 1) < 1e-9
                else ASTNode("*", (ASTNode("const", value=rounded), feature))
            )
            terms.append(term)
        if terms:
            node = terms[0]
            for term in terms[1:]:
                # Emit subtraction for negative unit factors, preserving angular-momentum structure.
                if (
                    term.op == "*"
                    and term.children[0].op == "const"
                    and term.children[0].value == -1
                ):
                    node = ASTNode("-", (node, term.children[1]))
                else:
                    node = ASTNode("+", (node, term))
            add(node)
        for _ in range(self.max_depth - 1):
            accurate = [c for c in candidates if c.r2 >= r2_target]
            if accurate:
                break
            beam = sorted(candidates, key=lambda c: c.mse + self.mdl_weight * c.node.complexity())[
                : self.beam_width
            ]
            for c in beam:
                for op in self.grammar.unary_ops:
                    add(ASTNode(op, (c.node,)))
                for other in terminals:
                    for op in self.grammar.binary_ops:
                        add(ASTNode(op, (c.node, other)))
                        if op in ("-", "/"):
                            add(ASTNode(op, (other, c.node)))
        ordered = sorted(candidates, key=lambda c: (c.node.complexity(), c.mse))
        self.frontier = []
        best = float("inf")
        for c in ordered:
            if c.mse < best:
                self.frontier.append(c)
                best = c.mse
        accurate = [c for c in self.frontier if c.r2 >= r2_target]
        if accurate:
            return min(accurate, key=lambda c: (c.node.complexity(), c.mse)).node
        return min(self.frontier, key=lambda c: c.mse + self.mdl_weight * c.node.complexity()).node
