from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ASTNode:
    op: str
    children: tuple = ()
    value: float | int | None = None

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))
        arity = {
            "const": 0,
            "var": 0,
            "+": 2,
            "-": 2,
            "*": 2,
            "/": 2,
            "sin": 1,
            "cos": 1,
            "exp": 1,
            "ln": 1,
            "square": 1,
            "sqrt": 1,
        }
        if self.op not in arity or len(self.children) != arity[self.op]:
            raise ValueError("unknown operator or invalid arity")
        if self.op == "const" and (self.value is None or not np.isfinite(self.value)):
            raise ValueError("constant must be finite")
        if self.op == "var" and (not isinstance(self.value, int) or self.value < 0):
            raise ValueError("variable requires a nonnegative integer index")

    def evaluate(self, z):
        z = np.asarray(z)
        if self.op == "const":
            return np.full(z.shape[:-1], self.value, dtype=float)
        if self.op == "var":
            return z[..., self.value]
        args = [c.evaluate(z) for c in self.children]
        operations = {
            "+": np.add,
            "-": np.subtract,
            "*": np.multiply,
            "/": np.divide,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "ln": np.log,
            "square": np.square,
            "sqrt": np.sqrt,
        }
        # Invalid domains remain invalid: no protected division/log that changes the equation.
        with np.errstate(all="ignore"):
            return operations[self.op](*args)

    def to_latex(self):
        if self.op == "const":
            return f"{self.value:g}"
        if self.op == "var":
            return f"z_{{{self.value}}}"
        c = [child.to_latex() for child in self.children]
        if self.op in ("+", "-", "*"):
            symbol = r"\cdot" if self.op == "*" else self.op
            return f"({c[0]} {symbol} {c[1]})"
        if self.op == "/":
            return rf"\frac{{{c[0]}}}{{{c[1]}}}"
        if self.op == "square":
            return f"({c[0]})^2"
        if self.op == "sqrt":
            return rf"\sqrt{{{c[0]}}}"
        return rf"\{self.op}({c[0]})"

    def complexity(self):
        return 1 + sum(c.complexity() for c in self.children)

    def depth(self):
        return 1 + max((c.depth() for c in self.children), default=0)


@dataclass
class ASTGrammar:
    variable_count: int
    constants: tuple = (0.0, 1.0, -1.0, 0.5)
    binary_ops: tuple = ("+", "-", "*", "/")
    unary_ops: tuple = ("sin", "cos", "exp", "ln", "square", "sqrt")
    registered: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.variable_count < 1:
            raise ValueError("variable_count must be positive")

    def terminals(self):
        return (
            [ASTNode("var", value=i) for i in range(self.variable_count)]
            + [ASTNode("const", value=c) for c in self.constants]
            + list(self.registered.values())
        )
