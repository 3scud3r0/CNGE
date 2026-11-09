from .ast_grammar import ASTNode


class DynamicOperatorRegistry:
    def __init__(self, base_grammar):
        self.grammar = base_grammar

    def register_crystallized_invariant(self, symbolic_node: ASTNode, latex_name: str):
        if not latex_name.strip():
            raise ValueError("operator name must be nonempty")

        def validate(node):
            if node.op == "var" and node.value >= self.grammar.variable_count:
                raise ValueError("operator references an out-of-domain variable")
            for child in node.children:
                validate(child)

        validate(symbolic_node)
        if (
            latex_name in self.grammar.registered
            and self.grammar.registered[latex_name] != symbolic_node
        ):
            raise ValueError("operator name already registered")
        # Preserve the expanded node's complexity; no cost-free macro compression.
        self.grammar.registered[latex_name] = symbolic_node

    def get_active_grammar(self):
        return self.grammar
