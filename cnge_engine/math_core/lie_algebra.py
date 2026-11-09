import torch


class LieAlgebraEngine:
    @staticmethod
    def lie_bracket_fields(X_fn, Y_fn, z):
        _, dy_x = torch.autograd.functional.jvp(Y_fn, z, X_fn(z), create_graph=True)
        _, dx_y = torch.autograd.functional.jvp(X_fn, z, Y_fn(z), create_graph=True)
        return dy_x - dx_y

    def structure_constants(self, basis_fields, z_samples):
        if not basis_fields:
            raise ValueError("empty basis")
        design = torch.stack([fn(z_samples).reshape(-1) for fn in basis_fields], -1)
        rows = []
        for x in basis_fields:
            rows.append(
                torch.stack(
                    [
                        torch.linalg.lstsq(
                            design, self.lie_bracket_fields(x, y, z_samples).reshape(-1, 1)
                        ).solution[:, 0]
                        for y in basis_fields
                    ]
                )
            )
        return torch.stack(rows)

    @staticmethod
    def killing_residual(X_fn, metric_fn, z):
        """Pointwise Lie derivative of a metric, shape [B,n,n]."""

        def point(x):
            X = X_fn(x[None])[0]
            g = metric_fn(x[None])[0]
            jac = torch.func.jacrev(lambda t: X_fn(t[None])[0])(x)
            dg = torch.func.jacrev(lambda t: metric_fn(t[None])[0])(x)
            return torch.einsum("ijk,k->ij", dg, X) + jac.T @ g + g @ jac

        return torch.stack([point(x) for x in z])
