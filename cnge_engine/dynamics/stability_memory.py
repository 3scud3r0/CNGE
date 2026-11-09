import torch


class SymplecticSubspaceProtector:
    """Low-rank Euclidean parameter projection; preservation is local, not a theorem of retention."""

    def __init__(self, parameter_dim, damping=1e-4):
        if parameter_dim < 1 or damping < 0:
            raise ValueError("invalid projection dimensions or damping")
        self.parameter_dim, self.damping = parameter_dim, damping
        self.rows = []

    def register_invariant(self, Q_invariant_fn, H_model, validation_samples):
        z = validation_samples.detach().requires_grad_(True)
        residual = H_model.manifold.poisson_bracket(Q_invariant_fn(z), H_model(z), z).mean()
        params = [p for p in H_model.parameters() if p.requires_grad]
        if sum(p.numel() for p in params) != self.parameter_dim:
            raise ValueError("parameter dimension mismatch")
        grads = torch.autograd.grad(residual, params, allow_unused=True)
        self.rows.append(
            torch.cat(
                [
                    (torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip(params, grads)
                ]
            ).detach()
        )

    def project_gradient(self, raw_gradient):
        if raw_gradient.shape != (self.parameter_dim,):
            raise ValueError("expected a flat parameter gradient")
        if not self.rows:
            return raw_gradient.clone()
        A = torch.stack(self.rows).to(raw_gradient)
        if self.damping == 0:
            return raw_gradient - torch.linalg.pinv(A) @ (A @ raw_gradient)
        gram = A @ A.T + self.damping * torch.eye(len(A), device=A.device, dtype=A.dtype)
        return raw_gradient - A.T @ torch.linalg.solve(gram, A @ raw_gradient)
