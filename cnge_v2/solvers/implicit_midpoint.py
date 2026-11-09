from cnge_engine.math_core.integrators import ImplicitMidpointIntegrator


class ImplicitMidpointSolver(ImplicitMidpointIntegrator):
    """Shared validated midpoint implementation, including explicit nonconvergence errors."""

    def __init__(self, tol=1e-10, max_iter=15):
        super().__init__(max_iters=max_iter, tol=tol)

    def integrate_step(self, z_t, vector_field_fn, dt):
        return self.step(z_t, vector_field_fn, dt)
