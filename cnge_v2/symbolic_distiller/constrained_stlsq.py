import math

import numpy as np


class ConstrainedSTLSQ:
    """Monotone-support STLSQ; polynomial for a fixed explicit library, no L0 optimality claim."""

    def __init__(self, threshold=0.05, alpha=1e-4, max_iter=20):
        if not all(math.isfinite(v) and v >= 0 for v in (threshold, alpha)) or max_iter < 1:
            raise ValueError("invalid STLSQ controls")
        self.threshold, self.alpha, self.max_iter = threshold, alpha, max_iter
        self.n_iter_ = 0
        self.coefficients_ = None

    def _solve(self, X, y, C, alpha):
        if X.shape[1] == 0:
            return np.empty(0)
        basis = np.eye(X.shape[1])
        if C is not None and C.size:
            _, singular, vh = np.linalg.svd(C, full_matrices=C.shape[0] < C.shape[1])
            tol = max(C.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 1)
            rank = int(np.sum(singular > tol))
            basis = vh[rank:].T
            if basis.shape[1] == 0:
                return np.zeros(X.shape[1])
        design = X @ basis
        if alpha:
            design = np.concatenate((design, np.sqrt(alpha) * np.eye(basis.shape[1])))
            y = np.concatenate((y, np.zeros(basis.shape[1])))
        return basis @ np.linalg.lstsq(design, y, rcond=None)[0]

    def fit(self, Theta, Target, feature_names, *, constraints=None):
        X = np.asarray(Theta, dtype=float)
        y = np.asarray(Target, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if (
            X.ndim != 2
            or y.shape != (len(X),)
            or min(X.shape) < 1
            or len(feature_names) != X.shape[1]
            or not np.isfinite(X).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("expected finite Theta[N,M], Target[N,1], and M feature names")
        C = None if constraints is None else np.asarray(constraints, dtype=float)
        if C is not None and (C.ndim != 2 or C.shape[1] != X.shape[1] or not np.isfinite(C).all()):
            raise ValueError("constraints must be a finite matrix C[:,M], enforcing C @ xi=0")
        scales = np.sqrt(np.mean(X * X, axis=0))
        nonzero = scales > np.finfo(float).eps
        scales = np.where(nonzero, scales, 1.0)
        normalized = X / scales
        normalized_C = None if C is None else C / scales
        active = nonzero.copy()
        coefficients = np.zeros(X.shape[1])
        coefficients[active] = self._solve(
            normalized[:, active], y, None if C is None else normalized_C[:, active], self.alpha
        )
        for iteration in range(self.max_iter):
            support = active & (np.abs(coefficients) >= self.threshold)
            self.n_iter_ = iteration + 1
            stable = np.array_equal(active, support)
            active = support
            coefficients[:] = 0
            coefficients[active] = self._solve(
                normalized[:, active],
                y,
                None if C is None else normalized_C[:, active],
                0.0 if stable else self.alpha,
            )
            if stable:
                break
        # Final debiasing refit on the selected support, still respecting homogeneous constraints.
        coefficients[:] = 0
        coefficients[active] = self._solve(
            normalized[:, active], y, None if C is None else normalized_C[:, active], 0.0
        )
        coefficients /= scales
        predictions = X @ coefficients
        variance = np.sum((y - y.mean()) ** 2)
        error = np.sum((y - predictions) ** 2)
        r2 = float(1 - error / variance) if variance > 1e-24 else float(error <= 1e-24)
        terms = [
            f"{coefficient:.10g}\\,({name})"
            for coefficient, name in zip(coefficients, feature_names)
            if coefficient != 0
        ]
        formula = " + ".join(terms).replace("+ -", "- ") or "0"
        self.coefficients_ = coefficients.copy()
        return formula, coefficients.copy(), r2

    def fit_gradient(self, dictionary, z, target_gradients, *, constraints=None):
        gradients, names = dictionary.gradient_library(z)
        target = target_gradients.detach().cpu().numpy()
        if target.shape != tuple(z.shape):
            raise ValueError("target gradients must have shape [N,2d]")
        design = gradients.permute(0, 2, 1).detach().cpu().numpy().reshape(-1, len(names))
        return self.fit(design, target.reshape(-1), names, constraints=constraints)
