# CNGE-v2: implemented contracts and mathematical boundaries

The v2 package replaces finite-difference momentum reconstruction, AST enumeration, soft degeneracy penalties and dense parameter-gradient protection in the recommended workflow. Legacy modules remain import-compatible and explicitly opt-in. V2 does not call the legacy AST compiler or parameter protector. It reuses the validated canonical autodiff primitives, smooth activations and implicit midpoint algorithm.

## 1. Differential coordinate learning and noise filtering

`SymplecticAutoencoder` learns an encoder/decoder with reconstruction, decoder-JVP tangent consistency and an integrated prediction residual. All temporal predictions use implicit midpoint. Solver convergence has a machine-precision floor (especially in float32) and fails explicitly when the requested iteration budget is insufficient; no infinite-horizon structural guarantee is inferred. The tangent term remains available as specified, but its contribution is normalized by `1 + 2 R/dt²` using the configured observation-noise variance. This prevents its expected white-noise contribution from diverging as the observation interval shrinks. The integrated observation loss avoids dividing noisy data by dt. Momentum is an encoder output, never a finite difference of a configuration component.

This is a differential autoencoder, not a probabilistic Gaussian VAE: the requested contract specifies no posterior, prior or KL/ELBO. It imposes a Hamiltonian **latent vector field**. It does not guarantee that an arbitrary observation map is canonical or that noisy observations uniquely identify physical q and p. Such identification requires observability, sufficient data, an appropriate model and calibration of gauge/scale.

A position-only image is identical for states `(q,+p)` and `(q,-p)`; a deterministic single-frame encoder cannot distinguish them. `ObservationHistory` provides bounded delay coordinates for a sensor needing temporal context. Configure `obs_dim = frame_dim × history_length` and calibrate on pairs of complete history windows. History improves observability but is not a universal proof of it.

For an encoder from 128 observation dimensions to 2d latent dimensions, its Jacobian is rectangular. `Dψᵀ J Dψ` is a 128×128 pullback, not the 2d×2d canonical form in the acceptance formula. No canonical observation two-form was supplied. The meaningful structural test implemented here checks the **square latent flow Jacobian**. It does not mislabel that test as proof of canonical encoder recovery.

The EKF propagates covariance with the actual midpoint Jacobian, solves the innovation system by Cholesky and uses Joseph-form covariance updates. Covariances are symmetrized and input PSD is checked. Measurement updates are dissipative statistical estimation, not symplectomorphisms; only the deterministic Hamiltonian prediction has the symplectic contract. Q and R are per-tick isotropic covariances and must be calibrated to the sensor and timestep.

## 2. A well-defined scalar gauge

For a real skew matrix J, `uᵀJu = 0` for every u. The proposed symplectic "norm" has a zero denominator, so it cannot define Gram–Schmidt. Also, the proposed symplectic correlation penalty is a normalized mean Poisson residual: it is zero for `Q=H` and does not exclude that candidate. A finite norm penalty leaves the constant candidate a stationary point even when its loss is nonzero.

V2 instead fixes a scalar-function gauge on a persistent reference set using the positive Sobolev pairing:

```
a = mean(∇f · ∇H) / mean(||∇H||²)
g = f − mean(f) − a (H − mean(H))
Q = g / sqrt(mean(||∇g||²))
```

The coefficients are constant with respect to the **query phase point**, so Q remains an actual scalar and its gradient is integrable. On the reference set this gives unit RMS gradient and zero mean Euclidean gradient correlation with H. Constants, energy aliases and candidates locally functionally dependent on energy are hard-rejected with `DegenerateGaugeError`. No epsilon division fabricates a generator from an exactly degenerate candidate. The additive network bias is removed because the gauge makes it unidentifiable.

These conditions are finite-reference conditions. They do not guarantee independent conserved quantities exist, optimizer convergence, or nondegeneracy everywhere. A generic one-degree-of-freedom autonomous system has no additional functionally independent commuting invariant; the runtime may correctly report no discovery. Training retains higher-order autodiff graphs only for a step; telemetry and buffers are detached, gradients are consumed, and temporary graph references are released.

## 3. Explicit functional dictionaries and constrained STLSQ

The dictionary contains degree-bounded monomials and optional trig functions, dynamic coordinate names, pre-regression scaling/units filtering and an enforced feature budget. For a fixed degree k, construction is polynomial in coordinate count. Allowing arbitrary k still gives combinatorial growth `M = binomial(n+k,k)`; the budget is checked **before** enumeration. There is no expression-tree search.

Runge–Lenz contains `q_i/r`, which is absent from a cubic/trig library. `include_kepler=True` adds those radial functions explicitly and rejects collisions at r=0. Without them, exact Kepler recovery would be an invalid claim. Units filtering assumes dimensionless coefficients; dimensional constants such as μ require explicit dimensional metadata/custom terms. Automatic radial scaling/units filtering is rejected rather than silently dropping required physical terms.

STLSQ normalizes columns, uses stable least squares (no explicitly inverted normal equations), monotonically prunes support, refits active terms and debiases the selected support. Optional homogeneous constraints `C ξ=0` are imposed through a numerical nullspace basis. `conservation_constraints` constructs C from library Lie derivatives along the current vector field, allowing terms to cancel collectively. Pruning individual terms just because they fail to commute would incorrectly eliminate combinations such as angular momentum. `fit_gradient` fits gradients of scalar library functions, retaining integrability and leaving the additive constant undetermined.

A conservative dense cost is `O(K (N M² + M³))`, plus constraint-SVD work, not O(M³) independent of sample count. STLSQ is not a global L0/Pareto optimizer and exact sparse recovery needs identifiability, suitable terms and adequate conditioning. Runtime/accuracy bounds are measured on a particular CPU and dataset, not guaranteed on every machine.

Distillation uses separate reference subsets for fit and validation, a variance gate, neural Poisson residual, held-out scalar R² and the compiled expression's own Poisson residual. Only then is an adapter consolidated. These are empirical checks on the learned latent model; they are not proofs about unknown physical laws. Calibration references should span multiple trajectories/energies; a single orbit alone is insufficient to establish independent invariants.

## 4. Low-rank adapters, exact snapshots and finite capacity

Adapters modify the scalar Hamiltonian, so its vector field remains Hamiltonian. Ordinary rectangular weight matrices are not symplectic matrices; the name does not assert otherwise. Base weights and inference statistics are frozen; active A/B factors are trainable. Each consolidated adapter stores its effective low-rank factors. Named task inference replays the frozen prefix of adapters in **every** layer.

Two explicit policies are available:

- **`activation` (default):** new adapter inputs are projected through the complement of an SVD-orthonormalized union of protected activation spans. Overlapping bases are merged; the incorrect formula `I − Σ U_k U_kᵀ` is never used for overlapping subspaces. Captured spans are finite-sample constraints. When a layer is saturated, `ActivationCapacityError` stops learning instead of silently returning a zero plastic gradient. Effects outside the sampled span and nonlinear network interactions still require task validation.
- **`task`:** new adapters have their full input capacity; named frozen task snapshots give deterministic retention when the correct task is routed. This is the policy used for sequential oscillator → quartic → Hénon–Heiles learning. It relies on an explicit task context and a fixed encoder/decoder. It is not context-free preservation by activation orthogonality.

An n-dimensional activation space cannot contain arbitrarily many independent protected directions while always retaining a nonempty complement. Distinct vector fields assigning different derivatives to the same state also cannot be represented by one unconditional autonomous Hamiltonian. Thus the requested unlimited plasticity and zero-degradation theorem does not follow from LoRA. V2 exposes the policy choice, hard capacity limits (including a 128-task snapshot budget), context and failure explicitly.

The coordinate chart is frozen after calibration; recalibration after consolidating tasks is rejected. Checkpoints save the chart, base/adapters, reference set, miner and optimizer states. On restore, transient EKF and sensor state are reset; runtime tasks/connections are not serialized. State restoration uses tensor-only safe loading (`weights_only=True`), and files are replaced atomically. Only use checkpoints from a source whose tensor sizes/resource requirements you trust.

## Acceptance criteria: exact scope

| Requested criterion | Evidence / boundary |
|---|---|
| Noisy 128D pixel pendulum uniquely recovers physical canonical state | Impossible from an ambiguous single image without more information; a counterexample is tested. A separate **identifiable full-state 128D telemetry** experiment with 15% noise exceeds R² .98 after training; an affine-aligned latent score is diagnostic, not a symplectic gauge proof. |
| Latent symplectic defect <1e−5 | The actual latent midpoint flow has defect below 1e−10 in the test. No rectangular encoder is mislabeled a symplectomorphism. |
| Runge–Lenz <200 ms, R² >.999, <6 terms | Tested on CPU with one BLAS thread, using cubic/trig **plus required radial terms**; five-term recovery is checked on separate points. The input is analytical Runge–Lenz evaluations, not an autonomously discovered neural candidate. |
| No Lz forgetting after learning quartic and Hénon–Heiles | New regimes measurably learn; old named oscillator predictions remain identical; 500-step routed oscillator rollouts keep Lz drift below 1e−4. This validates task-routed snapshots, not the claimed unrestricted activation-space theorem. |

The code is a tested research/development engine with explicit contracts. GPU memory stability over an unbounded stream, a real camera/laboratory deployment, arbitrary semantic control, and production availability/SLA remain unverified. It does not assert production certification from unit tests.
