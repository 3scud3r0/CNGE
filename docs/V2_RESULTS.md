# CNGE-v2: local validation results

Local CPU validation, Python 3.12, PyTorch 2.15.0+cpu, NumPy 2.5.4. CPU numerical tests and benchmarks use one PyTorch/BLAS thread. No GPU was available. These are observed results, not universal hardware guarantees.

## Executed current checks

- **58 tests passed**, zero failures, in approximately **5 seconds**: 36 legacy fast tests and 22 v2 tests. Three legacy long tests were explicitly skipped in this invocation.
- `ruff check` and `ruff format --check`: passed for both packages, tests and tools.
- Bytecode compilation of v2 and `pip check`: passed.
- Wheel and source distribution builds passed. A wheel installed outside the checkout executed the v2 smoke demo successfully; the source distribution was checked for v2 code, verification tests, tools, interface and the complete TXT.
- Full v2 demo: 40 calibration steps and 20 online observation ticks completed. Calibration objective decreased from **0.727947** to **0.417233**. Last posterior covariance trace: **0.00029819**. This smoke run does not establish accurate physical coordinate identification or successful autonomous invariant discovery; its Noether residual did not satisfy the discovery gate, so no discovery is asserted.
- Calibration, real adapter parameter updates, validated symbolic consolidation, rejection without consolidation, frozen coordinate-chart enforcement, atomic checkpoint save/load, optimizer state restoration and named task prediction equivalence are exercised.
- Bounded asynchronous producer/worker ordering and error cancellation are exercised. Camera capture and a downloaded embedding model are not prerequisites of v2 and were not validated.

## Quantitative experiments

| Experiment | Measured result | Scope |
|---|---|---|
| 128D full-state telemetry with additive Gaussian noise at 15% of signal standard deviation | Denoised held-out observation **R² = 0.99826994**; affine-aligned latent **R² = 0.99821123** | Learned bottleneck autoencoder; no physical state targets in reconstruction training. The final affine alignment uses labelled training states only as a validation diagnostic. This is **not** position-only pixel pendulum recovery or a canonical encoder proof. |
| 3D Kepler Runge–Lenz component | **5 active terms**, fit and held-out **R² = 1.0** | Cubic/trig library plus required q_i/r terms; analytical target evaluations, not neural invariant discovery. |
| Kepler STLSQ fit | **4.96 ms** in the test; **6.85 ms** in the demo | Warm local CPU, one BLAS thread; library construction excluded from the fit timer. Both observations satisfy the 200 ms target for this dataset. |
| Oscillator → quartic → Hénon–Heiles adapter learning | Each new energy-fit error decreases by >100×; old named oscillator predictions are exactly unchanged | Task-routed low-rank energy adapters. This test does not claim a shared unconditioned vector field preserves all regimes. |
| Old oscillator rollout after those updates | Maximum relative Lz drift **8.88×10⁻¹⁶** over 500 midpoint steps | Explicitly selecting the saved oscillator task; initial Lz=1, away from a singular relative normalization. |
| EKF denoising | Average filtered state MSE below 20% of raw noisy MSE | Controlled oscillator with known dynamics and identity observation; every posterior covariance is symmetric PSD. |
| Gauge and integration | Constants, H and H² rejected; unit RMS reference gradient, zero reference Sobolev correlation, symmetric scalar Hessian; latent flow defect <10⁻¹⁰ | Reference-set checks in float64, not a global proof for an unknown physical observation map. |

Additional negative tests cover ambiguous instantaneous images, activation-capacity exhaustion, overlapping subspace bases, invalid/rank-deficient regression data, undefined Kepler collision terms, dictionary-budget rejection, missing calibration, changing a frozen chart, malformed calibration inputs, and sensor failure propagation.

## Reproducibility

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q -o junit_family=legacy --junitxml=artifacts/v2-validation.xml
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m cnge_v2.demo --output artifacts/v2-demo.json
```

The JUnit file records the numerical R², fit timing and routed Lz metrics as test properties. Times may vary. CI is configured for Python 3.11 and 3.12 but a remote CI result is not claimed until GitHub runs it.

## Historical long validation

The unchanged shared midpoint implementation and legacy Yoshida integrator previously completed three 50,000-step checks (150,000 steps total) in 842.43 seconds. Those results are recorded in `RESULTS.md`. The v2 change adds a direct latent midpoint Jacobian check and EKF covariance tests. The 14-minute historical suite was not rerun because the integrator implementation was unchanged; it is not counted as a fresh v2 run.

## Deployment boundaries

See `V2_VALIDATION.md` for the precise mathematical corrections and acceptance limits. Strict canonical recovery from ambiguous single frames, unconstrained global sparse optimality, unlimited orthogonal plasticity, context-free zero forgetting and production/GPU guarantees are not demonstrated.
