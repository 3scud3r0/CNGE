# Implement CNGE scientific engine and reproducible source archive

The repository previously contained only a title. This change adds an executable CPU-first reference implementation of the attached Continuous Neuro-Geometric Epistemic Engine architecture: canonical mechanics, differentiable integrators, Hamiltonian/Lagrangian networks, invariant mining, local parameter protection, bounded symbolic synthesis, optional semantic/camera adapters, online learning, and a live canvas with WebSocket interventions.

The README provides installation, demos, tests and integration instructions. `docs/VALIDATION.md` maps the implementation to the specification and explicitly distinguishes empirical checks from the unproven research and production guarantees in the original request. The oscillator demo uses a known analytical invariant and does not claim autonomous neural discovery. `CNGE_SOURCE.txt` contains the complete versioned text sources, including tests and documentation, with per-file SHA-256 hashes and a deterministic regeneration script.

Validation: see `docs/RESULTS.md` for the executed checks and measured results. Optional GPU, real-camera and downloaded foundation-model integrations are not validated in this environment. The workflow includes CPU CI and reproducible source-export verification.
