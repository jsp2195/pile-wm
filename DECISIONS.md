# Decisions and experiments

- The repository initially contained only LICENSE. The user explicitly requested implementation, so source changes are authorized despite the earlier onboarding-only scope.
- Python 3.11, uv lockfile, dataclasses and YAML; package/artifact caches live under /workspace because the home directory is read-only in the cloud machine.
- No CUDA device is available. Validate CPU smoke; small GPU experiments and full experiments remain unrun.
- Simulator uses simultaneous Jacobi disc contact projection with per-particle contact-count damping, wall clamping, and five substeps of at most 0.008 table units. Diagonal actions are norm-limited to 0.04. Contacts are geometric and overdamped, without inertia or an assumed friction law.
- Initial configurations mix 20% scatter with 1–3 Gaussian clusters, then settle with 300 projection iterations. State IDs preserve particle count.
- Renderer tiles particles to bound temporary memory and runs with native PyTorch on the selected device. Ground-truth evaluation maps use bilinear count splatting, making particle-density mass explicit.
- Smoke is a plumbing/measurement budget, not evidence of learned physical competence. No encoder substitution is permitted; tiny test stubs will be limited to unit tests.
- Milestone 1 validation: `python -m pytest -q` passed 4 tests on CPU; `python -m scripts.random_push --config configs/smoke.yaml` produced a 40-frame GIF. PyTorch 2.14.1 reports CUDA and MPS unavailable. Installed the normal portable PyTorch package (including CUDA runtime dependencies) so the lockfile also supports compatible GPU hosts.
