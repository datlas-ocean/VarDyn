# GPU 4DVar minimization benchmarks

## Decision

VarDyn maintains two 4DVar Minimizers:

- `scipy`: the Historical SciPy Minimizer and default compatibility path;
- `optax-decoupled`: the Device-Resident Optax Minimizer selected explicitly by
  GPU configurations.

JAXopt and monolithic Optax minimization were experimental benchmark paths.
They are not production options. The monolithic Optax step nested the complete
4DVar Cost-Gradient Evaluation in its line search, producing a serialized
executable larger than 4 GB and paying about 63 seconds of avoidable compilation
on the monthly WHIRLS case.

## Benchmark problem

- configuration: WHIRLS QG;
- Analysis Window: 1 June to 1 July 2025;
- grid: 251 × 501;
- Control Vector: 1,249,496 coefficients;
- checkpoints: 121;
- L-BFGS history: 10;
- JAX: 0.6.2;
- device: an otherwise idle NVIDIA RTX 4090;
- model and compiled cost path: float32 (`USE_FLOAT64=False`).

Each variant ran in a fresh process. Setup, first Cost-Gradient Evaluation,
minimization, and device memory were recorded separately.

## Fixed ten-iteration comparison

| 4DVar Minimizer | Evaluations | Cost compile | Minimization | Numerical total | Warm iteration | GPU peak |
|---|---:|---:|---:|---:|---:|---:|
| Historical SciPy | 12 | 78.62 s | 47.87 s | 126.49 s | 4.39 s | 3,369 MiB |
| scan + SciPy | 12 | 71.38 s | 30.75 s | 102.12 s | 2.80 s | 3,590 MiB |
| monolithic Optax | 12 | 71.09 s | 93.39 s | 164.58 s | 2.76 s | 3,878 MiB |
| Device-Resident Optax | 12 | 71.43 s | 29.92 s | 101.82 s | 2.72 s | 3,864 MiB |

Rolling checkpoint loops with `lax.scan` reduced SciPy minimization time by
about 36%, with the same accepted-iteration and evaluation counts. Decoupling
the Armijo line search removed the 65.82-second first Optax iteration. The
replacement first iteration took 2.72 seconds and the persistent-cache protobuf
error disappeared.

The Device-Resident Optax and scan + SciPy final costs differed by `2.82e-5`
relative after ten iterations.

## Common-convergence comparison

The production candidates were then compared using the same stopping rule:

```text
||g_k||_2 / ||g_0||_2 <= 0.10
for two consecutive accepted iterations,
after at least five iterations,
with maxiter = 50.
```

| Metric | Historical SciPy | Device-Resident Optax |
|---|---:|---:|
| Accepted iterations | 18 | 17 |
| Cost-Gradient Evaluations | 20 | 19 |
| Final relative gradient | 0.0888 | 0.0676 |
| Final cost | 628,849 | 665,055 |
| Minimization | 84.45 s | 48.87 s |
| Numerical total | 161.92 s | 120.66 s |
| Setup plus numerical total | 362.85 s | 323.64 s |
| GPU peak | 3,368 MiB | 3,864 MiB |

The Device-Resident Optax Minimizer reduced minimization time by 42.1% and
numerical cold-start time by 25.5%. Setup still dominated the end-to-end run, so
the full-window improvement was 10.8%, or about 39 seconds.

SciPy performed one extra iteration because its relative gradient at iteration
16 was 0.10105, narrowly above the threshold. At the common iteration 17, the
costs differed by about 1.27%; the larger difference between stopped final costs
is therefore mostly a discrete stopping effect.

## Operational conclusions

- Keep `scipy` as the default to preserve existing configurations.
- Select `optax-decoupled` only with `device_resident_state=True`,
  `jit_cost_and_grad=True`, and `cost_and_grad_schedule='scan'`.
- Use relative gradient stopping for comparisons across 4DVar Minimizers.
- Keep `save_minimization=False` on the Optax hot path. Per-iteration saves
  transfer and write the complete Control Vector; the final `Xres.nc` remains
  available without them.
- Device memory is not limiting on the benchmark GPU; the Optax L-BFGS history
  adds about 0.5 GB.
- Further end-to-end gains should target the roughly 200-second experiment
  setup and the roughly 71-second Cost-Gradient Evaluation compilation.

## Reproducibility

- `mapping/examples/WHIRLS/benchmark_4dvar_iteration.py`
- `mapping/examples/WHIRLS/benchmark_4dvar_minimizers.py`
- `mapping/examples/WHIRLS/benchmark_4dvar_convergence.ipynb`
- `mapping/examples/WHIRLS/benchmark_4dvar_convergence_results.json`
- `mapping/examples/WHIRLS/benchmark_4dvar_minimizers_results.json`
