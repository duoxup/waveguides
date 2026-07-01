# Design: Consolidate per-frequency computation into one vectorized kernel

- **Date:** 2026-07-01
- **Status:** Draft — awaiting review
- **Scope:** `src/waveguides/core.py`, `src/waveguides/heavy_computation.py`, tests
- **Author:** duoxup (with Claude)
- **Follows:** [2026-07-01-vectorize-waveguide-eval-design.md](2026-07-01-vectorize-waveguide-eval-design.md)

## 1. Motivation

The previous change vectorized `heavy_computation.{propagation_factor,impedance,phaseshift}_array`
but deliberately left `core.py`'s scalar methods untouched to serve as an independent test
oracle. The result is two problems this design resolves:

- **Slow `WG` methods.** `WG.{propagation_factor,impedance,phaseshift,wavelength}_at_list`
  are pure Python loops (`np.array([self.x_at(f) for f in f_list])`) — the same per-point
  Python-loop cost the pool approach had. `impedance_at`, `wavelength_at`, and
  `calc_propagation_factor_*` also loop over modes.
- **Duplicated physics.** `heavy_computation` now holds a vectorized copy of the propagation
  factor / impedance formulas, while `core.py` holds the scalar copy. Naively vectorizing
  `core.py` in place would create a *third* copy.

## 2. Goal

One vectorized kernel, in `core.py`, used by both the `WG.*_at` / `*_at_list` methods and the
`heavy_computation.*_array` functions. This removes the duplication and makes every
`*_at_list` method as fast as the vectorized `heavy_computation` path, with the public
signatures and per-call return shapes preserved.

## 3. Architecture

Because `heavy_computation` imports `core` (not the reverse), the shared kernel must live in
`core.py`; `heavy_computation` delegates to it.

```
core.py
  propagation_factor_matrix(wg, fs, lossless=False) -> (M, N)   # the kernel
  impedance_matrix(wg, fs)                          -> (M, N)
  wavelength_matrix(wg, fs)                          -> (M, N)
  _freq_mode_grid(wg, fs)  (private helper)
        ▲                                   ▲
        │ WG.*_at / *_at_list delegate      │ heavy_computation.*_array delegate
```

### 3.1 Kernel (new module-level functions in `core.py`)

- `_freq_mode_grid(wg, fs)` → `(fs_1d, k, kc, beta, mode_type, m1, m2)` with `k=(M,1)`,
  `kc=(1,N)`, `beta=(M,N)` complex128, `mode_type/m1/m2=(1,N)`. Uses the existing cached
  `wg._mode_arrays()`. This is the `_grid` helper currently in `heavy_computation`, moved
  here and extended to also expose `mode_type/m1/m2`.
- `propagation_factor_matrix(wg, fs, lossless=False)` → `(M,N)` complex128. Inlines the
  rec/cir TE/TM `alpha` formulas (moved from `heavy_computation`), selects branches with
  `np.where`. When `lossless`, `r_s = 0` (mirrors the scalar `calc_propagation_factor_*`).
- `impedance_matrix(wg, fs)` → `(M,N)` complex128.
- `wavelength_matrix(wg, fs)` → `(M,N)` complex128, equal to `2*pi / beta` (the current
  `wavelength_at` computes `2*pi / sqrt(k0**2 - kc**2)` per mode).

Phase shift needs no separate kernel: it is `np.angle(propagation_factor_matrix(wg, fs))`
(the argument is loss-independent, so the lossy factor gives the same angle).

These three matrix functions are the module-level public kernel API of `core.py`; the `WG`
methods are thin conveniences over them. (`_freq_mode_grid` stays private.)

### 3.2 `WG` methods (delegate; return shapes unchanged)

| Method | New body | Returns |
|---|---|---|
| `propagation_factor_at(f, lossless=False)` | `propagation_factor_matrix(self, f, lossless)[0]` | `(N,)` |
| `propagation_factor_at_list(f_list, lossless=False)` | `propagation_factor_matrix(self, f_list, lossless)` | `(M,N)` |
| `impedance_at(f)` | `impedance_matrix(self, f)[0]` | `(N,)` |
| `impedance_at_list(f_list)` | `impedance_matrix(self, f_list)` | `(M,N)` |
| `phaseshift_at(f)` | `np.angle(propagation_factor_matrix(self, f))[0]` | `(N,)` |
| `phaseshift_at_list(f_list)` | `np.angle(propagation_factor_matrix(self, f_list))` | `(M,N)` |
| `wavelength_at(f)` | `wavelength_matrix(self, f)[0]` | `(N,)` |
| `wavelength_at_list(f_list)` | `wavelength_matrix(self, f_list)` | `(M,N)` |

`gamma_at` / `gamma_at_list` stay `NotImplementedError` (unchanged).

### 3.3 `heavy_computation` (thin delegates)

```python
def propagation_factor_array(wg, fs, lossless=False):
    return propagation_factor_matrix(wg, fs, lossless)
def impedance_array(wg, fs):
    return impedance_matrix(wg, fs)
def phaseshift_array(wg, fs):
    return np.angle(propagation_factor_matrix(wg, fs))
```

The local `_grid` helper is removed (moved to `core._freq_mode_grid`).

## 4. `lossless` unification (decided)

`propagation_factor` gains `lossless` on **both** sides: `heavy_computation.propagation_factor_array`
gets `lossless=False` to match `WG.propagation_factor_at`. The kernel implements the lossless
path regardless (needed to preserve `propagation_factor_at`), so exposing it is a one-keyword
forward. `impedance`, `wavelength`, and `phaseshift` are loss-independent and take no
`lossless` argument on either side (already consistent).

## 5. Removed / kept

- **Remove:** `calc_propagation_factor_rec`, `calc_propagation_factor_cir` (superseded by the
  kernel); the `_grid` helper in `heavy_computation`.
- **Keep:** `alpha_rec_te/tm`, `alpha_cir_te/tm` (atomic physics formulas; also the atoms the
  test oracle re-derives from), `norm_*` (used by mode construction and field distribution),
  and all mode-construction / field-distribution code.

## 6. Numerics and edge cases

This is a **faithful refactor**: away from degenerate points the kernel reproduces the current
scalar implementation to `< 1e-12` (prototype for the earlier change observed `3.6e-15`). At
the measure-zero exact-cutoff point (`kc == k` bit-exactly) the kernel reproduces the scalar
oracle's degenerate behavior:

- **Lossy, exact cutoff:** `alpha -> inf`, `pf = 0` — kept via the existing guard
  `np.where(ratio2 == 1.0, 0.0, pf)`, which the scalar path also yields (`0`).
- **Lossless, exact cutoff:** the scalar path computes `0 / 0 = nan` (because `r_s = 0` and
  `sqrt(1 - ratio2) = 0`); the kernel reproduces `nan` there. The lossy guard is applied
  **only when `not lossless`**, so lossless is left as `nan` to match the oracle.
- **`phaseshift` at exact cutoff:** defined as `angle(lossy pf)` = `angle(0)` = `0` (a
  measure-zero point where the old `phaseshift_at`, built on the lossless factor, returned
  `nan`). This is the same guarded-lossy behavior; documented, not a concern for sampled data.
- **Evanescent (below cutoff):** complex `beta` via `np.sqrt(x.astype(np.complex128))`,
  identical to the oracle's principal branch.
- **`wavelength` at exact cutoff:** `2*pi / 0` → `inf/nan`, identical to the scalar path.

## 7. Testing

The consolidation removes the previous oracle (`WG.*_at` now routes through the same kernel as
`heavy_computation`, so the existing `heavy_computation`-vs-`WG.*_at` tests would become
tautological). Re-establish independence with a **self-contained scalar reference in the test
file**: plain `(frequency, mode)` double loops that re-derive `pf` / impedance / wavelength
from the retained atomic `alpha_*` functions and scalar `sqrt`. This is structurally
independent of the vectorized assembly (explicit loops vs. broadcasting + `np.where`).

Coverage:

- **All vectorized results vs. the scalar reference** to `< 1e-12`, for `RecWG` and `CirWG`,
  `N ∈ {1, 40, 800}`, frequencies spanning below / near / above cutoff.
- **Both entry points** per quantity: `WG.*_at(f)` → `(N,)`, `WG.*_at_list(fs)` → `(M,N)`,
  and `heavy_computation.*_array(wg, fs)` → `(M,N)`; scalar / list / empty (`(0,N)`) inputs.
- **`lossless` for propagation factor** on both sides: `lossless=True` and `False` vs. the
  scalar reference; confirm `heavy_computation.propagation_factor_array` now accepts `lossless`.
- **`wavelength`** vs. scalar reference (new coverage).
- **Edge pins:** lossy exact-cutoff `pf == 0`; lossless exact-cutoff `pf` is `nan`
  (documented behavior); `phaseshift` unaffected by `lossless` (loss-independence).
- Update the existing `tests/test_vectorized_eval.py` oracle to the scalar reference; add
  `WG`-method and `wavelength` tests (a `tests/test_core_vectorized.py` module, or extend the
  existing file — the implementation plan decides).

## 8. Out of scope (YAGNI)

- Mode-construction / sorting loops (`calc_sorted_mode_info_*`, `mode_info_array`) — one-time
  setup cost, low payoff, higher risk.
- `gamma_at` / `gamma_at_list` — remain `NotImplementedError`.
- Field-distribution code — already vectorized over grid points.

## 9. Expected outcome

`WG.{propagation_factor,impedance,phaseshift,wavelength}_at_list` drop from Python-loop speed
(hundreds of µs/point) to the vectorized rate (single-digit µs/point at scale), the physics
lives in exactly one place, and `heavy_computation` becomes a thin, pool-free delegation layer.
Public signatures and return shapes are unchanged except for the added `lossless` on
`heavy_computation.propagation_factor_array`.
