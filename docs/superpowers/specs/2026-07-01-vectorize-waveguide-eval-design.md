# Design: Vectorized waveguide evaluation for adaptive frequency sampling

- **Date:** 2026-07-01
- **Status:** Draft — awaiting review
- **Scope:** `src/waveguides/heavy_computation.py`, `src/waveguides/core.py`
- **Author:** duoxup (with Claude)

## 1. Motivation

`heavy_computation.py` exposes `propagation_factor_array`, `phaseshift_array`, and
`impedance_array`. Each takes **all frequency points at once** (`fs: Sequence[float]`)
and fills an `(M, N)` matrix (`M` = frequencies, `N` = modes) by dispatching every
`(frequency, mode)` cell as an independent task to a `multiprocessing.Pool`.

A downstream package now needs the same waveguide quantities but drives the evaluation
with **adaptive frequency sampling**: it decides frequencies incrementally, one new
point per refinement iteration, and therefore cannot supply the full `fs` up front.

Two questions motivated this design:

1. How much slower is per-point (adaptive) access than the batched path?
2. Is there a more elegant interface that serves both batched and incremental use?

## 2. Findings (measured, not assumed)

The per-cell work is tiny: a few scalar ops, one `scimath.sqrt`, a closed-form `alpha`
(no runtime Bessel evaluation — zeros are precomputed), and one `exp`. The current
design is therefore **bound by argument marshalling and IPC, not by physics**: building
the object array of per-cell tuples via `np.fromfunction(np.vectorize(...))` and pickling
tasks to/from workers dominates.

### 2.1 Small case — N = 40 modes

Batch of `M = 2000`; one-at-a-time over 200 points. Per-point cost:

| Strategy | µs/point | vs batch+pool |
|---|---|---|
| batch + pool (intended) | 277 | 1× |
| batch serial (no pool) | 518 | 1.9× |
| per-freq + pool (naive adaptive) | **1151** | **4.2× (worst)** |
| per-freq serial API | 546 | 2.0× |
| per-freq `wg.propagation_factor_at` (in-process) | 482 | 1.7× |
| **vectorized, batch** | **1.5** | **≈180× faster** |
| **vectorized, one-at-a-time** | **57** | **≈5× faster** |

At small `N`, `chunksize=64 > N` collapses each per-frequency pool call onto a single
worker, so naive per-point pool use is the *worst* option.

### 2.2 Typical case — N = 800 modes, M = 100 frequencies

(16-core host, pool = 8 processes.)

Batch (all 100 points at once):

| Strategy | total | µs/point |
|---|---|---|
| batch + pool | 559 ms | 5588 |
| batch serial | 1119 ms | 11194 |
| **vectorized batch** | **2.8 ms** | **28** |

Adaptive (one point per iteration, 100 points):

| Strategy | total | µs/point |
|---|---|---|
| per-freq + pool | 417 ms | 4174 |
| per-freq serial API | 1117 ms | 11171 |
| `wg.propagation_factor_at` (in-process) | 1029 ms | 10290 |
| **vectorized one-at-a-time** | **9.9 ms** | **99** |

Two regime-specific observations:

- At `N = 800 ≫ 64`, a single per-frequency pool call already saturates the workers, so
  **adaptive-via-pool is not slower than batch** (0.42 s vs 0.56 s). The original worry —
  "adaptive is penalized because it can't submit all frequencies" — does **not** hold at
  this scale even with the existing code.
- The pool yields only ~2× over serial (8 processes), confirming the workload is
  marshalling/IPC-bound, not compute-bound.

### 2.3 Correctness of the vectorized kernel

A prototype vectorized propagation factor matched the existing implementation to
`max |Δ| = 3.6e-15` (double-precision round-off) across both cases.

## 3. Goal

Provide a single evaluation path that:

- serves batched and incremental (adaptive, one-point-per-iteration) use equally well;
- is dramatically faster in both (≈40–200× on the typical case);
- removes the `multiprocessing` machinery (pool, `chunksize`, picklability constraints);
- preserves the existing public names, return shapes, and numerics (downstream package
  needs **zero changes** to keep working).

## 4. Approaches considered

**Approach A — Full vectorization, drop the pool (recommended).**
Reimplement the three quantities as NumPy operations broadcast over the frequency axis:
`k, r_s` as `(M, 1)`, mode parameters (`kc, mode_type, m/n`) as `(1, N)`; compute
`beta / alpha / Z` on the full `(M, N)` grid, selecting TE/TM branches with masks.
- *Pros:* one implementation optimal for both batch and single-point; deletes IPC
  complexity; ~40–200× faster; verified numerically identical; backward compatible.
- *Cons:* the four `alpha` formulas + impedance + cutoff handling must be vectorized
  carefully and pinned by regression tests.

**Approach B — Keep the pool; point adaptive callers at `wg.propagation_factor_at`.**
- *Pros:* zero code change, zero risk.
- *Cons:* still ~10 ms → ~1 s per sweep (≈100–300× slower than A); leaves the root cause
  (pool is wrong for tiny tasks) in place; two parallel code paths.

**Approach C — Stateful evaluator with a frequency cache.**
- *Cons:* the sampler uses one *new* point per iteration and does not re-query, so caching
  is YAGNI; adds state and invalidation complexity. Rejected.

**Decision: Approach A.**

## 5. Vectorization technique

Two independent forms of per-mode variation, handled two different ways:

1. **Continuous per-mode parameters** (`kc, m, n`): expressed as length-`N` arrays shaped
   `(1, N)`. Broadcasting against `k`-side `(M, 1)` arrays fills the `(M, N)` grid so that
   cell `[i, j]` automatically uses `(f_i, mode_j)`. No branching — this replaces the
   double `for` loop.

2. **Discrete formula branches** (TE vs TM — structurally different formulas): computed as
   *compute-both-then-select*. Evaluate the TE formula and the TM formula over the whole
   `(M, N)` grid, then `alpha = np.where(mode_type[None,:] > 0, alpha_te, alpha_tm)`.
   Nested scalar conditionals (`eps_m = 2 if m == 0 else 1`) become `np.where(m == 0, 2, 1)`
   on `(1, N)` and are constant per waveguide (precomputable once).

### 5.1 Branch inventory — "how many times is each cell computed?"

Branches are *structural* and **do not grow with `M` or `N`**.

| Quantity | Top-level branch (doubles `(M, N)` work) | Internal small branches (`(1, N)`, precomputable) |
|---|---|---|
| propagation factor · rec | `mode_type` TE/TM → 2 blocks | `eps_m` (`m==0?`), `eps_n` (`n==0?`) |
| propagation factor · cir | `mode_type` TE/TM → 2 blocks | none |
| impedance (rec = cir) | `mode_type` TE/TM → 2 blocks | none |
| phaseshift | **none** | none |

Per `(M, N)` pass count for the heaviest quantity (rec propagation factor):
`k, r_s, beta, exp` each run **once**; only `alpha` runs **twice** (TE block + TM block).
So the redundant work is a single extra `alpha` evaluation — roughly +30–50% over a
zero-waste implementation, against a baseline already ~180× faster than the pool.

`phaseshift = angle(propagation_factor)` depends only on `Re(beta)`, independent of
`alpha` and loss, so it reduces to `-Re(beta) * l` (wrapped to `(-π, π]` by `np.angle`)
with **no branch**. Implementation reuses `np.angle` of the vectorized propagation factor
to guarantee identical wrapping to the current output.

*Optional micro-optimization (not required):* replace `np.where` with masked assignment
(`alpha[:, te] = ...; alpha[:, tm] = ...`) to compute each formula only on its own mode
columns — exactly 1× work, no waste. Deferred: the `(M, N)` grid is already sub-10-ms.

## 6. Interface

Public names and return shapes are **unchanged** so the downstream package needs no edits.

```python
propagation_factor_array(wg, fs, *, pool=None, chunksize=64) -> np.ndarray  # (M, N)
phaseshift_array(wg, fs, *, pool=None, chunksize=64)        -> np.ndarray  # (M, N)
impedance_array(wg, fs, *, pool=None, chunksize=64)         -> np.ndarray  # (M, N)
```

- `fs` accepts a scalar or any 1-D array-like; internally coerced with `np.atleast_1d`.
  A scalar or length-1 input returns shape `(1, N)`.
- `pool` and `chunksize` are **retained but ignored**. If `pool is not None`, emit a
  `DeprecationWarning` ("pool is ignored; evaluation is now vectorized") once. They are
  kept only so existing call sites do not break; they may be removed in a future major.
- Empty `fs` returns an empty `(0, N)` array (the current pool path raises on empty; the
  vectorized path can return the natural empty result — this is a deliberate, documented
  behavior change).

### 6.1 Internal structure

- A single vectorized kernel per (cross-section, quantity) lives in `core.py`, beside the
  existing scalar helpers and the mode data (avoids a `core ← heavy_computation` import
  cycle).
- `heavy_computation.*_array` become thin wrappers over these kernels.
- `WG.propagation_factor_at_list`, `impedance_at_list`, `phaseshift_at_list` are redirected
  to the same kernels (they currently do a Python `for`-loop over scalar calls). This keeps
  one source of truth and speeds up those methods too. The scalar `*_at(f)` methods are
  retained (useful as the test oracle) and may internally call the kernel with a length-1
  array.
- Mode parameter arrays (`kc, mode_type, mode_num1, mode_num2` as `(N,)`) are computed once
  and cached on the `WG` instance (mode list is fixed after construction). This cache is
  what makes single-point calls cheap (~99 µs at `N = 800`).

## 7. Correctness and edge cases

- **Below cutoff (evanescent):** `numpy.lib.scimath.sqrt(k**2 - kc**2)` yields complex
  `beta` — identical to the current workers.
- **At cutoff (`beta = 0`):** TE impedance `k/beta → inf`; preserved to match current
  behavior (no special-casing introduced).
- **Unused-branch invalid values:** `np.where` evaluates both branches, so the "wrong"
  formula may produce `inf/nan` for some cells (then discarded). Wrap the alpha block in
  `np.errstate(divide="ignore", invalid="ignore")`. Selected modes never hit a real
  zero denominator (TM modes have `m,n ≥ 1`; TE `m=n=0` is excluded via `kc = nan` sort).
- **Dtype/shape contract:** complex128 output, shape `(M, N)`, `M ≥ 0`.

## 8. Testing

- **Oracle:** the existing scalar methods `wg.propagation_factor_at(f)` / `impedance_at(f)`
  and the current pool-based `*_array`, treated as ground truth.
- **Parametrization:** `RecWG` and `CirWG`; `N ∈ {1, 40, 800}`; `fs` spanning below-cutoff,
  near-cutoff, and above-cutoff points; scalar, length-1, and multi-point inputs.
- **Assertions:** `max |Δ| < 1e-12` vs oracle (observed 3.6e-15); shape and dtype contract;
  `phaseshift` matches `np.angle(propagation_factor)` (wrapped `-Re(beta) * l`);
  empty-input returns `(0, N)`; `DeprecationWarning` emitted when `pool` is passed.

## 9. Out of scope (YAGNI)

- No frequency memoization/caching (sampler uses one new point per iteration, no repeats).
- No immediate deletion of the pool code beyond turning it into an ignored no-op.
- No changes to field-distribution (`get_mode_efield_distribution_at_gridpoints`) code.

## 10. Expected outcome

For the typical 800-mode × 100-point sweep, the full adaptive scan drops from ~0.42 s
(current best, per-freq + pool) to ~0.01 s, with the pool and its picklability
constraints removed and the downstream package requiring no changes.
