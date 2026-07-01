# Vectorized Waveguide Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the multiprocessing-pool per-cell evaluation in `heavy_computation.py` with NumPy vectorization over the frequency axis, so both batched and one-point-at-a-time adaptive sampling are fast, and delete the legacy pool code.

**Architecture:** Add a cached per-mode parameter-array accessor on `WG`. Reimplement the three public `*_array` functions as fully vectorized NumPy expressions that broadcast `(M,1)` frequency-side quantities against `(1,N)` mode-side quantities and select TE/TM formula branches with `np.where`. The existing scalar `WG.*_at(f)` methods are left untouched and serve as the independent test oracle. Legacy pool workers stay only as a dev-time reference and are deleted in the final task.

**Tech Stack:** Python ≥3.10 (dev env 3.13), NumPy, SciPy, pytest 9.

## Global Constraints

- Dependencies limited to `numpy` + `scipy` (already declared); add no new runtime deps.
- Public function names kept exactly: `propagation_factor_array`, `phaseshift_array`, `impedance_array`. No `pool` / `chunksize` parameters.
- Output contract: shape `(M, N)` where `M = len(atleast_1d(fs))`, dtype `complex128`.
- Numerics must match the scalar oracle to `max |Δ| < 1e-12` (prototype observed `3.6e-15`).
- `numpy.lib.scimath.sqrt(x)` and `np.sqrt(x.astype(np.complex128))` agree on the principal branch; use the latter to guarantee a complex dtype.
- ruff line-length = 100.
- Tests run from repo root: `python -m pytest tests/test_vectorized_eval.py -v` (package is editable-installed).

## File Structure

- **Modify** `src/waveguides/core.py` — add `WG._mode_arrays()` cached method (pure addition; no existing behavior changes). Insert after `mode_info_array()` (~core.py:430).
- **Modify** `src/waveguides/heavy_computation.py` — add `_grid()` helper; reimplement the three public functions vectorized; remove `pool`/`chunksize`. Legacy workers/dispatch/arg-builders/alias remain until Task 5, then deleted.
- **Create** `tests/test_vectorized_eval.py` — regression tests vs the scalar oracle plus shape/dtype/contract tests.

---

### Task 1: Cached per-mode parameter arrays on `WG`

**Files:**
- Modify: `src/waveguides/core.py` (add method to class `WG`, after `mode_info_array`, ~line 430)
- Test: `tests/test_vectorized_eval.py`

**Interfaces:**
- Consumes: `WG.mode_info_list` (list of `ModeInfo`, fixed after construction).
- Produces: `WG._mode_arrays() -> dict[str, np.ndarray]` with keys `"kc"` (float `(N,)`), `"mode_type"` (int `(N,)`), `"mode_num1"` (int `(N,)`), `"mode_num2"` (int `(N,)`); memoized on the instance.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vectorized_eval.py`:

```python
import numpy as np
import pytest

from waveguides import RecWG, CirWG
from waveguides.heavy_computation import (
    propagation_factor_array,
    impedance_array,
    phaseshift_array,
)


def _rec(N):
    return RecWG(a=0.02286, b=0.01016, l=0.1, N=N, er=1.0, sigma=5.8e7)


def _cir(N):
    return CirWG(r=0.01, l=0.1, N=N, er=1.0, sigma=5.8e7)


REC_FS = np.array([7e9, 10e9, 15e9])
CIR_FS = np.array([9e9, 12e9, 20e9])


def test_mode_arrays_values_and_cache():
    wg = _rec(5)
    ma = wg._mode_arrays()
    assert set(ma) == {"kc", "mode_type", "mode_num1", "mode_num2"}
    assert ma["kc"].shape == (5,)
    assert list(ma["mode_type"]) == [m.mode_type for m in wg.mode_info_list]
    assert list(ma["mode_num1"]) == [m.mode_num1 for m in wg.mode_info_list]
    assert list(ma["mode_num2"]) == [m.mode_num2 for m in wg.mode_info_list]
    np.testing.assert_allclose(ma["kc"], [m.kc for m in wg.mode_info_list])
    assert wg._mode_arrays() is ma  # memoized: same object on second call
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vectorized_eval.py::test_mode_arrays_values_and_cache -v`
Expected: FAIL with `AttributeError: 'RecWG' object has no attribute '_mode_arrays'`

- [ ] **Step 3: Write minimal implementation**

In `src/waveguides/core.py`, inside class `WG`, immediately after the `mode_info_array` method:

```python
    def _mode_arrays(self):
        """Cached per-mode parameter arrays for vectorized evaluation.

        Returns a dict of length-N arrays: ``kc``, ``mode_type``,
        ``mode_num1``, ``mode_num2``. The mode list is fixed after
        construction, so the result is memoized on the instance.
        """
        cache = getattr(self, "_mode_arrays_cache", None)
        if cache is None:
            mi = self.mode_info_list
            cache = {
                "kc": np.array([m.kc for m in mi], dtype=float),
                "mode_type": np.array([m.mode_type for m in mi], dtype=int),
                "mode_num1": np.array([m.mode_num1 for m in mi], dtype=int),
                "mode_num2": np.array([m.mode_num2 for m in mi], dtype=int),
            }
            self._mode_arrays_cache = cache
        return cache
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vectorized_eval.py::test_mode_arrays_values_and_cache -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/waveguides/core.py tests/test_vectorized_eval.py
git commit -m "feat(core): add cached WG._mode_arrays() for vectorized eval"
```

---

### Task 2: Vectorized `propagation_factor_array`

**Files:**
- Modify: `src/waveguides/heavy_computation.py` (add `_grid`; replace `propagation_factor_array`, ~lines 195-207)
- Test: `tests/test_vectorized_eval.py`

**Interfaces:**
- Consumes: `WG._mode_arrays()` (Task 1); `wg.cross_tag`, `wg.er`, `wg.sigma`, `wg.l`, and `wg.a`/`wg.b` (rec) or `wg.r` (cir); `C_LIGHT` from `.core`.
- Produces:
  - `_grid(wg, fs) -> (ma, fs, k, kc, beta)` where `ma = wg._mode_arrays()`, `fs` is 1-D `(M,)` float, `k` is `(M,1)` float, `kc` is `(1,N)` float, `beta` is `(M,N)` complex128.
  - `propagation_factor_array(wg, fs) -> np.ndarray` complex128 `(M, N)`. `fs` accepts scalar or 1-D array-like. No `pool`/`chunksize`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vectorized_eval.py`:

```python
@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
@pytest.mark.parametrize("N", [1, 40, 800])
def test_propagation_factor_matches_oracle(make_wg, fs, N):
    wg = make_wg(N)
    got = propagation_factor_array(wg, fs)
    ref = np.array([wg.propagation_factor_at(f) for f in fs])
    assert got.shape == (len(fs), N)
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_propagation_factor_scalar_returns_2d():
    wg = _rec(10)
    out = propagation_factor_array(wg, 10e9)
    assert out.shape == (1, 10)
    ref = wg.propagation_factor_at(10e9)
    np.testing.assert_allclose(out[0], ref, rtol=1e-9, atol=1e-12)


def test_propagation_factor_empty_returns_0xN():
    wg = _rec(10)
    out = propagation_factor_array(wg, [])
    assert out.shape == (0, 10)


def test_propagation_factor_no_pool_kwarg():
    wg = _rec(4)
    with pytest.raises(TypeError):
        propagation_factor_array(wg, [10e9], pool=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vectorized_eval.py -k propagation_factor -v`
Expected: FAIL overall. Specifically: `test_propagation_factor_scalar_returns_2d` fails (old impl calls `len(fs)` on a scalar → `TypeError`), `test_propagation_factor_empty_returns_0xN` fails (old impl raises `ValueError` on empty), and `test_propagation_factor_no_pool_kwarg` fails (old signature still accepts `pool`). The parametrized oracle-equality tests may already pass because the old serial implementation returns the same values — that is fine; the new contract tests are what drive this task.

- [ ] **Step 3: Write minimal implementation**

In `src/waveguides/heavy_computation.py`, add the `_grid` helper in the "Internal helpers" section and replace the body of `propagation_factor_array`:

```python
def _grid(wg, fs):
    """Shared frequency/mode grid for vectorized evaluation.

    Returns (mode_arrays, fs_1d, k, kc, beta) with k shaped (M,1),
    kc shaped (1,N), beta shaped (M,N) complex128.
    """
    ma = wg._mode_arrays()
    fs = np.atleast_1d(np.asarray(fs, dtype=float))
    k = (2 * np.pi * fs / C_LIGHT * np.sqrt(wg.er))[:, None]
    kc = ma["kc"][None, :]
    beta = np.sqrt((k**2 - kc**2).astype(np.complex128))
    return ma, fs, k, kc, beta
```

```python
def propagation_factor_array(wg: WG, fs) -> np.ndarray:
    """Complex propagation factor exp(-(alpha + j*beta)*l) for each mode of
    *wg* over frequencies *fs*.

    *fs* may be a scalar or a 1-D array-like. Returns a complex array of
    shape (len(fs), N), N = number of modes. Fully vectorized (no pool).
    """
    ma, fs, k, kc, beta = _grid(wg, fs)
    mode_type = ma["mode_type"][None, :]
    m1 = ma["mode_num1"][None, :].astype(float)
    m2 = ma["mode_num2"][None, :].astype(float)
    r_s = np.sqrt(np.pi * fs * 4 * np.pi * 1e-7 / wg.sigma)[:, None]
    ratio2 = (kc / k) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        root = np.sqrt((1 - ratio2).astype(np.complex128))
        if wg.cross_tag == "rec":
            a, b = wg.a, wg.b
            eps_m = np.where(m1 == 0, 2.0, 1.0)
            eps_n = np.where(m2 == 0, 2.0, 1.0)
            denom = m1**2 * b**2 + m2**2 * a**2
            alpha_te = (
                1.0 / (60 * np.pi) * r_s / (eps_m * eps_n * root)
                * (ratio2 * (eps_m / b + eps_n / a)
                   + (1 - ratio2) * (m1**2 * b + m2**2 * a) / denom)
            )
            alpha_tm = (
                r_s / root * (m1**2 * b**3 + m2**2 * a**3) / denom
                / (60 * np.pi * a * b)
            )
        elif wg.cross_tag == "cir":
            rc = wg.r
            alpha_te = (
                r_s / root * (m1**2 / (kc**2 * rc**2 - m1**2) + ratio2)
                / (120 * np.pi * rc)
            )
            alpha_tm = r_s / root / (120 * np.pi * rc)
        else:
            raise ValueError(f"Unknown waveguide cross_tag: {wg.cross_tag!r}")
        alpha = np.where(mode_type > 0, alpha_te, alpha_tm)
    return np.exp(-(np.imag(beta) + np.abs(alpha) + 1j * np.real(beta)) * wg.l)
```

Note: leave the legacy `_build_pf_args`, `_pf_worker_*`, `_dispatch`, `_select_pf_worker` functions in place for now (deleted in Task 5). Remove `pool`/`chunksize` from this function's signature only.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vectorized_eval.py -k propagation_factor -v`
Expected: PASS (all parametrized + scalar + empty + no_pool_kwarg)

- [ ] **Step 5: Commit**

```bash
git add src/waveguides/heavy_computation.py tests/test_vectorized_eval.py
git commit -m "feat(heavy_computation): vectorize propagation_factor_array, drop pool"
```

---

### Task 3: Vectorized `impedance_array`

**Files:**
- Modify: `src/waveguides/heavy_computation.py` (replace `impedance_array`, ~lines 223-233)
- Test: `tests/test_vectorized_eval.py`

**Interfaces:**
- Consumes: `_grid(wg, fs)` (Task 2); `wg.er`; `ma["mode_type"]`.
- Produces: `impedance_array(wg, fs) -> np.ndarray` complex128 `(M, N)`. No `pool`/`chunksize`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vectorized_eval.py`:

```python
@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
@pytest.mark.parametrize("N", [1, 40, 800])
def test_impedance_matches_oracle(make_wg, fs, N):
    wg = make_wg(N)
    got = impedance_array(wg, fs)
    ref = np.array([wg.impedance_at(f) for f in fs])
    assert got.shape == (len(fs), N)
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-6)


def test_impedance_no_pool_kwarg():
    wg = _rec(4)
    with pytest.raises(TypeError):
        impedance_array(wg, [10e9], pool=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vectorized_eval.py -k impedance -v`
Expected: FAIL — `test_impedance_no_pool_kwarg` fails (old signature still accepts `pool`).

- [ ] **Step 3: Write minimal implementation**

In `src/waveguides/heavy_computation.py`, replace the body of `impedance_array`:

```python
def impedance_array(wg: WG, fs) -> np.ndarray:
    """Wave-impedance matrix for *wg* over frequencies *fs*.

    *fs* may be a scalar or 1-D array-like. Returns a complex array of
    shape (len(fs), N). Fully vectorized (no pool).
    """
    ma, fs, k, kc, beta = _grid(wg, fs)
    mode_type = ma["mode_type"][None, :]
    eta = 120 * np.pi / np.sqrt(wg.er)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(mode_type > 0, k / beta * eta, beta / k * eta)
    return Z
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vectorized_eval.py -k impedance -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/waveguides/heavy_computation.py tests/test_vectorized_eval.py
git commit -m "feat(heavy_computation): vectorize impedance_array, drop pool"
```

---

### Task 4: Vectorized `phaseshift_array`

**Files:**
- Modify: `src/waveguides/heavy_computation.py` (replace `phaseshift_array`, ~lines 210-220)
- Test: `tests/test_vectorized_eval.py`

**Interfaces:**
- Consumes: `propagation_factor_array(wg, fs)` (Task 2).
- Produces: `phaseshift_array(wg, fs) -> np.ndarray` real `(M, N)`, equal to `np.angle(propagation_factor_array(wg, fs))`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vectorized_eval.py`:

```python
@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
@pytest.mark.parametrize("N", [1, 40, 800])
def test_phaseshift_matches_oracle(make_wg, fs, N):
    wg = make_wg(N)
    got = phaseshift_array(wg, fs)
    ref = np.angle(np.array([wg.propagation_factor_at(f) for f in fs]))
    assert got.shape == (len(fs), N)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_phaseshift_no_pool_kwarg():
    wg = _rec(4)
    with pytest.raises(TypeError):
        phaseshift_array(wg, [10e9], pool=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vectorized_eval.py -k phaseshift -v`
Expected: FAIL — `test_phaseshift_no_pool_kwarg` fails (old signature still accepts `pool`).

- [ ] **Step 3: Write minimal implementation**

In `src/waveguides/heavy_computation.py`, replace the body of `phaseshift_array`:

```python
def phaseshift_array(wg: WG, fs) -> np.ndarray:
    """Phase shift (rad) for each mode of *wg* over frequencies *fs*.

    Defined as angle(propagation_factor) = -beta*l wrapped to (-pi, pi].
    *fs* may be a scalar or 1-D array-like. Returns shape (len(fs), N).
    """
    return np.angle(propagation_factor_array(wg, fs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vectorized_eval.py -k phaseshift -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/waveguides/heavy_computation.py tests/test_vectorized_eval.py
git commit -m "feat(heavy_computation): vectorize phaseshift_array, drop pool"
```

---

### Task 5: Delete legacy pool code and unused imports

**Files:**
- Modify: `src/waveguides/heavy_computation.py` (remove legacy block, alias, unused imports)
- Test: `tests/test_vectorized_eval.py` (full run, unchanged)

**Interfaces:**
- Consumes: nothing new.
- Produces: `heavy_computation.py` containing only `_grid`, `propagation_factor_array`, `impedance_array`, `phaseshift_array`, and their imports.

- [ ] **Step 1: Confirm no in-repo callers of legacy symbols**

Run:
```bash
grep -rn --include=*.py -E "_pf_worker|_imp_worker|_dispatch|_build_pf_args|_build_imp_args|_results_to_matrix_auto_shape|_select_pf_worker|impedance_array_multifreq" src tools | grep -v "src/waveguides/heavy_computation.py"
```
Expected: no output (only definitions inside `heavy_computation.py`).

- [ ] **Step 2: Delete the legacy definitions**

In `src/waveguides/heavy_computation.py`, delete these functions entirely: `_build_pf_args`, `_build_imp_args`, `_results_to_matrix_auto_shape`, `_pf_worker_cir`, `_pf_worker_rec`, `_imp_worker`, `_dispatch`, `_select_pf_worker`, and the deprecated alias `impedance_array_multifreq`. Also delete the `ComplexLike = Union[...]` type alias.

- [ ] **Step 3: Fix imports to only what remains**

Replace the top-of-file imports so the header reads exactly:

```python
from __future__ import annotations

import numpy as np

from .core import WG, C_LIGHT
```

(Removes `warnings`, the `typing` imports, `numpy.lib.scimath.sqrt`, and the `alpha_*` imports — all now unused. The `_pf_worker`/`_imp_worker` docstring section headers may be removed too.)

- [ ] **Step 4: Lint for anything left unused**

Run: `ruff check src/waveguides/heavy_computation.py`
Expected: no errors (in particular no `F401` unused-import or `F811`/`F821`).

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest tests/test_vectorized_eval.py -v`
Expected: PASS — every test still passes; the oracle (`WG.*_at`) is independent of the deleted legacy code.

- [ ] **Step 6: Commit**

```bash
git add src/waveguides/heavy_computation.py
git commit -m "refactor(heavy_computation): remove legacy pool code and unused imports"
```

---

## Notes / deliberate deviations from the spec

- **`WG.*_at_list` not redirected.** §6.1 of the design suggested redirecting `propagation_factor_at_list` / `impedance_at_list` / `phaseshift_at_list` to the new kernels for a single source of truth. This plan intentionally leaves the scalar `WG.*_at` / `*_at_list` methods untouched so they remain an *independent* oracle for the regression tests. Redirecting them is a safe follow-up once the vectorized path is trusted; it is out of scope here.

## Self-Review

- **Spec coverage:** §3 goals → Tasks 2-5 (vectorized, no pool, legacy removed). §5 technique (broadcasting + `np.where` masks) → Task 2/3 implementations. §6 interface (names kept, kwargs dropped, scalar/1-D `fs`, empty→`(0,N)`) → Tasks 2-4 + contract tests. §6.1 mode-array cache → Task 1. §6.2 staged legacy removal → Task 5 (and the dev-time reference is simply the code present until Task 5). §7 edge cases (evanescent via complex sqrt; cutoff via `errstate`) → Task 2/3. §8 testing (oracle = `WG.*_at`; rec+cir; N∈{1,40,800}; below/near/above cutoff via mixed-mode freqs; scalar/multi/empty) → all test steps. §9 YAGNI (no caching of freqs, no field-code changes) → respected.
- **Placeholder scan:** none — every code and command step is concrete.
- **Type consistency:** `_grid` returns `(ma, fs, k, kc, beta)` and is consumed identically in Tasks 2 and 3; `_mode_arrays()` dict keys (`kc`, `mode_type`, `mode_num1`, `mode_num2`) are used verbatim in `_grid` and `propagation_factor_array`; public function names match the spec and the imports in the test file.
