# Core Vectorization Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all per-frequency waveguide computation into one vectorized kernel in `core.py` that both the `WG.*_at` / `*_at_list` methods and the `heavy_computation.*_array` functions delegate to, removing duplicated physics and the Python loops.

**Architecture:** Add module-level vectorized kernels (`propagation_factor_matrix`, `impedance_matrix`, `wavelength_matrix`, plus a private `_freq_mode_grid`) to `core.py`. Rewire the `WG` methods and `heavy_computation` to call them. The scalar physics atoms (`alpha_*`) stay and become the atoms of an independent scalar test oracle; the scalar assemblers (`calc_propagation_factor_*`) and `heavy_computation._grid` are removed.

**Tech Stack:** Python ≥3.10 (dev 3.13), NumPy, pytest 9.

## Global Constraints

- Dependencies limited to `numpy` + `scipy`; add no new runtime deps.
- Kernel API in `core.py` (module-level): `propagation_factor_matrix(wg, fs, lossless=False)`, `impedance_matrix(wg, fs)`, `wavelength_matrix(wg, fs)` — each returns `(M, N)` `complex128` where `M = len(atleast_1d(fs))`. Private helper `_freq_mode_grid(wg, fs)`.
- `WG` method return shapes unchanged: `*_at(f) -> (N,)`, `*_at_list(fs) -> (M,N)`. `propagation_factor_at`/`_at_list` keep `lossless=False`. `gamma_at`/`gamma_at_list` stay `NotImplementedError`.
- `heavy_computation` public names unchanged; `propagation_factor_array` gains `lossless=False`; `impedance_array`/`phaseshift_array` unchanged signatures.
- Numerics match an independent scalar oracle to `< 1e-12` away from exact cutoff.
- Edge cases (faithful to the scalar path): lossy exact cutoff `pf = 0` (guard applied only when `not lossless`); lossless exact cutoff `pf = nan`; `phaseshift` at exact cutoff `= 0`; `wavelength` at exact cutoff `= inf/nan`; evanescent via `np.sqrt(x.astype(np.complex128))`.
- Keep `alpha_rec_te/tm`, `alpha_cir_te/tm`, `norm_*`, mode-construction and field code. Remove `calc_propagation_factor_rec`, `calc_propagation_factor_cir`, and `heavy_computation._grid`.
- ruff line-length = 100. Tests run from repo root: `python -m pytest -q`.

## File Structure

- **Modify** `src/waveguides/core.py` — add the kernel section (`_freq_mode_grid` + three `*_matrix` functions); rewrite the eight `WG` `*_at`/`*_at_list` methods to delegate; delete `calc_propagation_factor_rec` and `calc_propagation_factor_cir`.
- **Modify** `src/waveguides/heavy_computation.py` — the three `*_array` functions delegate to the core kernel; `propagation_factor_array` gains `lossless`; delete `_grid`; fix imports.
- **Modify** `tests/test_vectorized_eval.py` — add an independent scalar oracle (`scalar_pf`, `scalar_impedance`, `scalar_wavelength`); repoint existing tests to it; add kernel, `WG`-method, `wavelength`, and `lossless` tests.

---

### Task 1: Independent scalar oracle; repoint existing tests

Test-only change. It re-implements the current scalar physics as a self-contained oracle in the test file and points the existing equality tests at it, so that once the kernel later powers both entry points the tests stay meaningful (not kernel-vs-kernel). Production code is untouched; all tests still pass.

**Files:**
- Modify: `tests/test_vectorized_eval.py`

**Interfaces:**
- Consumes: `waveguides.core.{alpha_rec_te,alpha_rec_tm,alpha_cir_te,alpha_cir_tm,C_LIGHT}` (existing module-level names); `wg.mode_info_list`, `wg.{a,b,r,er,sigma,l,cross_tag,N}`.
- Produces (in the test module): `scalar_pf(wg, f, lossless=False) -> (N,) complex`, `scalar_impedance(wg, f) -> (N,) complex`, `scalar_wavelength(wg, f) -> (N,) complex`.

- [ ] **Step 1: Add the scalar oracle helpers and update imports**

At the top of `tests/test_vectorized_eval.py`, add these imports (alongside the existing ones):

```python
from numpy.lib.scimath import sqrt as _csqrt
from waveguides.core import (
    alpha_rec_te, alpha_rec_tm, alpha_cir_te, alpha_cir_tm, C_LIGHT,
)
```

After the `_rec` / `_cir` factories and `REC_FS` / `CIR_FS` constants, add:

```python
def scalar_pf(wg, f, lossless=False):
    """Independent scalar reference for the propagation factor (per mode)."""
    r_s = 0.0 if lossless else np.sqrt(np.pi * f * 4 * np.pi * 1e-7 / wg.sigma)
    k = 2 * np.pi * f / C_LIGHT * np.sqrt(wg.er)
    out = np.zeros(len(wg.mode_info_list), dtype=complex)
    for i, m in enumerate(wg.mode_info_list):
        beta = _csqrt(k**2 - m.kc**2)
        if wg.cross_tag == "rec":
            if m.mode_type > 0:
                alpha = alpha_rec_te(wg.a, wg.b, r_s, m.mode_num1, m.mode_num2, k, m.kc)
            else:
                alpha = alpha_rec_tm(wg.a, wg.b, r_s, m.mode_num1, m.mode_num2, k, m.kc)
        else:
            if m.mode_type > 0:
                alpha = alpha_cir_te(wg.r, r_s, m.mode_num1, k, m.kc)
            else:
                alpha = alpha_cir_tm(wg.r, r_s, k, m.kc)
        out[i] = np.exp(-(np.imag(beta) + np.abs(alpha) + 1j * np.real(beta)) * wg.l)
    return out


def scalar_impedance(wg, f):
    """Independent scalar reference for the wave impedance (per mode)."""
    k = 2 * np.pi * f / C_LIGHT * np.sqrt(wg.er)
    eta = 120 * np.pi / np.sqrt(wg.er)
    out = np.zeros(len(wg.mode_info_list), dtype=complex)
    for i, m in enumerate(wg.mode_info_list):
        beta = np.sqrt(complex(k**2 - m.kc**2))
        out[i] = (k / beta * eta) if m.mode_type > 0 else (beta / k * eta)
    return out


def scalar_wavelength(wg, f):
    """Independent scalar reference for the guide wavelength (per mode)."""
    k0 = f / C_LIGHT * 2 * np.pi * np.sqrt(wg.er)
    out = np.zeros(len(wg.mode_info_list), dtype=complex)
    for i, m in enumerate(wg.mode_info_list):
        out[i] = 2 * np.pi / np.sqrt(complex(k0**2 - m.kc**2))
    return out
```

- [ ] **Step 2: Repoint the existing equality tests to the scalar oracle**

In `tests/test_vectorized_eval.py`, replace the four oracle expressions:

- In `test_propagation_factor_matches_oracle`:
  `ref = np.array([wg.propagation_factor_at(f) for f in fs])`
  → `ref = np.array([scalar_pf(wg, f) for f in fs])`
- In `test_propagation_factor_scalar_returns_2d`:
  `ref = wg.propagation_factor_at(10e9)`
  → `ref = scalar_pf(wg, 10e9)`
- In `test_impedance_matches_oracle`:
  `ref = np.array([wg.impedance_at(f) for f in fs])`
  → `ref = np.array([scalar_impedance(wg, f) for f in fs])`
- In `test_phaseshift_matches_oracle`:
  `ref = wg.phaseshift_at_list(fs)`
  → `ref = np.angle(np.array([scalar_pf(wg, f) for f in fs]))`

- [ ] **Step 3: Run the full suite to confirm still-green**

Run: `python -m pytest tests/test_vectorized_eval.py -q`
Expected: PASS (25 tests). The oracle is now an independent scalar re-derivation; values are unchanged, so every test still passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_vectorized_eval.py
git commit -m "test: add independent scalar oracle, repoint eval tests to it"
```

---

### Task 2: Vectorized kernel in core.py

Add the three kernel functions + the shared grid helper. Nothing calls them yet; they are tested directly against the scalar oracle from Task 1.

**Files:**
- Modify: `src/waveguides/core.py` (add a kernel section just before `class WG`, i.e. after `calc_propagation_factor_cir`, ~line 242)
- Modify: `tests/test_vectorized_eval.py` (add kernel tests)

**Interfaces:**
- Consumes: `WG._mode_arrays()` (existing), `C_LIGHT`, `wg.{cross_tag,a,b,r,er,sigma,l}`.
- Produces: `_freq_mode_grid(wg, fs) -> (fs_1d, k, kc, beta, mode_type, m1, m2)` with `k=(M,1)`, `kc=(1,N)`, `beta=(M,N)` complex128, `mode_type/m1/m2=(1,N)`; `propagation_factor_matrix(wg, fs, lossless=False) -> (M,N)`; `impedance_matrix(wg, fs) -> (M,N)`; `wavelength_matrix(wg, fs) -> (M,N)`.

- [ ] **Step 1: Write failing kernel tests**

First, extend the existing top-of-file `from waveguides.core import (...)` block (added in Task 1) to also import the kernels — keep it a single import at the top of the file to avoid ruff `E402`:

```python
from waveguides.core import (
    alpha_rec_te, alpha_rec_tm, alpha_cir_te, alpha_cir_tm, C_LIGHT,
    propagation_factor_matrix, impedance_matrix, wavelength_matrix,
)
```

Then append the test functions to the end of `tests/test_vectorized_eval.py`:

```python
@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
@pytest.mark.parametrize("N", [1, 40, 800])
def test_kernel_pf_matches_scalar(make_wg, fs, N):
    wg = make_wg(N)
    got = propagation_factor_matrix(wg, fs)
    ref = np.array([scalar_pf(wg, f) for f in fs])
    assert got.shape == (len(fs), N)
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
@pytest.mark.parametrize("N", [1, 40, 800])
def test_kernel_pf_lossless_matches_scalar(make_wg, fs, N):
    wg = make_wg(N)
    got = propagation_factor_matrix(wg, fs, lossless=True)
    ref = np.array([scalar_pf(wg, f, lossless=True) for f in fs])
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
@pytest.mark.parametrize("N", [1, 40, 800])
def test_kernel_impedance_matches_scalar(make_wg, fs, N):
    wg = make_wg(N)
    got = impedance_matrix(wg, fs)
    ref = np.array([scalar_impedance(wg, f) for f in fs])
    assert got.shape == (len(fs), N)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
@pytest.mark.parametrize("N", [1, 40, 800])
def test_kernel_wavelength_matches_scalar(make_wg, fs, N):
    wg = make_wg(N)
    got = wavelength_matrix(wg, fs)
    ref = np.array([scalar_wavelength(wg, f) for f in fs])
    assert got.shape == (len(fs), N)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_kernel_scalar_and_empty_shapes():
    wg = _rec(10)
    assert propagation_factor_matrix(wg, 10e9).shape == (1, 10)
    assert impedance_matrix(wg, 10e9).shape == (1, 10)
    assert wavelength_matrix(wg, 10e9).shape == (1, 10)
    assert propagation_factor_matrix(wg, []).shape == (0, 10)


def test_kernel_pf_exact_cutoff():
    wg = _rec(40)
    i = 5
    fc = wg.mode_info_list[i].fc
    lossy = propagation_factor_matrix(wg, fc)[0]
    assert np.isfinite(lossy).all()
    assert lossy[i] == 0                     # lossy exact cutoff -> 0 (guarded)
    lossless = propagation_factor_matrix(wg, fc, lossless=True)[0]
    assert np.isnan(lossless[i])             # lossless exact cutoff -> nan (matches scalar)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_vectorized_eval.py -k kernel -v`
Expected: FAIL at import (`ImportError: cannot import name 'propagation_factor_matrix' from 'waveguides.core'`).

- [ ] **Step 3: Add the kernel to core.py**

In `src/waveguides/core.py`, add a new section immediately before `class WG` (after `calc_propagation_factor_cir`):

```python
# *********************************************************
# %% Vectorized evaluation kernel
# *********************************************************

def _freq_mode_grid(wg, fs):
    """Broadcast frequency/mode grid shared by the vectorized kernels.

    Returns (fs_1d, k, kc, beta, mode_type, m1, m2) with fs_1d (M,),
    k (M,1), kc (1,N), beta (M,N) complex128, mode_type/m1/m2 (1,N).
    """
    ma = wg._mode_arrays()
    fs = np.atleast_1d(np.asarray(fs, dtype=float))
    k = (2 * np.pi * fs / C_LIGHT * np.sqrt(wg.er))[:, None]
    kc = ma["kc"][None, :]
    beta = np.sqrt((k**2 - kc**2).astype(np.complex128))
    mode_type = ma["mode_type"][None, :]
    m1 = ma["mode_num1"][None, :].astype(float)
    m2 = ma["mode_num2"][None, :].astype(float)
    return fs, k, kc, beta, mode_type, m1, m2


def propagation_factor_matrix(wg, fs, lossless=False):
    """Complex propagation factor exp(-(alpha + j*beta)*l) for each mode of
    *wg* over frequencies *fs* (scalar or 1-D). Shape (M, N), complex128.
    When *lossless*, wall loss is dropped (r_s = 0)."""
    fs, k, kc, beta, mode_type, m1, m2 = _freq_mode_grid(wg, fs)
    r_s = 0.0 if lossless else np.sqrt(np.pi * fs * 4 * np.pi * 1e-7 / wg.sigma)[:, None]
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
    pf = np.exp(-(np.imag(beta) + np.abs(alpha) + 1j * np.real(beta)) * wg.l)
    if not lossless:
        # Exact cutoff (kc == k): alpha -> inf, so the physical limit is pf = 0;
        # the complex-zero division above yields nan there — pin it to the limit.
        # (Lossless keeps the scalar path's nan at that measure-zero point.)
        pf = np.where(ratio2 == 1.0, 0.0, pf)
    return pf


def impedance_matrix(wg, fs):
    """Wave-impedance matrix for *wg* over frequencies *fs* (scalar or 1-D).
    Shape (M, N), complex128."""
    fs, k, kc, beta, mode_type, m1, m2 = _freq_mode_grid(wg, fs)
    eta = 120 * np.pi / np.sqrt(wg.er)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mode_type > 0, k / beta * eta, beta / k * eta)


def wavelength_matrix(wg, fs):
    """Guide-wavelength matrix (2*pi / beta) for *wg* over frequencies *fs*
    (scalar or 1-D). Shape (M, N), complex128."""
    fs, k, kc, beta, mode_type, m1, m2 = _freq_mode_grid(wg, fs)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2 * np.pi / beta
```

- [ ] **Step 4: Run kernel tests to verify they pass**

Run: `python -m pytest tests/test_vectorized_eval.py -k kernel -v`
Expected: PASS (all kernel tests).

- [ ] **Step 5: Run the full suite and ruff**

Run: `python -m pytest tests/test_vectorized_eval.py -q` (expect all pass) and `ruff check src/waveguides/core.py` (expect no new errors).

- [ ] **Step 6: Commit**

```bash
git add src/waveguides/core.py tests/test_vectorized_eval.py
git commit -m "feat(core): add vectorized propagation_factor/impedance/wavelength kernels"
```

---

### Task 3: Rewire WG methods; remove scalar assemblers

Point the eight `WG` `*_at`/`*_at_list` methods at the kernel and delete the now-unused `calc_propagation_factor_rec/cir`. `alpha_*` stay (physics atoms + oracle).

**Files:**
- Modify: `src/waveguides/core.py` (rewrite methods `wavelength_at`..`phaseshift_at_list`, ~lines 354-423; delete `calc_propagation_factor_rec` ~111-124 and `calc_propagation_factor_cir` ~228-241)
- Modify: `tests/test_vectorized_eval.py` (add WG-method tests)

**Interfaces:**
- Consumes: `propagation_factor_matrix`, `impedance_matrix`, `wavelength_matrix` (Task 2).
- Produces: `WG.{wavelength,impedance,propagation_factor,phaseshift}_at(...)` returning `(N,)` and `..._at_list(...)` returning `(M,N)`; `propagation_factor_at(f, lossless=False)`.

- [ ] **Step 1: Write failing WG-method tests**

Append to `tests/test_vectorized_eval.py`:

```python
@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
@pytest.mark.parametrize("N", [1, 40, 800])
def test_wg_methods_match_scalar(make_wg, fs, N):
    wg = make_wg(N)
    f0 = fs[0]
    # scalar *_at -> (N,)
    np.testing.assert_allclose(wg.impedance_at(f0), scalar_impedance(wg, f0),
                               rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(wg.wavelength_at(f0), scalar_wavelength(wg, f0),
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(wg.propagation_factor_at(f0), scalar_pf(wg, f0),
                               rtol=1e-9, atol=1e-12)
    assert wg.impedance_at(f0).shape == (N,)
    # list *_at_list -> (M,N)
    imp = wg.impedance_at_list(fs)
    assert imp.shape == (len(fs), N)
    np.testing.assert_allclose(imp, np.array([scalar_impedance(wg, f) for f in fs]),
                               rtol=1e-9, atol=1e-6)
    wl = wg.wavelength_at_list(fs)
    np.testing.assert_allclose(wl, np.array([scalar_wavelength(wg, f) for f in fs]),
                               rtol=1e-9, atol=1e-9)
    pf = wg.propagation_factor_at_list(fs)
    np.testing.assert_allclose(pf, np.array([scalar_pf(wg, f) for f in fs]),
                               rtol=1e-9, atol=1e-12)
    ps = wg.phaseshift_at_list(fs)
    np.testing.assert_allclose(ps, np.angle(np.array([scalar_pf(wg, f) for f in fs])),
                               rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
def test_wg_propagation_factor_lossless(make_wg, fs):
    wg = make_wg(40)
    got = wg.propagation_factor_at_list(fs, lossless=True)
    ref = np.array([scalar_pf(wg, f, lossless=True) for f in fs])
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)
    assert wg.propagation_factor_at(fs[0], lossless=True).shape == (40,)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_vectorized_eval.py -k "wg_methods or wg_propagation_factor_lossless" -v`
Expected: FAIL — `wavelength_at_list`/`impedance_at_list` currently return Python-loop results that still pass equality, but `test_wg_methods_match_scalar` also asserts `wg.impedance_at(f0).shape == (N,)` which holds; the genuinely failing assertion is the `(M,N)` shape/values are fine too... Run it: the new lossless-list test passes on the old code as well. **If all pass on the unmodified code, that is acceptable** — these tests pin behavior the rewrite must preserve; proceed to Step 3 and confirm they still pass after the rewrite. (The rewrite is a refactor; its safety net is that these tests keep passing.)

- [ ] **Step 3: Rewire the WG methods**

In `src/waveguides/core.py`, replace the bodies of the eight methods (`wavelength_at` through `phaseshift_at_list`, ~lines 354-423) with:

```python
    def wavelength_at(self, f):
        return wavelength_matrix(self, f)[0]

    def wavelength_at_list(self, f_list):
        return wavelength_matrix(self, f_list)

    def impedance_at(self, f):
        return impedance_matrix(self, f)[0]

    def impedance_at_list(self, f_list):
        return impedance_matrix(self, f_list)

    def gamma_at(self, f):
        raise NotImplementedError()

    def gamma_at_list(self, f_list):
        raise NotImplementedError()

    def propagation_factor_at(self, f, lossless=False):
        return propagation_factor_matrix(self, f, lossless)[0]

    def propagation_factor_at_list(self, f_list, lossless=False):
        return propagation_factor_matrix(self, f_list, lossless)

    def phaseshift_at(self, f):
        """Return the phase shift (rad) for each mode at frequency f.

        Defined as the argument of the propagation factor:
        angle(exp(-(alpha + j*beta) * l)) = -beta * l (rad).
        """
        return np.angle(propagation_factor_matrix(self, f))[0]

    def phaseshift_at_list(self, f_list):
        """Return the phase shift (rad) for each mode over a list of frequencies.

        Shape: (len(f_list), N).
        """
        return np.angle(propagation_factor_matrix(self, f_list))
```

- [ ] **Step 4: Delete the scalar assemblers**

In `src/waveguides/core.py`, delete `calc_propagation_factor_rec` (the whole `def`, ~lines 111-124 in the rectangular section) and `calc_propagation_factor_cir` (~lines 228-241 in the circular section). Do NOT delete `alpha_rec_te/tm`, `alpha_cir_te/tm`, or any `norm_*` / mode-construction function.

- [ ] **Step 5: Run tests + ruff**

Run: `python -m pytest tests/test_vectorized_eval.py -q` (expect all pass) and `ruff check src/waveguides/core.py` (expect clean — no `F821` from a lingering reference to the deleted functions).

- [ ] **Step 6: Commit**

```bash
git add src/waveguides/core.py tests/test_vectorized_eval.py
git commit -m "refactor(core): delegate WG.*_at/*_at_list to kernel; drop scalar assemblers"
```

---

### Task 4: Rewire heavy_computation to the kernel

Make the three `heavy_computation` functions thin delegates to the core kernel, add `lossless` to `propagation_factor_array`, delete the local `_grid`, and fix imports.

**Files:**
- Modify: `src/waveguides/heavy_computation.py` (rewrite all three functions; delete `_grid`; imports)
- Modify: `tests/test_vectorized_eval.py` (add a lossless test for `propagation_factor_array`)

**Interfaces:**
- Consumes: `waveguides.core.{propagation_factor_matrix, impedance_matrix}` (Task 2).
- Produces: `propagation_factor_array(wg, fs, lossless=False) -> (M,N)`; `impedance_array(wg, fs) -> (M,N)`; `phaseshift_array(wg, fs) -> (M,N)`.

- [ ] **Step 1: Write the failing lossless test**

Append to `tests/test_vectorized_eval.py`:

```python
@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
def test_hc_propagation_factor_lossless(make_wg, fs):
    # propagation_factor_array is already imported at the top of the file
    wg = make_wg(40)
    got = propagation_factor_array(wg, fs, lossless=True)
    ref = np.array([scalar_pf(wg, f, lossless=True) for f in fs])
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_vectorized_eval.py::test_hc_propagation_factor_lossless -v`
Expected: FAIL — `TypeError: propagation_factor_array() got an unexpected keyword argument 'lossless'`.

- [ ] **Step 3: Rewrite heavy_computation.py**

Replace the entire body of `src/waveguides/heavy_computation.py` below the module docstring with:

```python
from __future__ import annotations

import numpy as np

from .core import WG, impedance_matrix, propagation_factor_matrix


def propagation_factor_array(wg: WG, fs, lossless: bool = False) -> np.ndarray:
    """Complex propagation factor exp(-(alpha + j*beta)*l) for each mode of
    *wg* over frequencies *fs* (scalar or 1-D). Shape (len(fs), N), complex128.
    When *lossless*, wall loss is dropped."""
    return propagation_factor_matrix(wg, fs, lossless)


def phaseshift_array(wg: WG, fs) -> np.ndarray:
    """Phase shift (rad) for each mode of *wg* over frequencies *fs*
    (scalar or 1-D). angle(propagation_factor), shape (len(fs), N)."""
    return np.angle(propagation_factor_matrix(wg, fs))


def impedance_array(wg: WG, fs) -> np.ndarray:
    """Wave-impedance matrix for *wg* over frequencies *fs* (scalar or 1-D).
    Shape (len(fs), N), complex128."""
    return impedance_matrix(wg, fs)
```

This deletes `_grid` and the old inlined formulas, and drops the now-unused `C_LIGHT` import.

- [ ] **Step 4: Run the lossless test, full suite, and ruff**

Run: `python -m pytest tests/test_vectorized_eval.py::test_hc_propagation_factor_lossless -v` (expect PASS), then `python -m pytest -q` (expect all pass), then `ruff check src/waveguides/heavy_computation.py` (expect clean — no `F401`).

- [ ] **Step 5: Commit**

```bash
git add src/waveguides/heavy_computation.py tests/test_vectorized_eval.py
git commit -m "refactor(heavy_computation): delegate to core kernel; add lossless"
```

---

## Notes / deliberate choices

- **Oracle independence:** the test oracle (`scalar_pf`/`scalar_impedance`/`scalar_wavelength`) is a self-contained scalar re-derivation living in the test file, built on the retained `alpha_*` atoms. After consolidation both entry points share the kernel, so the scalar loop is what keeps the tests non-tautological.
- **`alpha_*` retained though production-unused:** they are small pure physics primitives (and the oracle's atoms). Not lint errors; not removed.
- **`propagation_factor_at` param name:** the kernel is called positionally as `propagation_factor_matrix(self, f, lossless)` so the method keeps its public `lossless=` keyword.

## Self-Review

- **Spec coverage:** §3.1 kernel → Task 2. §3.2 WG delegation + shapes → Task 3. §3.3 heavy_computation delegation → Task 4. §4 lossless on both sides → Task 3 (WG kept) + Task 4 (hc added) + kernel signature (Task 2). §5 remove `calc_propagation_factor_*` → Task 3; remove `_grid` → Task 4; keep `alpha_*`/`norm_*` → honored (only `calc_*` deleted). §6 edge cases → `test_kernel_pf_exact_cutoff` (Task 2) + guard code; evanescent via complex sqrt in `_freq_mode_grid`. §7 independent scalar oracle + coverage (rec+cir, N∈{1,40,800}, below/near/above cutoff, both entry points, lossless, wavelength, empty/scalar shapes) → Tasks 1-4. §8 YAGNI (no mode-construction/gamma/field changes) → respected.
- **Placeholder scan:** none; every code/command step is concrete. Task 3 Step 2 explicitly notes the refactor tests may pass pre-change (safety-net refactor) — intentional, not a placeholder.
- **Type consistency:** `_freq_mode_grid` returns the same 7-tuple consumed in all three `*_matrix` functions; `propagation_factor_matrix(wg, fs, lossless=False)` is called with the same name/arg order by `WG.propagation_factor_at` (Task 3) and `heavy_computation.propagation_factor_array` (Task 4); oracle names `scalar_pf/scalar_impedance/scalar_wavelength` are defined in Task 1 and used verbatim in Tasks 2-4.
