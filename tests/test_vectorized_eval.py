import numpy as np
import pytest
from numpy.lib.scimath import sqrt as _csqrt

from waveguides import RecWG, CirWG
from waveguides.heavy_computation import (
    propagation_factor_array,
    impedance_array,
    phaseshift_array,
)
from waveguides.core import (
    alpha_rec_te, alpha_rec_tm, alpha_cir_te, alpha_cir_tm, C_LIGHT,
    propagation_factor_matrix, impedance_matrix, wavelength_matrix,
)


def _rec(N):
    return RecWG(a=0.02286, b=0.01016, l=0.1, N=N, er=1.0, sigma=5.8e7)


def _cir(N):
    return CirWG(r=0.01, l=0.1, N=N, er=1.0, sigma=5.8e7)


REC_FS = np.array([7e9, 10e9, 15e9])
CIR_FS = np.array([9e9, 12e9, 20e9])


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


@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
@pytest.mark.parametrize("N", [1, 40, 800])
def test_propagation_factor_matches_oracle(make_wg, fs, N):
    wg = make_wg(N)
    got = propagation_factor_array(wg, fs)
    ref = np.array([scalar_pf(wg, f) for f in fs])
    assert got.shape == (len(fs), N)
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_propagation_factor_scalar_returns_2d():
    wg = _rec(10)
    out = propagation_factor_array(wg, 10e9)
    assert out.shape == (1, 10)
    ref = scalar_pf(wg, 10e9)
    np.testing.assert_allclose(out[0], ref, rtol=1e-9, atol=1e-12)


def test_propagation_factor_empty_returns_0xN():
    wg = _rec(10)
    out = propagation_factor_array(wg, [])
    assert out.shape == (0, 10)


def test_propagation_factor_no_pool_kwarg():
    wg = _rec(4)
    with pytest.raises(TypeError):
        propagation_factor_array(wg, [10e9], pool=None)


@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
@pytest.mark.parametrize("N", [1, 40, 800])
def test_impedance_matches_oracle(make_wg, fs, N):
    wg = make_wg(N)
    got = impedance_array(wg, fs)
    ref = np.array([scalar_impedance(wg, f) for f in fs])
    assert got.shape == (len(fs), N)
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-6)


def test_impedance_no_pool_kwarg():
    wg = _rec(4)
    with pytest.raises(TypeError):
        impedance_array(wg, [10e9], pool=None)


@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
@pytest.mark.parametrize("N", [1, 40, 800])
def test_phaseshift_matches_oracle(make_wg, fs, N):
    wg = make_wg(N)
    got = phaseshift_array(wg, fs)
    ref = np.angle(np.array([scalar_pf(wg, f) for f in fs]))
    assert got.shape == (len(fs), N)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_phaseshift_no_pool_kwarg():
    wg = _rec(4)
    with pytest.raises(TypeError):
        phaseshift_array(wg, [10e9], pool=None)


def test_propagation_factor_at_exact_cutoff_is_zero():
    wg = _rec(40)
    i = 5
    fc = wg.mode_info_list[i].fc          # exact cutoff frequency of mode i
    row = propagation_factor_array(wg, fc)[0]
    assert np.isfinite(row).all()          # no nan/inf anywhere in the row
    assert row[i] == 0                     # physical limit exp(-inf) = 0 at cutoff


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
    assert impedance_matrix(wg, []).shape == (0, 10)
    assert wavelength_matrix(wg, []).shape == (0, 10)


def test_kernel_pf_exact_cutoff():
    wg = _rec(40)
    i = 5
    fc = wg.mode_info_list[i].fc
    lossy = propagation_factor_matrix(wg, fc)[0]
    assert np.isfinite(lossy).all()
    assert lossy[i] == 0                     # lossy exact cutoff -> 0 (guarded)
    lossless = propagation_factor_matrix(wg, fc, lossless=True)[0]
    assert np.isnan(lossless[i])             # lossless exact cutoff -> nan (matches scalar)


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
@pytest.mark.parametrize("N", [1, 40, 800])
def test_wg_propagation_factor_lossless(make_wg, fs, N):
    wg = make_wg(N)
    got = wg.propagation_factor_at_list(fs, lossless=True)
    ref = np.array([scalar_pf(wg, f, lossless=True) for f in fs])
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)
    assert wg.propagation_factor_at(fs[0], lossless=True).shape == (N,)


@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
@pytest.mark.parametrize("N", [1, 40, 800])
def test_hc_propagation_factor_lossless(make_wg, fs, N):
    # propagation_factor_array is already imported at the top of the file
    wg = make_wg(N)
    got = propagation_factor_array(wg, fs, lossless=True)
    ref = np.array([scalar_pf(wg, f, lossless=True) for f in fs])
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("make_wg, fs", [(_rec, REC_FS), (_cir, CIR_FS)])
def test_wg_phaseshift_at_scalar(make_wg, fs):
    wg = make_wg(40)
    f0 = fs[0]
    got = wg.phaseshift_at(f0)
    assert got.shape == (40,)
    np.testing.assert_allclose(got, np.angle(scalar_pf(wg, f0)), rtol=1e-9, atol=1e-9)
