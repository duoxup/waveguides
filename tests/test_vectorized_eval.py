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
