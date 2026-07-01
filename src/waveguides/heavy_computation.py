#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 19:55:14 2026

@author: duoxup
"""

from __future__ import annotations

import numpy as np

from .core import WG, C_LIGHT

# *********************************************************
# %% Internal helpers
# *********************************************************

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


# *********************************************************
# %% Public APIs
# *********************************************************

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
    pf = np.exp(-(np.imag(beta) + np.abs(alpha) + 1j * np.real(beta)) * wg.l)
    # Exact cutoff (kc == k) makes alpha diverge, so the physical limit is pf = 0;
    # the complex-zero division above yields nan there instead — pin it to the limit.
    return np.where(ratio2 == 1.0, 0.0, pf)


def phaseshift_array(wg: WG, fs) -> np.ndarray:
    """Phase shift (rad) for each mode of *wg* over frequencies *fs*.

    Defined as angle(propagation_factor) = -beta*l wrapped to (-pi, pi].
    *fs* may be a scalar or 1-D array-like. Returns shape (len(fs), N).
    """
    return np.angle(propagation_factor_array(wg, fs))


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
