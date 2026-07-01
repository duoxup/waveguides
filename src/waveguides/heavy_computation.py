#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 19:55:14 2026

@author: duoxup
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.lib.scimath import sqrt

from .core import (WG, C_LIGHT,
                   alpha_cir_te, alpha_cir_tm,
                   alpha_rec_te, alpha_rec_tm)

# *********************************************************
# %% Internal helpers
# *********************************************************

def _build_pf_args(wg: WG, fs: Sequence[float]):
    mi = wg.mode_info_list
    match wg.cross_tag.lower():
        case 'rec':
            out = np.fromfunction(
                np.vectorize(lambda i, j: (int(i), int(j), fs[int(i)],
                                           wg.sigma, wg.a, wg.b, wg.l, wg.er,
                                           mi[int(j)].mode_type, mi[int(j)].mode_num1,
                                           mi[int(j)].mode_num2, mi[int(j)].kc),
                             otypes=[object]),
                (len(fs), len(mi)),
                dtype=int,
            )
        case 'cir':
            out = np.fromfunction(
                np.vectorize(lambda i, j: (int(i), int(j), fs[int(i)],
                                           wg.sigma, wg.r, wg.l, wg.er,
                                           mi[int(j)].mode_type, mi[int(j)].mode_num1,
                                           mi[int(j)].mode_num2, mi[int(j)].kc),
                             otypes=[object]),
                (len(fs), len(mi)),
                dtype=int,
            )
        case _:
            raise ValueError(f'Unknown waveguide cross_tag: {wg.cross_tag!r}')
    return out


def _build_imp_args(wg: WG, fs: Sequence[float]):
    mi = wg.mode_info_list
    out = np.fromfunction(
        np.vectorize(lambda i, j: (int(i), int(j), fs[int(i)], wg.er,
                                   mi[int(j)].mode_type, mi[int(j)].kc),
                     otypes=[object]),
        (len(fs), len(mi)),
        dtype=int,
    )
    return out


ComplexLike = Union[complex, np.complexfloating]


def _results_to_matrix_auto_shape(
    results: Iterable[Tuple[int, int, ComplexLike]],
    *,
    fill_value: ComplexLike = np.nan + 1j * np.nan,
    dtype: np.dtype = np.complex128,
    check_duplicates: bool = False,
    allow_negative_index: bool = False,
) -> np.ndarray:
    """
    Convert an unordered iterable of (i, j, x) into a 2D complex matrix X,
    where X[i, j] = x, and the output shape is inferred automatically as
    (max_i + 1, max_j + 1).

    Parameters
    ----------
    results:
        Iterable of (i, j, x). Typically from pool.imap_unordered.
    fill_value:
        Value used to initialise missing entries.
    dtype:
        Output dtype for X.
    check_duplicates:
        If True, raise ValueError when duplicate (i, j) pairs exist.
    allow_negative_index:
        If False (default), negative i/j raise ValueError.

    Returns
    -------
    X : np.ndarray, shape (max_i+1, max_j+1)
    """
    results = list(results)
    if len(results) == 0:
        raise ValueError("`results` is empty; cannot infer matrix shape.")

    res = np.asarray(results, dtype=object)
    if res.ndim != 2 or res.shape[1] != 3:
        raise ValueError("`results` must be an iterable of 3-tuples: (i, j, x).")

    ii = res[:, 0].astype(np.int64, copy=False)
    jj = res[:, 1].astype(np.int64, copy=False)

    if not allow_negative_index:
        if (ii < 0).any() or (jj < 0).any():
            bad = np.where((ii < 0) | (jj < 0))[0][0]
            raise ValueError(f"Negative index found at results[{bad}] = {results[bad]}")

    nrows = int(ii.max()) + 1
    ncols = int(jj.max()) + 1
    if nrows <= 0 or ncols <= 0:
        raise ValueError(f"Inferred invalid shape: ({nrows}, {ncols}).")

    if check_duplicates:
        lin = ii * ncols + jj
        if np.unique(lin).size != lin.size:
            raise ValueError("Duplicate (i, j) pairs found in `results`.")

    xx = np.asarray(res[:, 2], dtype=dtype)
    X = np.empty((nrows, ncols), dtype=dtype)
    X[...] = fill_value
    X[ii, jj] = xx
    return X


def _pf_worker_cir(args):
    i, j, f, sigma, r, l, er, mode_type, mode_num1, mode_num2, kc = args
    k = 2 * np.pi * f / C_LIGHT * np.sqrt(er)
    r_s = np.sqrt(np.pi * f * 4 * np.pi * 1e-7 / sigma)
    beta = sqrt(k**2 - kc**2)
    if mode_type > 0:  # TE mode
        alpha = alpha_cir_te(r, r_s, mode_num1, k, kc)
    else:  # TM mode
        alpha = alpha_cir_tm(r, r_s, k, kc)
    return i, j, np.exp(-(np.imag(beta) + np.abs(alpha) + 1j * np.real(beta)) * l)


def _pf_worker_rec(args):
    i, j, f, sigma, a, b, l, er, mode_type, mode_num1, mode_num2, kc = args
    k = 2 * np.pi * f / C_LIGHT * np.sqrt(er)
    r_s = np.sqrt(np.pi * f * 4 * np.pi * 1e-7 / sigma)
    beta = sqrt(k**2 - kc**2)
    if mode_type > 0:  # TE mode
        alpha = alpha_rec_te(a, b, r_s, mode_num1, mode_num2, k, kc)
    else:  # TM mode
        alpha = alpha_rec_tm(a, b, r_s, mode_num1, mode_num2, k, kc)
    return i, j, np.exp(-(np.imag(beta) + np.abs(alpha) + 1j * np.real(beta)) * l)


def _imp_worker(args):
    i, j, f, er, mode_type, kc = args
    k = 2 * np.pi * f / C_LIGHT * np.sqrt(er)
    beta = sqrt(k**2 - kc**2)
    if mode_type > 0:  # TE mode
        Z = k / beta * 120 * np.pi / np.sqrt(er)
    else:  # TM mode
        Z = beta / k * 120 * np.pi / np.sqrt(er)
    return i, j, Z


def _dispatch(func_worker,
              iterable: Iterable,
              *,
              pool=None,
              chunksize: Optional[int] = 1):
    res = []
    if pool:
        for item in pool.imap_unordered(func_worker, iterable, chunksize=chunksize):
            res.append(item)
    else:
        for args in iterable:
            res.append(func_worker(args))
    return res


def _select_pf_worker(wg: WG):
    match wg.cross_tag.lower():
        case 'rec':
            return _pf_worker_rec
        case 'cir':
            return _pf_worker_cir
        case _:
            raise ValueError(f'Unknown waveguide cross_tag: {wg.cross_tag!r}')


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
    return np.exp(-(np.imag(beta) + np.abs(alpha) + 1j * np.real(beta)) * wg.l)


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


# *********************************************************
# %% Deprecated aliases — will be removed in a future version
# *********************************************************


def impedance_array_multifreq(wg: WG, fs: Sequence[float], *,
                              pool=None, chunksize: int = 64) -> np.ndarray:
    """Deprecated: use ``impedance_array()`` instead."""
    warnings.warn(
        "impedance_array_multifreq() is deprecated; use impedance_array() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return impedance_array(wg, fs, pool=pool, chunksize=chunksize)
