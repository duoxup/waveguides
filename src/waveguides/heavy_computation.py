#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 19:55:14 2026

@author: duoxup
"""

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
