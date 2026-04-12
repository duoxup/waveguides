#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 17:01:48 2025

@author: duoxup
"""

import dataclasses
import warnings
import numpy as np
from importlib import resources


def load_memmap_from_package(filename: str) -> np.ndarray:
    """
    Load a .npy file shipped inside the 'waveguides' package.
    Returns a numpy memmap-backed array (read-only).
    """
    ref = resources.files("waveguides").joinpath(filename)
    with resources.as_file(ref) as p:
        return np.load(p, mmap_mode="r")


@dataclasses.dataclass
class EField2D:
    """2D electric field distribution at grid points.

    Attributes
    ----------
    X, Y : np.ndarray
        Grid point coordinates (metres).
    Ex, Ey, Ez : np.ndarray
        Cartesian electric field components at each grid point.
    """
    X: np.ndarray
    Y: np.ndarray
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray


class ModeInfo:
    def __init__(self, kc=0, mode_type=0, mode_num1=0, mode_num2=0, fc=0,
                 plus_dir=1, polar_dir=1, norm_constant=1):
        """
        kc           : cutoff wavenumber
        mode_type    : 1 = TE, 0 = TM
        mode_num1    : first mode index  (m for rectangular, q for circular)
        mode_num2    : second mode index (n for rectangular, r for circular)
        fc           : cutoff frequency
        plus_dir     : ±1 sign factor (see Wei Zhao's Thesis, p13-17)
        polar_dir    : polarisation direction for circular waveguides
                       (1 = cos, 0 = sin, -1 = unused)
        norm_constant: field normalisation constant
        """
        self.kc = kc
        self.mode_type = mode_type
        self.mode_num1 = mode_num1
        self.mode_num2 = mode_num2
        self.fc = fc
        self.plus_dir = plus_dir
        self.polar_dir = polar_dir
        self.norm_constant = norm_constant

    def __repr__(self):
        fields = ', '.join(f'{k}={v!r}' for k, v in vars(self).items())
        return f'ModeInfo({fields})'

    def to_array(self):
        return [self.kc, self.mode_type, self.mode_num1, self.mode_num2,
                self.fc, self.plus_dir, self.polar_dir, self.norm_constant]

    def toArray(self):
        """Deprecated: use ``to_array()`` instead."""
        warnings.warn(
            "ModeInfo.toArray() is deprecated; use to_array() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_array()
