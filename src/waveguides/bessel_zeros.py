#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:11:14 2026

@author: duoxup
"""



from __future__ import annotations

import numpy as np

from .utils import load_memmap_from_package


# Lazy-loaded globals shared across imports in the same process
_A: np.ndarray | None = None
_B: np.ndarray | None = None


def get_tables() -> tuple[np.ndarray, np.ndarray]:
    global _A, _B
    if _A is None:
        _A = load_memmap_from_package("A.npy")
    if _B is None:
        _B = load_memmap_from_package("B.npy")
    return _A, _B