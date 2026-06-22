#!/usr/bin/env python3
"""Regenerate the Bessel-zero lookup tables at full double precision.

``A.npy`` and ``B.npy`` ship inside the package and back
``bessel_zeros.get_tables()``.  The original tables held only ~4 significant
digits (errors up to ~5e-4), which turned the removable cutoff-coincidence
singularity in pwmma's circular<->circular coupling matrix into a visible
spurious pole (see pwmma ``local/PATCH-cm_cc-removable-singularity.md``).  This
script rebuilds them from SciPy so every entry is accurate to ~1e-15.

Conventions (must match the loader / consumers):

* ``A[q, r]`` = (r+1)-th positive zero of ``J_q``   -> TM cutoffs
  (``scipy.special.jn_zeros``).
* ``B[q, r]`` = (r+1)-th positive zero of ``J'_q``  -> TE cutoffs
  (``scipy.special.jnp_zeros``).  For ``q == 0`` this excludes the trivial
  ``x = 0``, matching the original table (``B[0, 0] == 3.8317...``).

Run:  ``python tools/generate_bessel_zeros.py``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import jn_zeros, jnp_zeros

Q = R = 1000  # table dimensions (orders q in [0, Q), zeros r in [0, R))

DEST = Path(__file__).resolve().parent.parent / "src" / "waveguides"


def build() -> tuple[np.ndarray, np.ndarray]:
    A = np.array([jn_zeros(q, R) for q in range(Q)], dtype=np.float64)
    B = np.array([jnp_zeros(q, R) for q in range(Q)], dtype=np.float64)
    assert A.shape == (Q, R) and B.shape == (Q, R)
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(B))
    # zeros are strictly increasing along each order -> sanity against bad rows
    assert np.all(np.diff(A, axis=1) > 0) and np.all(np.diff(B, axis=1) > 0)
    return A, B


def main() -> None:
    A, B = build()
    np.save(DEST / "A.npy", A)
    np.save(DEST / "B.npy", B)
    print(f"wrote {DEST/'A.npy'} and {DEST/'B.npy'}  shape={A.shape} dtype={A.dtype}")


if __name__ == "__main__":
    main()
