#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 16:53:40 2025

@author: duoxup
"""
from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Mapping, Optional, Type

import numpy as np
from numpy.lib.scimath import sqrt
from scipy.special import jv, jvp

from waveguides.utils import EField2D, ModeInfo
from .bessel_zeros import get_tables

# *********************************************************
# %% Constants
# *********************************************************

C_LIGHT = 299_792_458.0  # speed of light in vacuum (m/s), exact by SI definition


# *********************************************************
# %% Rectangular waveguide helpers
# *********************************************************

_M_MAX_REC, _N_MAX_REC = 100, 100
_IDX_ROWS_REC = _M_MAX_REC + 1
_IDX_COLS_REC = _N_MAX_REC + 1


def calc_sorted_wavenumber_and_indices_rec(a, b):
    kc_raw = np.zeros([_M_MAX_REC + 1, _N_MAX_REC + 1])
    for m in range(_M_MAX_REC + 1):
        for n in range(_N_MAX_REC + 1):
            kc_raw[m, n] = np.sqrt(((m) * np.pi / a)**2 + ((n) * np.pi / b)**2)
    kc_te = kc_raw.copy()
    kc_te[0, 0] = np.nan
    kc_tm = kc_raw.copy() * (1 + 1e-12)
    kc_tm[0, :] = np.nan
    kc_tm[:, 0] = np.nan
    # Prevent degenerate TE and TM modes from having the same cutoff frequency.
    # kc of TM modes is manually increased by a small epsilon.
    kc_all = np.block([kc_te, kc_tm]).flatten('F')
    kc_sorted, idx_sorted = np.sort(kc_all), np.argsort(kc_all)
    return kc_sorted, idx_sorted


def calc_mode_info_rec(a, b, kmn, idx, er):
    """Internal use only. Calculate mode info from kc and corresponding index."""
    mode_info = ModeInfo()
    mode_info.kc = kmn
    mode_info.mode_type = 1 if idx < _IDX_ROWS_REC * _IDX_COLS_REC else 0
    m = int(np.mod(idx, _IDX_ROWS_REC))
    n = int(np.mod(np.floor(idx / _IDX_ROWS_REC), _IDX_COLS_REC))
    mode_info.mode_num1 = m
    mode_info.mode_num2 = n
    mode_info.fc = mode_info.kc * C_LIGHT / np.sqrt(er) / 2 / np.pi
    mode_info.norm_constant = norm_mn(m, n, a, b, kmn)
    return mode_info


def calc_sorted_mode_info_rec(a, b, N, er):
    kc_sorted, idx_sorted = calc_sorted_wavenumber_and_indices_rec(a, b)
    mode_infos = []
    for i in range(N):
        mode_info = calc_mode_info_rec(a, b, kc_sorted[i], idx_sorted[i], er)
        mode_info.plus_dir = 1
        mode_info.polar_dir = -1
        mode_infos.append(mode_info)
    return mode_infos


def norm_mn(m, n, a, b, kc):
    """Normalisation constant N_{mn} for rectangular waveguide modes."""
    eps_m = 1 if m == 0 else 2
    eps_n = 1 if n == 0 else 2
    norm = np.sqrt(eps_m * eps_n / a / b) / kc
    return norm


def alpha_rec_te(a, b, r_s, m, n, k, kc):
    """Attenuation constant for TE_{mn} mode in a rectangular waveguide."""
    eps_m = 2 if m == 0 else 1
    eps_n = 2 if n == 0 else 1
    alpha = (
        1 / (60 * np.pi) * r_s
        / (eps_m * eps_n * sqrt(1 - (kc / k)**2))
        * ((kc / k)**2 * (eps_m / b + eps_n / a)
           + (1 - (kc / k)**2) * ((m**2 * b + n**2 * a) / (m**2 * b**2 + n**2 * a**2)))
    )
    return alpha


def alpha_rec_tm(a, b, r_s, m, n, k, kc):
    """Attenuation constant for TM_{mn} mode in a rectangular waveguide."""
    alpha = (
        r_s / sqrt(1 - (kc / k)**2)
        * (m**2 * b**3 + n**2 * a**3) / (m**2 * b**2 + n**2 * a**2)
        / (60 * np.pi * a * b)
    )
    return alpha


def calc_propagation_factor_rec(a, b, er, maxmodeN, sigma, f, l, mode_mat, lossless=False):
    k = 2 * np.pi * f / C_LIGHT * np.sqrt(er)
    ps = np.zeros(maxmodeN, dtype=complex)
    r_s = np.sqrt(np.pi * f * 4 * np.pi * 1e-7 / sigma)
    if lossless:
        r_s = 0
    for i in range(maxmodeN):
        beta = sqrt(k**2 - mode_mat[i, 0]**2)
        if mode_mat[i, 1] > 0:  # TE mode
            alpha = alpha_rec_te(a, b, r_s, int(mode_mat[i, 2]), int(mode_mat[i, 3]), k, mode_mat[i, 0])
        else:  # TM mode
            alpha = alpha_rec_tm(a, b, r_s, int(mode_mat[i, 2]), int(mode_mat[i, 3]), k, mode_mat[i, 0])
        ps[i] = np.exp(-(np.imag(beta) + np.abs(alpha) + 1j * np.real(beta)) * l)
    return ps


# *********************************************************
# %% Circular waveguide helpers
# *********************************************************

def _load_bessel_tables():
    return get_tables()


# _JN_ZEROS[q, r-1]  : r-th zero of J_q(x)         — used for TM modes
# _JNP_ZEROS[q, r-1] : r-th zero of J'_q(x)        — used for TE modes
_JN_ZEROS, _JNP_ZEROS = _load_bessel_tables()
_IDX_ROWS_CIR = _JN_ZEROS.shape[0]
_IDX_COLS_CIR = _JN_ZEROS.shape[1]


def calc_sorted_wavenumber_and_indices_cir(r):
    # Prevent degenerate TE and TM modes from having the same cutoff frequency.
    # kc of TM modes is manually increased by a small epsilon.
    C = np.block([_JNP_ZEROS, _JN_ZEROS * (1 + 1e-12)]).flatten('F')
    hqr_sorted = np.sort(C)
    idx_sorted = np.argsort(C)
    kqr_sorted = hqr_sorted / r
    return kqr_sorted, idx_sorted


def calc_mode_info_cir(kqr, idx, er):
    """Calculate mode info from kc and corresponding index."""
    mode_info = ModeInfo()
    mode_info.kc = kqr
    mode_type = 1 if idx < _IDX_ROWS_CIR * _IDX_COLS_CIR else 0
    mode_info.mode_type = mode_type
    q = int(np.mod(idx, _IDX_ROWS_CIR))
    r = int(np.mod(np.floor(idx / _IDX_ROWS_CIR), _IDX_COLS_CIR) + 1)
    mode_info.mode_num1 = q
    mode_info.mode_num2 = r
    mode_info.fc = mode_info.kc * C_LIGHT / np.sqrt(er) / 2 / np.pi
    mode_info.norm_constant = norm_qr_te(q, r) if mode_type > 0 else norm_qr_tm(q, r)
    return mode_info


def calc_sorted_mode_info_cir(r, N, er):
    kqr_sorted, idx_sorted = calc_sorted_wavenumber_and_indices_cir(r)
    i = 0
    j = 0
    mode_infos = []
    while i < N:
        mode_info = calc_mode_info_cir(kqr_sorted[j], idx_sorted[j], er)
        if mode_info.mode_num1 <= 0:
            mode_info.plus_dir = -1
            mode_info.polar_dir = 1
            mode_infos.append(mode_info)
            i += 1
            j += 1
        else:
            mode_info_p1 = copy.deepcopy(mode_info)
            mode_info.plus_dir = 1 if (mode_info.mode_type == 0 and np.mod(
                mode_info_p1.mode_num1, 2) == 0) else -1
            mode_info.polar_dir = 0 if mode_info.mode_type == 1 else 1
            mode_info_p1.plus_dir = -1 if (mode_info_p1.mode_type == 0 and np.mod(
                mode_info_p1.mode_num1, 2) != 0) else 1
            mode_info_p1.polar_dir = 1 - mode_info.polar_dir
            mode_infos.append(mode_info)
            if i < N - 1:
                mode_infos.append(mode_info_p1)
            i += 2
            j += 1
    return mode_infos


def norm_qr_te(q, r):
    """Normalisation constant N_{qr} for TE modes in a circular waveguide."""
    eps_q = 1 if q == 0 else 2
    kr = _JNP_ZEROS[q, r - 1]
    norm = np.sqrt(eps_q / np.pi) / (np.sqrt(kr * kr - q * q) * jv(q, kr))
    return norm


def norm_qr_tm(q, r):
    """Normalisation constant N_{qr} for TM modes in a circular waveguide."""
    eps_q = 1 if q == 0 else 2
    kr = _JN_ZEROS[q, r - 1]
    norm = np.sqrt(eps_q / np.pi) / (kr * jvp(q, kr))
    return norm


def alpha_cir_te(rc, r_s, q, k, kc):
    """Attenuation constant for TE_{qr} mode in a circular waveguide."""
    alpha = (
        r_s / sqrt(1 - (kc / k)**2)
        * (q**2 / (kc**2 * rc**2 - q**2) + (kc / k)**2)
        / (120 * np.pi * rc)
    )
    return alpha


def alpha_cir_tm(rc, r_s, k, kc):
    """Attenuation constant for TM_{qr} mode in a circular waveguide."""
    alpha = r_s / sqrt(1 - (kc / k)**2) / (120 * np.pi * rc)
    return alpha


def calc_propagation_factor_cir(rc, er, maxmodeN, sigma, f, l, mode_mat, lossless=False):
    k = 2 * np.pi * f / C_LIGHT * np.sqrt(er)
    ps = np.zeros(maxmodeN, dtype=complex)
    r_s = np.sqrt(np.pi * f * 4 * np.pi * 1e-7 / sigma)
    if lossless:
        r_s = 0
    for i in range(maxmodeN):
        beta = sqrt(k**2 - mode_mat[i, 0]**2)
        if mode_mat[i, 1] > 0:  # TE mode
            alpha = alpha_cir_te(rc, r_s, int(mode_mat[i, 2]), k, mode_mat[i, 0])
        else:  # TM mode
            alpha = alpha_cir_tm(rc, r_s, k, mode_mat[i, 0])
        ps[i] = np.exp(-(np.imag(beta) + np.abs(alpha) + 1j * np.real(beta)) * l)
    return ps


# *********************************************************
# %% Waveguide base class
# *********************************************************

class WG(ABC):
    """
    Base waveguide class with built-in serialization.

    Dump format (dict):
      {
        "version": 1,
        "wg_type": "<registry_key>",
        "l": float,
        "N": int,
        "er": float,
        "sigma": float,
        "cross_tag": str,
        "params": { ... subclass-specific ... }
      }
    """

    _SERIAL_VERSION: ClassVar[int] = 1
    _registry: ClassVar[Dict[str, Type["WG"]]] = {}
    _wg_type: ClassVar[str] = "WG"

    def __init_subclass__(cls, *, wg_type: Optional[str] = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        key = wg_type or cls.__name__
        cls._wg_type = key
        WG._registry[key] = cls

    def __init__(self, l, N, er, sigma):
        self.l = l
        self.N = N
        self.er = er
        self.sigma = sigma
        self.cross_tag = ""

    # -------------------------
    # Serialization
    # -------------------------

    def dump(self) -> Dict[str, Any]:
        """Serialize this WG instance into a JSON-compatible dict."""
        return {
            "version": self._SERIAL_VERSION,
            "wg_type": self._wg_type,
            "l": float(self.l),
            "N": int(self.N),
            "er": float(self.er),
            "sigma": float(self.sigma),
            "cross_tag": str(self.cross_tag),
            "params": self._dump_params(),
        }

    @classmethod
    def load(cls, data: Mapping[str, Any]) -> "WG":
        """
        Deserialize a WG instance from a dict produced by dump().

        Dispatches to the correct subclass by ``wg_type``.
        ``cls`` is ignored; the embedded wg_type selects the implementation.
        """
        if "wg_type" not in data:
            raise KeyError("WG.load: missing required key 'wg_type'.")

        wg_type = str(data["wg_type"])
        impl = WG._registry.get(wg_type)
        if impl is None:
            known = ", ".join(sorted(WG._registry.keys()))
            raise ValueError(f"WG.load: unknown wg_type={wg_type!r}. Known types: {known}")

        version = int(data.get("version", 0))
        if version != WG._SERIAL_VERSION:
            raise ValueError(
                f"WG.load: unsupported version={version}. Expected {WG._SERIAL_VERSION}."
            )

        return impl._load_from_dump(data)

    def _dump_params(self) -> Dict[str, Any]:
        """Subclasses override to dump geometry/config needed to reconstruct the object."""
        return {}

    @classmethod
    def _load_from_dump(cls, data: Mapping[str, Any]) -> "WG":
        """
        Default reconstruction logic: call constructor with base fields + params.
        Subclasses may override for special cases.
        """
        try:
            l = data["l"]
            N = data["N"]
            er = data["er"]
            sigma = data["sigma"]
        except KeyError as e:
            raise KeyError(f"{cls.__name__}._load_from_dump: missing required key {e!s}") from e

        params = dict(data.get("params", {}))
        params.update({"l": l, "N": N, "er": er, "sigma": sigma})
        obj = cls(**params)  # type: ignore[misc]

        if "cross_tag" in data:
            obj.cross_tag = str(data["cross_tag"])
        return obj

    # -------------------------
    # Physical quantities
    # -------------------------

    def wavelength_at(self, f):
        k0 = f / C_LIGHT * 2 * np.pi * np.sqrt(self.er)
        kz_list = []
        for mode_info in self.mode_info_list:
            kc = mode_info.kc
            kz = np.sqrt(complex(k0**2 - kc**2))
            kz_list.append(kz)
        wavelengths = 2 * np.pi / np.array(kz_list)
        return wavelengths

    def wavelength_at_list(self, f_list):
        wavelength_list = []
        for f in f_list:
            wavelength_list.append(self.wavelength_at(f))
        return np.array(wavelength_list)

    def impedance_at(self, f):
        info_mat = self.mode_info_array()
        k = 2 * np.pi * f / C_LIGHT * np.sqrt(self.er)
        Z = np.zeros(self.N, dtype=complex)
        for i in range(self.N):
            beta = np.sqrt((k**2 - info_mat[i, 0]**2).astype(np.complex128))
            if info_mat[i, 1] > 0:  # TE mode
                Z[i] = k / beta * 120 * np.pi / np.sqrt(self.er)
            else:  # TM mode
                Z[i] = beta / k * 120 * np.pi / np.sqrt(self.er)
        return Z

    def impedance_at_list(self, f_list):
        return np.array([self.impedance_at(f) for f in f_list])

    def gamma_at(self, f):
        raise NotImplementedError()

    def gamma_at_list(self, f_list):
        raise NotImplementedError()

    def propagation_factor_at(self, f, lossless=False):
        match self.cross_tag:
            case 'cir':
                ps = calc_propagation_factor_cir(
                    self.r, self.er, self.N, self.sigma, f, self.l,
                    self.mode_info_array(), lossless=lossless,
                )
            case 'rec':
                ps = calc_propagation_factor_rec(
                    self.a, self.b, self.er, self.N, self.sigma, f, self.l,
                    self.mode_info_array(), lossless=lossless,
                )
            case _:
                raise TypeError('Unknown waveguide type.')
        return ps

    def propagation_factor_at_list(self, f_list, lossless=False):
        return np.array([self.propagation_factor_at(f, lossless=lossless) for f in f_list])

    def phaseshift_at(self, f):
        """Return the phase shift (rad) for each mode at frequency f.

        Defined as the argument of the propagation factor:
        angle(exp(-(alpha + j*beta) * l)) = -beta * l (rad).
        """
        return np.angle(self.propagation_factor_at(f, lossless=True))

    def phaseshift_at_list(self, f_list):
        """Return the phase shift (rad) for each mode over a list of frequencies.

        Shape: (len(f_list), N).
        """
        return np.array([self.phaseshift_at(f) for f in f_list])

    def mode_info_array(self):
        """Return mode info as a 2D numpy array with shape (N, 8)."""
        out = np.zeros((len(self.mode_info_list), len(self.mode_info_list[0].to_array())))
        for idx, mode in enumerate(self.mode_info_list):
            out[idx, :] = mode.to_array()
        return out

    @abstractmethod
    def get_mode_efield_distribution_at_gridpoints(self, mode_idx, X, Y) -> EField2D:
        pass

    @property
    def modeInfoMat(self):
        """Deprecated: use ``mode_info_list`` instead."""
        warnings.warn(
            "modeInfoMat is deprecated; use mode_info_list instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mode_info_list

    def get_readable_info_mat(self):
        """Deprecated: use ``mode_info_array()`` instead."""
        warnings.warn(
            "get_readable_info_mat() is deprecated; use mode_info_array() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mode_info_array()

    @property
    def normalization_constant(self):
        return np.array([mode.norm_constant for mode in self.mode_info_list])


# *********************************************************
# %% Rectangular waveguide
# *********************************************************

class RecWG(WG):
    """Rectangular waveguide."""

    def __init__(self, a=1, b=0.5, l=1, N=1, er=1, sigma=5.8e7):
        super().__init__(l, N, er, sigma)
        self.a = a
        self.b = b
        self.mode_info_list = calc_sorted_mode_info_rec(self.a, self.b, self.N, self.er)
        self.cross_tag = 'rec'

    def _dump_params(self) -> Dict[str, Any]:
        return {"a": float(self.a), "b": float(self.b)}

    def get_mode_efield_distribution_at_gridpoints(self, mode_idx, X, Y) -> EField2D:
        mode_info = self.mode_info_list[mode_idx]
        m = mode_info.mode_num1
        n = mode_info.mode_num2
        x_m = X + self.a / 2
        y_m = Y + self.b / 2
        if mode_info.mode_type == 1:  # TE
            e_x = norm_mn(m, n, self.a, self.b, mode_info.kc) * n * np.pi / self.b * np.cos(m * np.pi * x_m / self.a) * np.sin(n * np.pi * y_m / self.b)
            e_y = -norm_mn(m, n, self.a, self.b, mode_info.kc) * m * np.pi / self.a * np.sin(m * np.pi * x_m / self.a) * np.cos(n * np.pi * y_m / self.b)
        elif mode_info.mode_type == 0:  # TM
            e_x = -norm_mn(m, n, self.a, self.b, mode_info.kc) * m * np.pi / self.a * np.cos(m * np.pi * x_m / self.a) * np.sin(n * np.pi * y_m / self.b)
            e_y = -norm_mn(m, n, self.a, self.b, mode_info.kc) * n * np.pi / self.b * np.sin(m * np.pi * x_m / self.a) * np.cos(n * np.pi * y_m / self.b)
        else:
            raise ValueError('mode_type must be 1 (TE) or 0 (TM).')
        e_x[np.abs(X) > self.a / 2] = 0
        e_x[np.abs(Y) > self.b / 2] = 0
        e_y[np.abs(X) > self.a / 2] = 0
        e_y[np.abs(Y) > self.b / 2] = 0
        field = EField2D(
            X=X, Y=Y,
            Ex=np.asarray(e_x, dtype=complex),
            Ey=np.asarray(e_y, dtype=complex),
            Ez=np.zeros_like(e_x, dtype=complex),
        )
        return field

    @property
    def mode_name_list(self):
        mode_matrix = self.mode_info_array()
        names = []
        for row in mode_matrix:
            pol = 'TE' if row[1] > 0 else 'TM'
            names.append(f"{pol}{row[2]:.0f},{row[3]:.0f}")
        return names

    @property
    def mode_name_list_latex(self):
        mode_matrix = self.mode_info_array()
        names = []
        for row in mode_matrix:
            pol = 'TE' if row[1] > 0 else 'TM'
            sub = '{' + f'{row[2]:.0f},{row[3]:.0f}' + '}'
            names.append(r'$\mathrm{' + f'{pol}_{sub}' + r'}$')
        return names

    @property
    def mode_name_list_for_mpl(self):
        """Deprecated: use ``mode_name_list_latex`` instead."""
        warnings.warn(
            "mode_name_list_for_mpl is deprecated; use mode_name_list_latex instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mode_name_list_latex

    @property
    def cross_dim_text(self):
        return rf'a{self.a * 1e3}_b{self.b * 1e3}'

    def __repr__(self):
        return '_'.join([
            self.cross_tag, self.cross_dim_text,
            f'l{self.l * 1e3}', f'er{self.er}', f'n{self.N}',
            f'sig{self.sigma:.2E}',
        ])


# *********************************************************
# %% Circular waveguide
# *********************************************************

class CirWG(WG):
    """Circular waveguide."""

    def __init__(self, r=1, l=1, N=1, er=1, sigma=5.8e7):
        super().__init__(l, N, er, sigma)
        self.r = r
        self.mode_info_list = calc_sorted_mode_info_cir(r, N, er)
        self.cross_tag = 'cir'

    def _dump_params(self) -> Dict[str, Any]:
        return {"r": float(self.r)}

    def get_mode_efield_distribution_at_gridpoints(self, mode_idx, X, Y) -> EField2D:
        R = np.sqrt(X * X + Y * Y)
        R[R == 0] = 1e-15  # avoid singularity at origin
        T = np.mod(np.arctan2(Y, X), 2 * np.pi)
        mode_info = self.mode_info_list[mode_idx]
        q = mode_info.mode_num1
        r = mode_info.mode_num2
        if mode_info.mode_type == 1 and mode_info.polar_dir == 0:
            pol_case = 1  # TE-S
        elif mode_info.mode_type == 1 and mode_info.polar_dir == 1:
            pol_case = 2  # TE-C
        elif mode_info.mode_type == 0 and mode_info.polar_dir == 0:
            pol_case = 3  # TM-S
        else:
            pol_case = 4  # TM-C
        match pol_case:
            case 1:  # TE-S
                e_r = -norm_qr_te(q, r) * q / R * jv(q, _JNP_ZEROS[q, r-1] / self.r * R) * np.cos(q * T) * mode_info.plus_dir
                e_t = norm_qr_te(q, r) * _JNP_ZEROS[q, r-1] / self.r * jvp(q, _JNP_ZEROS[q, r-1] / self.r * R) * np.sin(q * T) * mode_info.plus_dir
            case 2:  # TE-C
                e_r = -norm_qr_te(q, r) * q / R * jv(q, _JNP_ZEROS[q, r-1] / self.r * R) * -np.sin(q * T) * mode_info.plus_dir
                e_t = norm_qr_te(q, r) * _JNP_ZEROS[q, r-1] / self.r * jvp(q, _JNP_ZEROS[q, r-1] / self.r * R) * np.cos(q * T) * mode_info.plus_dir
            case 3:  # TM-S
                e_r = -norm_qr_tm(q, r) * _JN_ZEROS[q, r-1] / self.r * jvp(q, _JN_ZEROS[q, r-1] / self.r * R) * np.sin(q * T) * mode_info.plus_dir
                e_t = -norm_qr_tm(q, r) * q / R * jv(q, _JN_ZEROS[q, r-1] / self.r * R) * np.cos(q * T) * mode_info.plus_dir
            case 4:  # TM-C
                e_r = -norm_qr_tm(q, r) * _JN_ZEROS[q, r-1] / self.r * jvp(q, _JN_ZEROS[q, r-1] / self.r * R) * np.cos(q * T) * mode_info.plus_dir
                e_t = -norm_qr_tm(q, r) * q / R * jv(q, _JN_ZEROS[q, r-1] / self.r * R) * -np.sin(q * T) * mode_info.plus_dir
        e_r[R > self.r] = 0
        e_t[R > self.r] = 0
        e_x = e_r * np.cos(T) - e_t * np.sin(T)
        e_y = e_r * np.sin(T) + e_t * np.cos(T)
        field = EField2D(
            X=X, Y=Y,
            Ex=np.asarray(e_x, dtype=complex),
            Ey=np.asarray(e_y, dtype=complex),
            Ez=np.zeros_like(e_x, dtype=complex),
        )
        return field

    @property
    def mode_name_list(self):
        mode_matrix = self.mode_info_array()
        names = []
        for row in mode_matrix:
            pol = 'TE' if row[1] > 0 else 'TM'
            ori = 'C' if row[6] > 0 else 'S'
            names.append(f"{pol}{row[2]:.0f},{row[3]:.0f}{ori}")
        return names

    @property
    def mode_name_list_latex(self):
        mode_matrix = self.mode_info_array()
        names = []
        for row in mode_matrix:
            pol = 'TE' if row[1] > 0 else 'TM'
            sub = '{' + f'{row[2]:.0f},{row[3]:.0f}' + '}'
            ori = 'C' if row[6] > 0 else 'S'
            names.append(r'$\mathrm{' + f'{pol}_{sub}^{ori}' + r'}$')
        return names

    @property
    def _mode_name_list_internal(self):
        mode_matrix = self.mode_info_array()
        names = []
        for row in mode_matrix:
            sign = '+' if row[5] > 0 else '-'
            pol = 'TE' if row[1] > 0 else 'TM'
            ori = 'C' if row[6] > 0 else 'S'
            names.append(f"{sign}{pol}{row[2]:.0f},{row[3]:.0f}{ori}")
        return names

    @property
    def mode_name_list_for_mpl(self):
        """Deprecated: use ``mode_name_list_latex`` instead."""
        warnings.warn(
            "mode_name_list_for_mpl is deprecated; use mode_name_list_latex instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mode_name_list_latex

    @property
    def cross_dim_text(self):
        return rf'r{self.r * 1e3}'

    def __repr__(self):
        return '_'.join([
            self.cross_tag, self.cross_dim_text,
            f'l{self.l * 1e3}', f'er{self.er}', f'n{self.N}',
            f'sig{self.sigma:.2E}',
        ])


# *********************************************************
# %% Deprecated aliases — will be removed in a future version
# *********************************************************

def n_mn(*args, **kwargs):
    """Deprecated: use ``norm_mn()`` instead."""
    warnings.warn("n_mn() is deprecated; use norm_mn() instead.",
                  DeprecationWarning, stacklevel=2)
    return norm_mn(*args, **kwargs)


def n_qr_te(*args, **kwargs):
    """Deprecated: use ``norm_qr_te()`` instead."""
    warnings.warn("n_qr_te() is deprecated; use norm_qr_te() instead.",
                  DeprecationWarning, stacklevel=2)
    return norm_qr_te(*args, **kwargs)


def n_qr_tm(*args, **kwargs):
    """Deprecated: use ``norm_qr_tm()`` instead."""
    warnings.warn("n_qr_tm() is deprecated; use norm_qr_tm() instead.",
                  DeprecationWarning, stacklevel=2)
    return norm_qr_tm(*args, **kwargs)


def get_bessel_matrix_from_file():
    """Deprecated: use ``get_tables()`` from ``waveguides.bessel_zeros`` instead."""
    warnings.warn(
        "get_bessel_matrix_from_file() is deprecated; "
        "use get_tables() from waveguides.bessel_zeros instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _load_bessel_tables()
