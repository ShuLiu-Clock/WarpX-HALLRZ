"""HallRZ runtime boundary-data helpers.

This module only passes physical boundary data to the C++ HallRZ path.  It
does not pass EB geometry; WarpX C++ still owns EB2 and nodal FVM geometry.
"""

import numpy as np

from ._libwarpx import libwarpx


def set_eb_neumann(g_eb_2d):
    """Set persistent cell-centered EB Neumann data.

    Parameters
    ----------
    g_eb_2d : numpy.ndarray
        C-contiguous float64 array with shape ``(Nr, Nz)`` storing
        ``g_EB=dphi/dn`` at cell centers.  Only EB2 single-valued cells consume
        these values during HallRZ Poisson solves.
    """
    arr = np.asarray(g_eb_2d)
    if arr.dtype != np.float64:
        raise TypeError("hallrz.set_eb_neumann expects dtype np.float64")
    if arr.ndim != 2:
        raise ValueError("hallrz.set_eb_neumann expects a 2D array with shape (Nr, Nz)")
    if not arr.flags.c_contiguous:
        raise ValueError("hallrz.set_eb_neumann expects a C-contiguous array")
    libwarpx.libwarpx_so.hallrz_set_eb_neumann(arr)


def clear_eb_neumann():
    """Clear persistent Python EB Neumann data and return to scalar/default path."""
    libwarpx.libwarpx_so.hallrz_clear_eb_neumann()


def has_eb_neumann():
    """Return True if persistent Python EB Neumann data is currently active."""
    return libwarpx.libwarpx_so.hallrz_has_eb_neumann()


def _as_1d_float64(name, data):
    arr = np.asarray(data)
    if arr.dtype != np.float64:
        raise TypeError(f"hallrz.{name} expects dtype np.float64")
    if arr.ndim != 1:
        raise ValueError(f"hallrz.{name} expects a 1D node-centered array")
    if not arr.flags.c_contiguous:
        raise ValueError(f"hallrz.{name} expects a C-contiguous array")
    return arr


def set_robin_zhi(a, b, f):
    """Set persistent node-centered z-hi Robin data.

    Arrays must have shape ``(Nr+1,)`` and represent
    ``a*phi + b*dphi/dz = f`` on ``z=zmax``.
    """
    aa = _as_1d_float64("set_robin_zhi(a)", a)
    bb = _as_1d_float64("set_robin_zhi(b)", b)
    ff = _as_1d_float64("set_robin_zhi(f)", f)
    if aa.shape != bb.shape or aa.shape != ff.shape:
        raise ValueError("hallrz.set_robin_zhi expects a, b, and f to have the same shape")
    libwarpx.libwarpx_so.hallrz_set_robin_zhi(aa, bb, ff)


def clear_robin_zhi():
    """Clear persistent z-hi Robin data and return to scalar/default path."""
    libwarpx.libwarpx_so.hallrz_clear_robin_zhi()


def has_robin_zhi():
    """Return True if persistent Python z-hi Robin data is currently active."""
    return libwarpx.libwarpx_so.hallrz_has_robin_zhi()


def set_robin_rhi(a, b, f):
    """Set persistent node-centered r-hi Robin data.

    Arrays must have shape ``(Nz+1,)`` and represent
    ``a*phi + b*dphi/dr = f`` on ``r=rmax``.
    """
    aa = _as_1d_float64("set_robin_rhi(a)", a)
    bb = _as_1d_float64("set_robin_rhi(b)", b)
    ff = _as_1d_float64("set_robin_rhi(f)", f)
    if aa.shape != bb.shape or aa.shape != ff.shape:
        raise ValueError("hallrz.set_robin_rhi expects a, b, and f to have the same shape")
    libwarpx.libwarpx_so.hallrz_set_robin_rhi(aa, bb, ff)


def clear_robin_rhi():
    """Clear persistent r-hi Robin data and return to scalar/default path."""
    libwarpx.libwarpx_so.hallrz_clear_robin_rhi()


def has_robin_rhi():
    """Return True if persistent Python r-hi Robin data is currently active."""
    return libwarpx.libwarpx_so.hallrz_has_robin_rhi()


def set_inlet_dirichlet(phi):
    """Set persistent node-centered z-lo inlet Dirichlet data.

    The array must have shape ``(Nr+1,)``.  C++ only applies it on active
    channel inlet nodes.
    """
    pp = _as_1d_float64("set_inlet_dirichlet(phi)", phi)
    libwarpx.libwarpx_so.hallrz_set_inlet_dirichlet(pp)


def clear_inlet_dirichlet():
    """Clear persistent inlet Dirichlet data and return to scalar/default path."""
    libwarpx.libwarpx_so.hallrz_clear_inlet_dirichlet()


def has_inlet_dirichlet():
    """Return True if persistent Python inlet Dirichlet data is currently active."""
    return libwarpx.libwarpx_so.hallrz_has_inlet_dirichlet()
