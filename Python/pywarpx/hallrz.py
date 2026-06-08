"""HallRZ runtime boundary-data helpers.

This module only passes physical boundary data to the C++ HallRZ path.  It
does not pass EB geometry; WarpX C++ still owns EB2 and nodal FVM geometry.
"""

import numpy as np

from ._libwarpx import libwarpx


def _as_host_array(data):
    """Return a NumPy view/copy, explicitly copying CUDA arrays to host."""
    if hasattr(data, "__cuda_array_interface__"):
        if not hasattr(data, "get"):
            raise TypeError("hallrz CUDA array inputs must provide a get() method")
        data = data.get()
    return np.asarray(data)


def _boundary_level(lev, *, allow_all=False):
    if lev is None:
        if allow_all:
            return -1
        raise ValueError("hallrz boundary level cannot be None")
    lev = int(lev)
    if allow_all and lev == -1:
        return lev
    if lev not in (0, 1, 2):
        raise ValueError("hallrz boundary level currently supports only lev=0,1,2")
    return lev


def set_eb_neumann(g_eb_2d, lev=0):
    """Set persistent cell-centered EB Neumann data.

    Parameters
    ----------
    g_eb_2d : array-like
        C-contiguous float64 array with shape ``(Nr, Nz)`` storing
        ``g_EB=dphi/dn_EB`` at cell centers, where ``n_EB`` points from
        computational/fluid region to covered region.  Only EB2 single-valued
        cells consume these values during HallRZ Poisson solves.
    """
    lev = _boundary_level(lev)
    arr = _as_host_array(g_eb_2d)
    if arr.dtype != np.float64:
        raise TypeError("hallrz.set_eb_neumann expects dtype np.float64")
    if arr.ndim != 2:
        raise ValueError("hallrz.set_eb_neumann expects a 2D array with shape (Nr, Nz)")
    if not arr.flags.c_contiguous:
        raise ValueError("hallrz.set_eb_neumann expects a C-contiguous array")
    libwarpx.libwarpx_so.hallrz_set_eb_neumann(arr, lev)


def clear_eb_neumann(lev=None):
    """Clear persistent Python EB Neumann data.

    ``lev=None`` clears all stored HallRZ EB levels.
    """
    libwarpx.libwarpx_so.hallrz_clear_eb_neumann(_boundary_level(lev, allow_all=True))


def has_eb_neumann(lev=None):
    """Return True if persistent Python EB Neumann data is active."""
    return libwarpx.libwarpx_so.hallrz_has_eb_neumann(_boundary_level(lev, allow_all=True))


def _as_2d_float64(name, data):
    arr = _as_host_array(data)
    if arr.dtype != np.float64:
        raise TypeError(f"hallrz.{name} expects dtype np.float64")
    if arr.ndim != 2:
        raise ValueError(f"hallrz.{name} expects a 2D array with shape (Nr, Nz)")
    if not arr.flags.c_contiguous:
        raise ValueError(f"hallrz.{name} expects a C-contiguous array")
    return arr


def set_eb_robin(a, b, f, lev=0):
    """Set persistent cell-centered EB-FVM Robin data.

    Arrays must have shape ``(Nr, Nz)`` in WarpX RZ order ``(r,z)`` and store
    AMReX raw EB-FVM data ``a*phi + b*dphi/dn_EB = f``.  The EB normal
    ``n_EB`` points from computational/fluid region to covered region.  WarpX
    does not flip signs from segment or outward-normal conventions.
    """
    lev = _boundary_level(lev)
    aa = _as_2d_float64("set_eb_robin(a)", a)
    bb = _as_2d_float64("set_eb_robin(b)", b)
    ff = _as_2d_float64("set_eb_robin(f)", f)
    if aa.shape != bb.shape or aa.shape != ff.shape:
        raise ValueError("hallrz.set_eb_robin expects a, b, and f to have the same shape")
    libwarpx.libwarpx_so.hallrz_set_eb_robin(aa, bb, ff, lev)


def clear_eb_robin(lev=None):
    """Clear persistent Python EB Robin data.

    ``lev=None`` clears all stored HallRZ EB levels.
    """
    libwarpx.libwarpx_so.hallrz_clear_eb_robin(_boundary_level(lev, allow_all=True))


def has_eb_robin(lev=None):
    """Return True if persistent Python EB Robin data is active."""
    return libwarpx.libwarpx_so.hallrz_has_eb_robin(_boundary_level(lev, allow_all=True))


def _as_1d_float64(name, data):
    arr = _as_host_array(data)
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


def set_robin_rlo(a, b, f):
    """Set persistent node-centered r-lo Robin data.

    Arrays must have shape ``(Nz+1,)`` and represent
    AMReX raw physical-face data ``a*phi + b*dphi/dr = f`` on ``r=rmin``.
    This boundary is only valid when ``rmin > 0``.
    """
    aa = _as_1d_float64("set_robin_rlo(a)", a)
    bb = _as_1d_float64("set_robin_rlo(b)", b)
    ff = _as_1d_float64("set_robin_rlo(f)", f)
    if aa.shape != bb.shape or aa.shape != ff.shape:
        raise ValueError("hallrz.set_robin_rlo expects a, b, and f to have the same shape")
    libwarpx.libwarpx_so.hallrz_set_robin_rlo(aa, bb, ff)


def clear_robin_rlo():
    """Clear persistent r-lo Robin data and return to scalar/default path."""
    libwarpx.libwarpx_so.hallrz_clear_robin_rlo()


def has_robin_rlo():
    """Return True if persistent Python r-lo Robin data is currently active."""
    return libwarpx.libwarpx_so.hallrz_has_robin_rlo()


def set_dirichlet_rlo(phi):
    """Set persistent node-centered r-lo Dirichlet data.

    The array must have shape ``(Nz+1,)``.  This boundary is only valid when
    ``rmin > 0``.
    """
    pp = _as_1d_float64("set_dirichlet_rlo(phi)", phi)
    libwarpx.libwarpx_so.hallrz_set_dirichlet_rlo(pp)


def clear_dirichlet_rlo():
    """Clear persistent r-lo Dirichlet data and return to scalar/default path."""
    libwarpx.libwarpx_so.hallrz_clear_dirichlet_rlo()


def has_dirichlet_rlo():
    """Return True if persistent Python r-lo Dirichlet data is currently active."""
    return libwarpx.libwarpx_so.hallrz_has_dirichlet_rlo()


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
