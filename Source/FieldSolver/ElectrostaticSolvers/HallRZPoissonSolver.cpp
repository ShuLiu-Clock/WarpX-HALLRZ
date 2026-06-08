/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "HallRZPoissonSolver.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_Box.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLEBNodeFVLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Rect
{
    amrex::Real rlo;
    amrex::Real rhi;
    amrex::Real zlo;
    amrex::Real zhi;
};

struct HallDomain
{
    amrex::Real rlo;
    amrex::Real rhi;
    amrex::Real zlo;
    amrex::Real zhi;
    amrex::Real lob;
    amrex::Real hib;
    amrex::Real out;
    amrex::Real align_tol;
};

struct NodeGeom
{
    amrex::Real v2d = 0.0;
    amrex::Real vrz = 0.0;
    amrex::Real arm = 0.0;
    amrex::Real arp = 0.0;
    amrex::Real azm = 0.0;
    amrex::Real azp = 0.0;
    amrex::Real aeb = 0.0;
    amrex::Real arm_rz = 0.0;
    amrex::Real arp_rz = 0.0;
    amrex::Real azm_rz = 0.0;
    amrex::Real azp_rz = 0.0;
    amrex::Real aeb_rz = 0.0;
};

struct EBFluxStats
{
    amrex::Real total = 0.0;
    std::array<amrex::Real,4> segment{{0.0, 0.0, 0.0, 0.0}};
};

struct EBRobinStats
{
    amrex::Long active = 0;
    amrex::Long invalid_nonfinite = 0;
    amrex::Long invalid_degenerate = 0;
    std::array<amrex::Long,4> segment{{0, 0, 0, 0}};
};

constexpr int max_hallrz_python_eb_levels = 3;

struct PythonEBNeumannData
{
    bool active = false;
    int nr = 0;
    int nz = 0;
    std::vector<amrex::Real> data;
};

struct PythonEBRobinData
{
    bool active = false;
    int nr = 0;
    int nz = 0;
    std::vector<amrex::Real> a;
    std::vector<amrex::Real> b;
    std::vector<amrex::Real> f;
};

std::array<PythonEBNeumannData, max_hallrz_python_eb_levels> s_python_eb_neumann;
std::array<PythonEBRobinData, max_hallrz_python_eb_levels> s_python_eb_robin;

bool s_python_robin_zhi_active = false;
int s_python_robin_zhi_nrp1 = 0;
std::vector<amrex::Real> s_python_robin_zhi_a;
std::vector<amrex::Real> s_python_robin_zhi_b;
std::vector<amrex::Real> s_python_robin_zhi_f;

bool s_python_robin_rhi_active = false;
int s_python_robin_rhi_nzp1 = 0;
std::vector<amrex::Real> s_python_robin_rhi_a;
std::vector<amrex::Real> s_python_robin_rhi_b;
std::vector<amrex::Real> s_python_robin_rhi_f;

bool s_python_robin_rlo_active = false;
int s_python_robin_rlo_nzp1 = 0;
std::vector<amrex::Real> s_python_robin_rlo_a;
std::vector<amrex::Real> s_python_robin_rlo_b;
std::vector<amrex::Real> s_python_robin_rlo_f;

bool s_python_dirichlet_rlo_active = false;
int s_python_dirichlet_rlo_nzp1 = 0;
std::vector<amrex::Real> s_python_dirichlet_rlo_phi;

bool s_python_inlet_dirichlet_active = false;
int s_python_inlet_dirichlet_nrp1 = 0;
std::vector<amrex::Real> s_python_inlet_dirichlet_phi;

int
checkedPythonEBLevel (int lev, char const* api_name)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        lev >= 0 && lev < max_hallrz_python_eb_levels,
        std::string(api_name) + " currently supports only HallRZ AMR lev=0,1,2.");
    return lev;
}

int
pythonEBLevelForSolve (int amr_level)
{
    return checkedPythonEBLevel(amr_level < 0 ? 0 : amr_level, "HallRZ Python EB data");
}

template <typename T>
bool
anyPythonEBDataActive (std::array<T, max_hallrz_python_eb_levels> const& levels)
{
    return std::any_of(levels.begin(), levels.end(),
        [] (T const& level_data) { return level_data.active; });
}

std::string
normalizeBCName (std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(),
        [] (unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::replace(name.begin(), name.end(), '-', '_');
    return name;
}

HallRZPoissonSolver::RLoBoundary
parseRLoBoundary (std::string const& name)
{
    std::string const bc = normalizeBCName(name);
    if (bc == "auto" || bc.empty()) {
        return HallRZPoissonSolver::RLoBoundary::Auto;
    }
    if (bc == "axis" || bc == "none") {
        return HallRZPoissonSolver::RLoBoundary::Axis;
    }
    if (bc == "dirichlet" || bc == "pec") {
        return HallRZPoissonSolver::RLoBoundary::Dirichlet;
    }
    if (bc == "neumann" || bc == "homogeneous_neumann") {
        return HallRZPoissonSolver::RLoBoundary::Neumann;
    }
    if (bc == "robin") {
        return HallRZPoissonSolver::RLoBoundary::Robin;
    }
    WARPX_ABORT_WITH_MESSAGE(
        "warpx.hall_rz_bc_lo_r must be one of auto, axis, dirichlet, neumann, or robin.");
    return HallRZPoissonSolver::RLoBoundary::Auto;
}

HallRZPoissonSolver::EBBCMode
parseEBBCMode (std::string const& name)
{
    std::string const bc = normalizeBCName(name);
    if (bc == "neumann" || bc.empty()) {
        return HallRZPoissonSolver::EBBCMode::Neumann;
    }
    if (bc == "robin") {
        return HallRZPoissonSolver::EBBCMode::Robin;
    }
    WARPX_ABORT_WITH_MESSAGE("warpx.hall_rz_eb_bc_mode must be neumann or robin.");
    return HallRZPoissonSolver::EBBCMode::Neumann;
}

bool
isAxisRLo (amrex::Geometry const& geom)
{
    return std::abs(geom.ProbLo(0)) < 10.0 * std::numeric_limits<amrex::Real>::epsilon();
}

HallRZPoissonSolver::RLoBoundary
effectiveRLoBoundary (HallRZPoissonSolver::Params const& params,
                      amrex::Geometry const& geom)
{
    bool const on_axis = isAxisRLo(geom);
    if (on_axis) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            params.bc_lo_r == HallRZPoissonSolver::RLoBoundary::Auto ||
            params.bc_lo_r == HallRZPoissonSolver::RLoBoundary::Axis ||
            params.bc_lo_r == HallRZPoissonSolver::RLoBoundary::Neumann,
            "HallRZ r_min=0 is the RZ axis and only supports axis/homogeneous Neumann lo-r BC.");
        return HallRZPoissonSolver::RLoBoundary::Axis;
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.bc_lo_r != HallRZPoissonSolver::RLoBoundary::Axis,
        "warpx.hall_rz_bc_lo_r=axis is only valid when r_min=0.");

    if (params.bc_lo_r != HallRZPoissonSolver::RLoBoundary::Auto) {
        return params.bc_lo_r;
    }

    if (WarpX::field_boundary_lo[0] == FieldBoundaryType::PEC) {
        return HallRZPoissonSolver::RLoBoundary::Dirichlet;
    }
    if (WarpX::field_boundary_lo[0] == FieldBoundaryType::Neumann) {
        return HallRZPoissonSolver::RLoBoundary::Neumann;
    }

    WARPX_ABORT_WITH_MESSAGE(
        "HallRZ r_min>0 cannot infer lo-r Poisson BC from the lower radial field boundary. "
        "Set lower_boundary_conditions[0] to dirichlet/neumann or set warpx.hall_rz_bc_lo_r "
        "to dirichlet, neumann, or robin.");
    return HallRZPoissonSolver::RLoBoundary::Auto;
}

amrex::LinOpBCType
linOpBC (HallRZPoissonSolver::RLoBoundary bc)
{
    switch (bc) {
    case HallRZPoissonSolver::RLoBoundary::Axis:
    case HallRZPoissonSolver::RLoBoundary::Neumann:
        return amrex::LinOpBCType::Neumann;
    case HallRZPoissonSolver::RLoBoundary::Dirichlet:
        return amrex::LinOpBCType::Dirichlet;
    case HallRZPoissonSolver::RLoBoundary::Robin:
        return amrex::LinOpBCType::Robin;
    case HallRZPoissonSolver::RLoBoundary::Auto:
        break;
    }
    WARPX_ABORT_WITH_MESSAGE("Internal HallRZ error: unresolved lo-r Poisson BC.");
    return amrex::LinOpBCType::Neumann;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
overlap (amrex::Real alo, amrex::Real ahi, amrex::Real blo, amrex::Real bhi) noexcept
{
    return amrex::max(amrex::Real(0.0), amrex::min(ahi,bhi) - amrex::max(alo,blo));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool
contains (amrex::Real lo, amrex::Real hi, amrex::Real x, amrex::Real eps) noexcept
{
    return x >= lo-eps && x <= hi+eps;
}

amrex::Real
snapToCellFace (amrex::Real x, amrex::Real xlo, amrex::Real dx) noexcept
{
    return xlo + std::round((x - xlo) / dx) * dx;
}

HallDomain
makeHallDomain (HallRZPoissonSolver::Params const& params,
                amrex::Geometry const& geom) noexcept
{
    amrex::Real const* dx = geom.CellSize();
    amrex::Real const runtime_tol = amrex::max(
        params.align_tol,
        amrex::Real(1.0e-8) * amrex::max(dx[0], dx[1]));
    return HallDomain{
        geom.ProbLo(0), geom.ProbHi(0),
        geom.ProbLo(1), geom.ProbHi(1),
        snapToCellFace(params.lob, geom.ProbLo(0), dx[0]),
        snapToCellFace(params.hib, geom.ProbLo(0), dx[0]),
        snapToCellFace(params.out, geom.ProbLo(1), dx[1]),
        runtime_tol};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool
lowChannelBlockExists (HallDomain const& h) noexcept
{
    return h.lob > h.rlo + amrex::max(h.align_tol, amrex::Real(0.0));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool
hallSegmentExists (HallDomain const& h, int seg) noexcept
{
    if (seg == 0 || seg == 2) {
        return lowChannelBlockExists(h);
    }
    return seg == 1 || seg == 3;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int
hallSegment (HallDomain const& h, amrex::Real r, amrex::Real z) noexcept
{
    amrex::Real const eps = amrex::max(h.align_tol, amrex::Real(1.0e-12));
    if (lowChannelBlockExists(h) &&
        amrex::Math::abs(r - h.lob) <= eps && contains(h.zlo, h.out, z, eps))
    {
        return 0;
    }
    if (amrex::Math::abs(r - h.hib) <= eps && contains(h.zlo, h.out, z, eps)) {
        return 1;
    }
    if (lowChannelBlockExists(h) &&
        amrex::Math::abs(z - h.out) <= eps && contains(h.rlo, h.lob, r, eps))
    {
        return 2;
    }
    if (amrex::Math::abs(z - h.out) <= eps && contains(h.hib, h.rhi, r, eps)) {
        return 3;
    }
    return -1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
rectArea (Rect const& a, Rect const& b) noexcept
{
    return overlap(a.rlo,a.rhi,b.rlo,b.rhi) * overlap(a.zlo,a.zhi,b.zlo,b.zhi);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
rectRZVolume (Rect const& a, Rect const& b) noexcept
{
    amrex::Real const rlo = amrex::max(a.rlo,b.rlo);
    amrex::Real const rhi = amrex::min(a.rhi,b.rhi);
    amrex::Real const dz = overlap(a.zlo,a.zhi,b.zlo,b.zhi);
    if (rhi <= rlo || dz <= amrex::Real(0.0)) {
        return amrex::Real(0.0);
    }
    return amrex::Real(0.5) * (rhi*rhi - rlo*rlo) * dz;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
radialMomentOverlap (amrex::Real alo, amrex::Real ahi,
                     amrex::Real blo, amrex::Real bhi) noexcept
{
    amrex::Real const rlo = amrex::max(alo,blo);
    amrex::Real const rhi = amrex::min(ahi,bhi);
    if (rhi <= rlo) {
        return amrex::Real(0.0);
    }
    return amrex::Real(0.5) * (rhi*rhi - rlo*rlo);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
verticalFluidLength (HallDomain const& h, amrex::Real r, amrex::Real za, amrex::Real zb) noexcept
{
    amrex::Real len = 0.0;
    if (contains(h.lob, h.hib, r, h.align_tol)) {
        len += overlap(za, zb, h.zlo, h.out);
    }
    if (contains(h.rlo, h.rhi, r, h.align_tol)) {
        len += overlap(za, zb, h.out, h.zhi);
    }
    return len;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
verticalFluidRZArea (HallDomain const& h,
                     amrex::Real r,
                     amrex::Real za,
                     amrex::Real zb) noexcept
{
    return r * verticalFluidLength(h, r, za, zb);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
horizontalFluidLength (HallDomain const& h, amrex::Real z, amrex::Real ra, amrex::Real rb) noexcept
{
    amrex::Real len = 0.0;
    if (contains(h.zlo, h.out, z, h.align_tol)) {
        len += overlap(ra, rb, h.lob, h.hib);
    }
    if (contains(h.out, h.zhi, z, h.align_tol)) {
        len += overlap(ra, rb, h.rlo, h.rhi);
    }
    return len;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
horizontalFluidRZArea (HallDomain const& h,
                       amrex::Real z,
                       amrex::Real ra,
                       amrex::Real rb) noexcept
{
    amrex::Real area = 0.0;
    if (contains(h.zlo, h.out, z, h.align_tol)) {
        area += radialMomentOverlap(ra, rb, h.lob, h.hib);
    }
    if (contains(h.out, h.zhi, z, h.align_tol)) {
        area += radialMomentOverlap(ra, rb, h.rlo, h.rhi);
    }
    return area;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
ebLengthInNode (HallDomain const& h, int seg, Rect const& d) noexcept
{
    if (!hallSegmentExists(h, seg)) {
        return amrex::Real(0.0);
    }
    if (seg == 0) { // r=lob, zlo..out
        return contains(d.rlo, d.rhi, h.lob, h.align_tol)
            ? overlap(d.zlo,d.zhi,h.zlo,h.out) : amrex::Real(0.0);
    } else if (seg == 1) { // r=hib, zlo..out
        return contains(d.rlo, d.rhi, h.hib, h.align_tol)
            ? overlap(d.zlo,d.zhi,h.zlo,h.out) : amrex::Real(0.0);
    } else if (seg == 2) { // z=out, rlo..lob
        return contains(d.zlo, d.zhi, h.out, h.align_tol)
            ? overlap(d.rlo,d.rhi,h.rlo,h.lob) : amrex::Real(0.0);
    } else if (seg == 3) { // z=out, hib..rhi
        return contains(d.zlo, d.zhi, h.out, h.align_tol)
            ? overlap(d.rlo,d.rhi,h.hib,h.rhi) : amrex::Real(0.0);
    }
    return amrex::Real(0.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
ebRZAreaInNode (HallDomain const& h, int seg, Rect const& d) noexcept
{
    if (!hallSegmentExists(h, seg)) {
        return amrex::Real(0.0);
    }
    if (seg == 0) { // r=lob, zlo..out
        return contains(d.rlo, d.rhi, h.lob, h.align_tol)
            ? h.lob * overlap(d.zlo,d.zhi,h.zlo,h.out) : amrex::Real(0.0);
    } else if (seg == 1) { // r=hib, zlo..out
        return contains(d.rlo, d.rhi, h.hib, h.align_tol)
            ? h.hib * overlap(d.zlo,d.zhi,h.zlo,h.out) : amrex::Real(0.0);
    } else if (seg == 2) { // z=out, rlo..lob
        return contains(d.zlo, d.zhi, h.out, h.align_tol)
            ? radialMomentOverlap(d.rlo,d.rhi,h.rlo,h.lob) : amrex::Real(0.0);
    } else if (seg == 3) { // z=out, hib..rhi
        return contains(d.zlo, d.zhi, h.out, h.align_tol)
            ? radialMomentOverlap(d.rlo,d.rhi,h.hib,h.rhi) : amrex::Real(0.0);
    }
    return amrex::Real(0.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
NodeGeom
analyticNodeGeom (HallDomain const& h, int nr, int nz, int i, int j) noexcept
{
    amrex::Real const dr = (h.rhi-h.rlo) / amrex::Real(nr);
    amrex::Real const dz = (h.zhi-h.zlo) / amrex::Real(nz);
    amrex::Real const r = h.rlo + amrex::Real(i)*dr;
    amrex::Real const z = h.zlo + amrex::Real(j)*dz;
    Rect d{amrex::max(h.rlo,r-amrex::Real(0.5)*dr),
           amrex::min(h.rhi,r+amrex::Real(0.5)*dr),
           amrex::max(h.zlo,z-amrex::Real(0.5)*dz),
           amrex::min(h.zhi,z+amrex::Real(0.5)*dz)};

    Rect const channel{h.lob, h.hib, h.zlo, h.out};
    Rect const plume{h.rlo, h.rhi, h.out, h.zhi};

    NodeGeom g;
    g.v2d = rectArea(d, channel) + rectArea(d, plume);
    g.vrz = rectRZVolume(d, channel) + rectRZVolume(d, plume);
    g.arm = verticalFluidLength(h, d.rlo, d.zlo, d.zhi);
    g.arp = verticalFluidLength(h, d.rhi, d.zlo, d.zhi);
    g.azm = horizontalFluidLength(h, d.zlo, d.rlo, d.rhi);
    g.azp = horizontalFluidLength(h, d.zhi, d.rlo, d.rhi);
    g.arm_rz = verticalFluidRZArea(h, d.rlo, d.zlo, d.zhi);
    g.arp_rz = verticalFluidRZArea(h, d.rhi, d.zlo, d.zhi);
    g.azm_rz = horizontalFluidRZArea(h, d.zlo, d.rlo, d.rhi);
    g.azp_rz = horizontalFluidRZArea(h, d.zhi, d.rlo, d.rhi);
    for (int s = 0; s < 4; ++s) {
        g.aeb += ebLengthInNode(h, s, d);
        g.aeb_rz += ebRZAreaInNode(h, s, d);
    }
    return g;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
staticRho (HallRZPoissonSolver::Params const& params,
           amrex::Real zlo,
           amrex::Real zhi,
           amrex::Real r, amrex::Real z) noexcept
{
    if (params.static_rho_mode == 0) {
        return amrex::Real(0.0);
    }
    if (params.static_rho_mode == 2) {
        return params.static_rho_amp;
    }

    amrex::Real const rc = amrex::Real(0.5) * (params.lob + params.hib);
    amrex::Real const zc = amrex::Real(0.5) * (zlo + params.out);
    amrex::ignore_unused(zhi);
    amrex::Real const lr = params.static_rho_lambda_r;
    amrex::Real const lz = params.static_rho_lambda_z;
    amrex::Real const ar = (r - rc) / lr;
    amrex::Real const az = (z - zc) / lz;
    return params.static_rho_amp * std::exp(-(ar*ar + az*az));
}

std::pair<long,long>
countNaNInf (amrex::MultiFab const& mf)
{
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
    amrex::ReduceData<amrex::Long, amrex::Long> reduce_data(reduce_ops);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const& a = mf.const_array(mfi);

        reduce_ops.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
            {
                amrex::Real const v = a(i,j,k);
                return {static_cast<amrex::Long>(amrex::isnan(v)),
                        static_cast<amrex::Long>(amrex::isinf(v))};
            });
    }

    amrex::Long nnan = amrex::get<0>(reduce_data.value());
    amrex::Long ninf = amrex::get<1>(reduce_data.value());
    amrex::ParallelDescriptor::ReduceLongSum(nnan);
    amrex::ParallelDescriptor::ReduceLongSum(ninf);
    return {nnan, ninf};
}

EBFluxStats
fillEBNeumann (HallRZPoissonSolver::Params const& params,
               amrex::Geometry const& geom,
               amrex::EBFArrayBoxFactory const& eb_factory,
               amrex::MultiFab& eb_gn_cc,
               int amr_level)
{
    EBFluxStats stats;
    eb_gn_cc.setVal(amrex::Real(0.0));

    int const python_level = pythonEBLevelForSolve(amr_level);
    auto const& python_data = s_python_eb_neumann[python_level];
    bool const use_python = python_data.active;
    bool const use_scalar = params.eb_wall_flux && params.eb_g0 != amrex::Real(0.0);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!(use_python && use_scalar),
        "Python HallRZ EB Neumann array and scalar wall flux are mutually exclusive. "
        "Call hallrz.clear_eb_neumann() or set warpx.hall_rz_eb_wall_flux=0.");

    if (!use_python && !use_scalar) {
        return stats;
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(params.eb_wall_flux_segment >= -1 &&
                                     params.eb_wall_flux_segment <= 3,
                                     "warpx.hall_rz_eb_wall_flux_segment must be -1 or 0..3.");

    if (use_python) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geom.Domain().length(0) == python_data.nr &&
                                         geom.Domain().length(1) == python_data.nz,
            "Python HallRZ EB Neumann array shape does not match this AMR level cell domain.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geom.Domain().smallEnd(0) == 0 &&
                                         geom.Domain().smallEnd(1) == 0,
            "Python HallRZ EB Neumann array currently requires a zero-based cell domain.");
    }

    amrex::Gpu::DeviceVector<amrex::Real> d_python_eb_neumann;
    if (use_python) {
        d_python_eb_neumann.resize(python_data.data.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         python_data.data.begin(), python_data.data.end(),
                         d_python_eb_neumann.begin());
    }
    amrex::Real const* const python_eb =
        use_python ? d_python_eb_neumann.data() : nullptr;
    int const python_nz = python_data.nz;
    int const flux_segment = params.eb_wall_flux_segment;
    amrex::Real const scalar_g0 = params.eb_g0;

    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    HallDomain const hd = makeHallDomain(params, geom);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(flux_segment >= 0 && !hallSegmentExists(hd, flux_segment)),
        "warpx.hall_rz_eb_wall_flux_segment selects a HallRZ EB segment that does not exist "
        "for this geometry.");

    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_ops;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real,
                      amrex::Real, amrex::Real> reduce_data(reduce_ops);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (amrex::MFIter mfi(eb_gn_cc, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        if (!eb_factory.getBndryArea().ok(mfi)) {
            continue;
        }

        amrex::Box ccbx = mfi.tilebox();
        ccbx &= geom.Domain();

        auto const& gn = eb_gn_cc.array(mfi);
        auto const& ba = eb_factory.getBndryArea().const_array(mfi);
        auto const& bc = eb_factory.getBndryCent().const_array(mfi);
        auto const& bn = eb_factory.getBndryNormal().const_array(mfi);
        auto const& flg = eb_factory.getMultiEBCellFlagFab().const_array(mfi);

        reduce_ops.eval(ccbx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
            {
                if (!flg(i,j,0).isSingleValued()) {
                    return {amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0),
                            amrex::Real(0.0), amrex::Real(0.0)};
                }
                amrex::Real const area = ba(i,j,0);
                if (area <= amrex::Real(0.0)) {
                    return {amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0),
                            amrex::Real(0.0), amrex::Real(0.0)};
                }

                amrex::Real const fx = amrex::max(amrex::Real(0.0),
                    amrex::min(amrex::Real(1.0), amrex::Real(0.5) + bc(i,j,0,0)));
                amrex::Real const fy = amrex::max(amrex::Real(0.0),
                    amrex::min(amrex::Real(1.0), amrex::Real(0.5) + bc(i,j,0,1)));
                amrex::Real const rbc = plo[0] + (amrex::Real(i) + fx) * dx[0];
                amrex::Real const zbc = plo[1] + (amrex::Real(j) + fy) * dx[1];
                int const seg = hallSegment(hd, rbc, zbc);
                if (seg < 0) {
                    return {amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0),
                            amrex::Real(0.0), amrex::Real(0.0)};
                }
                if (flux_segment >= 0 && flux_segment != seg)
                {
                    return {amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0),
                            amrex::Real(0.0), amrex::Real(0.0)};
                }

                amrex::Real const geb = use_python
                    ? python_eb[i * python_nz + j]
                    : scalar_g0;
                gn(i,j,0,0) = geb;

                amrex::Real const len_x = dx[0]*bn(i,j,0,1);
                amrex::Real const len_y = dx[1]*bn(i,j,0,0);
                amrex::Real const len_scale = std::sqrt(len_x*len_x + len_y*len_y);
                amrex::Real q = -geb * area * len_scale;

                // Match mlebndfvlap_eb_inhom_neu_inject_2d: two-point
                // quadrature of the RZ radius along the EB segment.
                amrex::Real const tx = -bn(i,j,0,1);
                amrex::Real const ty =  bn(i,j,0,0);
                amrex::ignore_unused(ty);
                amrex::Real const half_len = amrex::Real(0.5) * area * len_scale;
                amrex::Real const gauss_absc = amrex::Real(0.57735026918962576451);
                amrex::Real const s = half_len * gauss_absc;
                amrex::Real r_avg = amrex::Real(0.0);
                for (int iq = 0; iq < 2; ++iq) {
                    amrex::Real const sign = (iq == 0) ? amrex::Real(-1.0) : amrex::Real(1.0);
                    amrex::Real fxq = fx + sign * s * tx / dx[0];
                    fxq = amrex::max(amrex::Real(0.0), amrex::min(amrex::Real(1.0), fxq));
                    amrex::Real const rq = amrex::max(plo[0] + (amrex::Real(i) + fxq) * dx[0],
                                                      amrex::Real(0.25) * dx[0]);
                    r_avg += amrex::Real(0.5) * rq;
                }
                q *= r_avg;

                return {q,
                        (seg == 0) ? q : amrex::Real(0.0),
                        (seg == 1) ? q : amrex::Real(0.0),
                        (seg == 2) ? q : amrex::Real(0.0),
                        (seg == 3) ? q : amrex::Real(0.0)};
            });
    }

    stats.total = amrex::get<0>(reduce_data.value());
    stats.segment[0] = amrex::get<1>(reduce_data.value());
    stats.segment[1] = amrex::get<2>(reduce_data.value());
    stats.segment[2] = amrex::get<3>(reduce_data.value());
    stats.segment[3] = amrex::get<4>(reduce_data.value());
    amrex::ParallelDescriptor::ReduceRealSum(stats.total);
    for (auto& s : stats.segment) {
        amrex::ParallelDescriptor::ReduceRealSum(s);
    }
    return stats;
}

EBRobinStats
fillEBRobin (HallRZPoissonSolver::Params const& params,
             amrex::Geometry const& geom,
             amrex::EBFArrayBoxFactory const& eb_factory,
             amrex::MultiFab& eb_a_cc,
             amrex::MultiFab& eb_b_cc,
             amrex::MultiFab& eb_f_cc,
             int amr_level)
{
    EBRobinStats stats;
    eb_a_cc.setVal(amrex::Real(0.0));
    eb_b_cc.setVal(amrex::Real(1.0));
    eb_f_cc.setVal(amrex::Real(0.0));

    int const python_level = pythonEBLevelForSolve(amr_level);
    auto const& python_data = s_python_eb_robin[python_level];

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(python_data.active,
        "warpx.hall_rz_eb_bc_mode=robin requires hallrz.set_eb_robin(a,b,f,lev=amr_level).");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!anyPythonEBDataActive(s_python_eb_neumann),
        "warpx.hall_rz_eb_bc_mode=robin is mutually exclusive with hallrz.set_eb_neumann().");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!params.eb_wall_flux && params.eb_g0 == amrex::Real(0.0),
        "warpx.hall_rz_eb_bc_mode=robin is mutually exclusive with scalar HallRZ EB wall flux.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(params.eb_wall_flux_segment == -1,
        "warpx.hall_rz_eb_wall_flux_segment is only valid for HallRZ EB Neumann mode.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geom.Domain().length(0) == python_data.nr &&
                                     geom.Domain().length(1) == python_data.nz,
        "Python HallRZ EB Robin array shape does not match this AMR level cell domain.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geom.Domain().smallEnd(0) == 0 &&
                                     geom.Domain().smallEnd(1) == 0,
        "Python HallRZ EB Robin arrays currently require a zero-based cell domain.");

    amrex::Gpu::DeviceVector<amrex::Real> d_a(python_data.a.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_b(python_data.b.size());
    amrex::Gpu::DeviceVector<amrex::Real> d_f(python_data.f.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     python_data.a.begin(), python_data.a.end(),
                     d_a.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     python_data.b.begin(), python_data.b.end(),
                     d_b.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     python_data.f.begin(), python_data.f.end(),
                     d_f.begin());

    amrex::Real const* const robin_a = d_a.data();
    amrex::Real const* const robin_b = d_b.data();
    amrex::Real const* const robin_f = d_f.data();
    int const robin_nz = python_data.nz;

    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    HallDomain const hd = makeHallDomain(params, geom);

    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum> reduce_ops;
    amrex::ReduceData<amrex::Long, amrex::Long, amrex::Long,
                      amrex::Long, amrex::Long, amrex::Long,
                      amrex::Long> reduce_data(reduce_ops);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (amrex::MFIter mfi(eb_a_cc, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        if (!eb_factory.getBndryArea().ok(mfi)) {
            continue;
        }

        amrex::Box ccbx = mfi.tilebox();
        ccbx &= geom.Domain();

        auto const& ea = eb_a_cc.array(mfi);
        auto const& eb = eb_b_cc.array(mfi);
        auto const& ef = eb_f_cc.array(mfi);
        auto const& ba = eb_factory.getBndryArea().const_array(mfi);
        auto const& bc = eb_factory.getBndryCent().const_array(mfi);
        auto const& flg = eb_factory.getMultiEBCellFlagFab().const_array(mfi);

        reduce_ops.eval(ccbx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
            {
                if (!flg(i,j,0).isSingleValued()) {
                    return {0, 0, 0, 0, 0, 0, 0};
                }
                amrex::Real const area = ba(i,j,0);
                if (area <= amrex::Real(0.0)) {
                    return {0, 0, 0, 0, 0, 0, 0};
                }

                amrex::Real const fx = amrex::max(amrex::Real(0.0),
                    amrex::min(amrex::Real(1.0), amrex::Real(0.5) + bc(i,j,0,0)));
                amrex::Real const fy = amrex::max(amrex::Real(0.0),
                    amrex::min(amrex::Real(1.0), amrex::Real(0.5) + bc(i,j,0,1)));
                amrex::Real const rbc = plo[0] + (amrex::Real(i) + fx) * dx[0];
                amrex::Real const zbc = plo[1] + (amrex::Real(j) + fy) * dx[1];
                int const seg = hallSegment(hd, rbc, zbc);
                if (seg < 0) {
                    return {0, 0, 0, 0, 0, 0, 0};
                }

                amrex::Long const offset = static_cast<amrex::Long>(i) *
                                           static_cast<amrex::Long>(robin_nz) +
                                           static_cast<amrex::Long>(j);
                amrex::Real const a = robin_a[offset];
                amrex::Real const b = robin_b[offset];
                amrex::Real const f = robin_f[offset];
                bool const nonfinite = !amrex::Math::isfinite(a) ||
                                       !amrex::Math::isfinite(b) ||
                                       !amrex::Math::isfinite(f);
                bool const degenerate = amrex::Math::abs(a) + amrex::Math::abs(b) <=
                                        amrex::Real(0.0);
                if (!nonfinite && !degenerate) {
                    ea(i,j,0,0) = a;
                    eb(i,j,0,0) = b;
                    ef(i,j,0,0) = f;
                }

                return {1,
                        static_cast<amrex::Long>(nonfinite),
                        static_cast<amrex::Long>(degenerate),
                        (seg == 0) ? 1 : 0,
                        (seg == 1) ? 1 : 0,
                        (seg == 2) ? 1 : 0,
                        (seg == 3) ? 1 : 0};
            });
    }

    stats.active = amrex::get<0>(reduce_data.value());
    stats.invalid_nonfinite = amrex::get<1>(reduce_data.value());
    stats.invalid_degenerate = amrex::get<2>(reduce_data.value());
    stats.segment[0] = amrex::get<3>(reduce_data.value());
    stats.segment[1] = amrex::get<4>(reduce_data.value());
    stats.segment[2] = amrex::get<5>(reduce_data.value());
    stats.segment[3] = amrex::get<6>(reduce_data.value());
    amrex::ParallelDescriptor::ReduceLongSum(stats.active);
    amrex::ParallelDescriptor::ReduceLongSum(stats.invalid_nonfinite);
    amrex::ParallelDescriptor::ReduceLongSum(stats.invalid_degenerate);
    for (auto& s : stats.segment) {
        amrex::ParallelDescriptor::ReduceLongSum(s);
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(stats.invalid_nonfinite == 0,
        "HallRZ EB Robin data contain NaN or Inf on active cut cells.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(stats.invalid_degenerate == 0,
        "HallRZ EB Robin data have a=b=0 on active cut cells.");
    return stats;
}

amrex::Real
sumOperatorRZWeightedOwner (amrex::MultiFab const& mf,
                            amrex::Geometry const& geom,
                            HallDomain const& hd,
                            int nr0,
                            int nz0)
{
    auto owner_mask = amrex::OwnerMask(mf, geom.periodicity());
    amrex::Real const dr = geom.CellSize(0);
    amrex::Real const rlo = geom.ProbLo(0);
    int const ilo = amrex::surroundingNodes(geom.Domain()).smallEnd(0);

    amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_ops);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const& a = mf.const_array(mfi);
        auto const& om = owner_mask->const_array(mfi);

        reduce_ops.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
            {
                if (om(i,j,0) == 0) {
                    return {amrex::Real(0.0)};
                }
                NodeGeom const ng = analyticNodeGeom(hd, nr0, nz0, i, j);
                amrex::Real const rnode = rlo + amrex::Real(i) * dr;
                amrex::Real const vfac = (rlo == amrex::Real(0.0) && i == ilo)
                    ? amrex::Real(0.25) * dr
                    : amrex::max(rnode, amrex::Real(0.25) * dr);
                return {ng.v2d * vfac * a(i,j,k)};
            });
    }

    amrex::Real sum = amrex::get<0>(reduce_data.value());
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    return sum;
}

} // namespace

HallRZPoissonSolver::Params
HallRZPoissonSolver::ReadParameters ()
{
    amrex::ParmParse const pp_warpx("warpx");

    Params params;
    pp_warpx.query("hall_rz_enable", params.enabled);
    pp_warpx.query("hall_rz_verbose", params.verbose);
    pp_warpx.query("hall_rz_diag_interval", params.diag_interval);
    pp_warpx.query("hall_rz_align_tol", params.align_tol);
    pp_warpx.query("hall_rz_max_iter", params.max_iter);
    pp_warpx.query("hall_rz_rel_tol", params.rel_tol);
    pp_warpx.query("hall_rz_abs_tol", params.abs_tol);
    pp_warpx.query("hall_rz_max_coarsening_level", params.max_coarsening_level);
    pp_warpx.query("hall_rz_static_rho_test", params.static_rho_test);
    pp_warpx.query("hall_rz_static_rho_mode", params.static_rho_mode);
    pp_warpx.query("hall_rz_static_rho_amp", params.static_rho_amp);
    pp_warpx.query("hall_rz_static_rho_lambda_r", params.static_rho_lambda_r);
    pp_warpx.query("hall_rz_static_rho_lambda_z", params.static_rho_lambda_z);
    std::string eb_bc_mode = "neumann";
    pp_warpx.query("hall_rz_eb_bc_mode", eb_bc_mode);
    params.eb_bc_mode = parseEBBCMode(eb_bc_mode);
    bool have_eb_g0 = pp_warpx.query("hall_rz_eb_g0", params.eb_g0);
    pp_warpx.query("hall_rz_eb_wall_flux", params.eb_wall_flux);
    pp_warpx.query("hall_rz_eb_wall_flux_segment", params.eb_wall_flux_segment);
    std::string bc_lo_r = "auto";
    pp_warpx.query("hall_rz_bc_lo_r", bc_lo_r);
    params.bc_lo_r = parseRLoBoundary(bc_lo_r);
    pp_warpx.query("hall_rz_potential_lo_r", params.potential_lo_r);
    pp_warpx.query("hall_rz_robin_lo_r_a", params.robin_lo_r_a);
    pp_warpx.query("hall_rz_robin_lo_r_b", params.robin_lo_r_b);
    pp_warpx.query("hall_rz_robin_lo_r_f", params.robin_lo_r_f);
    pp_warpx.query("hall_rz_robin_rlo_a", params.robin_lo_r_a);
    pp_warpx.query("hall_rz_robin_rlo_b", params.robin_lo_r_b);
    pp_warpx.query("hall_rz_robin_rlo_f", params.robin_lo_r_f);
    if (have_eb_g0 && params.eb_g0 != amrex::Real(0.0)) {
        params.eb_wall_flux = true;
    }

    if (params.enabled) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pp_warpx.query("hall_rz_lob", params.lob),
                                         "warpx.hall_rz_lob is required when hall_rz_enable=1.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pp_warpx.query("hall_rz_hib", params.hib),
                                         "warpx.hall_rz_hib is required when hall_rz_enable=1.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pp_warpx.query("hall_rz_out", params.out),
                                         "warpx.hall_rz_out is required when hall_rz_enable=1.");
    }

    return params;
}

bool
HallRZPoissonSolver::Enabled ()
{
    amrex::ParmParse const pp_warpx("warpx");
    bool enabled = false;
    pp_warpx.query("hall_rz_enable", enabled);
    return enabled;
}

void
HallRZPoissonSolver::SetPythonEBNeumann (int nr, int nz,
                                         std::vector<amrex::Real> data,
                                         int lev)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nr > 0 && nz > 0,
        "HallRZ Python EB Neumann array dimensions must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(data.size() == static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz),
        "HallRZ Python EB Neumann array data size does not match (nr,nz).");

    auto& level_data = s_python_eb_neumann[checkedPythonEBLevel(lev, "hallrz.set_eb_neumann")];
    level_data.nr = nr;
    level_data.nz = nz;
    level_data.data = std::move(data);
    level_data.active = true;
}

void
HallRZPoissonSolver::ClearPythonEBNeumann (int lev)
{
    auto clear_level = [] (PythonEBNeumannData& level_data)
    {
        level_data.active = false;
        level_data.nr = 0;
        level_data.nz = 0;
        level_data.data.clear();
    };

    if (lev < 0) {
        for (auto& level_data : s_python_eb_neumann) {
            clear_level(level_data);
        }
    } else {
        clear_level(s_python_eb_neumann[checkedPythonEBLevel(lev, "hallrz.clear_eb_neumann")]);
    }
}

bool
HallRZPoissonSolver::HasPythonEBNeumann (int lev)
{
    if (lev < 0) {
        return anyPythonEBDataActive(s_python_eb_neumann);
    }
    return s_python_eb_neumann[checkedPythonEBLevel(lev, "hallrz.has_eb_neumann")].active;
}

void
HallRZPoissonSolver::SetPythonEBRobin (int nr, int nz,
                                       std::vector<amrex::Real> a,
                                       std::vector<amrex::Real> b,
                                       std::vector<amrex::Real> f,
                                       int lev)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nr > 0 && nz > 0,
        "HallRZ Python EB Robin array dimensions must be positive.");
    auto const n = static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(a.size() == n && b.size() == n && f.size() == n,
        "HallRZ Python EB Robin array data sizes do not match (nr,nz).");

    auto& level_data = s_python_eb_robin[checkedPythonEBLevel(lev, "hallrz.set_eb_robin")];
    level_data.nr = nr;
    level_data.nz = nz;
    level_data.a = std::move(a);
    level_data.b = std::move(b);
    level_data.f = std::move(f);
    level_data.active = true;
}

void
HallRZPoissonSolver::ClearPythonEBRobin (int lev)
{
    auto clear_level = [] (PythonEBRobinData& level_data)
    {
        level_data.active = false;
        level_data.nr = 0;
        level_data.nz = 0;
        level_data.a.clear();
        level_data.b.clear();
        level_data.f.clear();
    };

    if (lev < 0) {
        for (auto& level_data : s_python_eb_robin) {
            clear_level(level_data);
        }
    } else {
        clear_level(s_python_eb_robin[checkedPythonEBLevel(lev, "hallrz.clear_eb_robin")]);
    }
}

bool
HallRZPoissonSolver::HasPythonEBRobin (int lev)
{
    if (lev < 0) {
        return anyPythonEBDataActive(s_python_eb_robin);
    }
    return s_python_eb_robin[checkedPythonEBLevel(lev, "hallrz.has_eb_robin")].active;
}

void
HallRZPoissonSolver::SetPythonRobinZHi (int nrp1, std::vector<amrex::Real> a,
                                        std::vector<amrex::Real> b,
                                        std::vector<amrex::Real> f)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nrp1 > 0,
        "HallRZ Python z-hi Robin array length must be positive.");
    auto const n = static_cast<std::size_t>(nrp1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(a.size() == n && b.size() == n && f.size() == n,
        "HallRZ Python z-hi Robin arrays must all have length Nr+1.");

    s_python_robin_zhi_nrp1 = nrp1;
    s_python_robin_zhi_a = std::move(a);
    s_python_robin_zhi_b = std::move(b);
    s_python_robin_zhi_f = std::move(f);
    s_python_robin_zhi_active = true;
}

void
HallRZPoissonSolver::ClearPythonRobinZHi ()
{
    s_python_robin_zhi_active = false;
    s_python_robin_zhi_nrp1 = 0;
    s_python_robin_zhi_a.clear();
    s_python_robin_zhi_b.clear();
    s_python_robin_zhi_f.clear();
}

bool
HallRZPoissonSolver::HasPythonRobinZHi ()
{
    return s_python_robin_zhi_active;
}

void
HallRZPoissonSolver::SetPythonRobinRHi (int nzp1, std::vector<amrex::Real> a,
                                        std::vector<amrex::Real> b,
                                        std::vector<amrex::Real> f)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nzp1 > 0,
        "HallRZ Python r-hi Robin array length must be positive.");
    auto const n = static_cast<std::size_t>(nzp1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(a.size() == n && b.size() == n && f.size() == n,
        "HallRZ Python r-hi Robin arrays must all have length Nz+1.");

    s_python_robin_rhi_nzp1 = nzp1;
    s_python_robin_rhi_a = std::move(a);
    s_python_robin_rhi_b = std::move(b);
    s_python_robin_rhi_f = std::move(f);
    s_python_robin_rhi_active = true;
}

void
HallRZPoissonSolver::ClearPythonRobinRHi ()
{
    s_python_robin_rhi_active = false;
    s_python_robin_rhi_nzp1 = 0;
    s_python_robin_rhi_a.clear();
    s_python_robin_rhi_b.clear();
    s_python_robin_rhi_f.clear();
}

bool
HallRZPoissonSolver::HasPythonRobinRHi ()
{
    return s_python_robin_rhi_active;
}

void
HallRZPoissonSolver::SetPythonRobinRLo (int nzp1, std::vector<amrex::Real> a,
                                        std::vector<amrex::Real> b,
                                        std::vector<amrex::Real> f)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nzp1 > 0,
        "HallRZ Python r-lo Robin array length must be positive.");
    auto const n = static_cast<std::size_t>(nzp1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(a.size() == n && b.size() == n && f.size() == n,
        "HallRZ Python r-lo Robin arrays must all have length Nz+1.");

    s_python_robin_rlo_nzp1 = nzp1;
    s_python_robin_rlo_a = std::move(a);
    s_python_robin_rlo_b = std::move(b);
    s_python_robin_rlo_f = std::move(f);
    s_python_robin_rlo_active = true;
}

void
HallRZPoissonSolver::ClearPythonRobinRLo ()
{
    s_python_robin_rlo_active = false;
    s_python_robin_rlo_nzp1 = 0;
    s_python_robin_rlo_a.clear();
    s_python_robin_rlo_b.clear();
    s_python_robin_rlo_f.clear();
}

bool
HallRZPoissonSolver::HasPythonRobinRLo ()
{
    return s_python_robin_rlo_active;
}

void
HallRZPoissonSolver::SetPythonDirichletRLo (int nzp1, std::vector<amrex::Real> phi)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nzp1 > 0,
        "HallRZ Python r-lo Dirichlet array length must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(phi.size() == static_cast<std::size_t>(nzp1),
        "HallRZ Python r-lo Dirichlet array must have length Nz+1.");

    s_python_dirichlet_rlo_nzp1 = nzp1;
    s_python_dirichlet_rlo_phi = std::move(phi);
    s_python_dirichlet_rlo_active = true;
}

void
HallRZPoissonSolver::ClearPythonDirichletRLo ()
{
    s_python_dirichlet_rlo_active = false;
    s_python_dirichlet_rlo_nzp1 = 0;
    s_python_dirichlet_rlo_phi.clear();
}

bool
HallRZPoissonSolver::HasPythonDirichletRLo ()
{
    return s_python_dirichlet_rlo_active;
}

void
HallRZPoissonSolver::SetPythonInletDirichlet (int nrp1, std::vector<amrex::Real> phi)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nrp1 > 0,
        "HallRZ Python inlet Dirichlet array length must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(phi.size() == static_cast<std::size_t>(nrp1),
        "HallRZ Python inlet Dirichlet array must have length Nr+1.");

    s_python_inlet_dirichlet_nrp1 = nrp1;
    s_python_inlet_dirichlet_phi = std::move(phi);
    s_python_inlet_dirichlet_active = true;
}

void
HallRZPoissonSolver::ClearPythonInletDirichlet ()
{
    s_python_inlet_dirichlet_active = false;
    s_python_inlet_dirichlet_nrp1 = 0;
    s_python_inlet_dirichlet_phi.clear();
}

bool
HallRZPoissonSolver::HasPythonInletDirichlet ()
{
    return s_python_inlet_dirichlet_active;
}

void
HallRZPoissonSolver::ValidateGeometry (Params const& params, amrex::Geometry const& geom)
{
    if (!params.enabled) { return; }

#if !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("HallRZPoissonSolver requires a RZ build.");
#else
    amrex::Real const rlo = geom.ProbLo(0);
    amrex::Real const rhi = geom.ProbHi(0);
    amrex::Real const zlo = geom.ProbLo(1);
    amrex::Real const zhi = geom.ProbHi(1);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rlo >= amrex::Real(0.0),
                                     "HallRZ requires r_min >= 0.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        rlo <= params.lob + params.align_tol &&
        params.lob < params.hib && params.hib < rhi,
        "HallRZ requires r_min <= lob < hib < r_max.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(zlo < params.out && params.out < zhi,
                                     "HallRZ requires z_min < out < z_max.");
    if (isAxisRLo(geom)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            params.bc_lo_r == RLoBoundary::Auto ||
            params.bc_lo_r == RLoBoundary::Axis ||
            params.bc_lo_r == RLoBoundary::Neumann,
            "HallRZ r_min=0 is the RZ axis and only supports axis/homogeneous Neumann lo-r BC.");
    } else {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(params.bc_lo_r != RLoBoundary::Axis,
                                         "warpx.hall_rz_bc_lo_r=axis is only valid when r_min=0.");
    }

    amrex::Real const* dx = geom.CellSize();
    auto aligned = [&params] (amrex::Real x, amrex::Real xlo, amrex::Real h) noexcept {
        amrex::Real const idx = (x - xlo) / h;
        return std::abs(idx - std::round(idx)) < params.align_tol;
    };
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(aligned(params.lob, rlo, dx[0]) &&
                                     aligned(params.hib, rlo, dx[0]) &&
                                     aligned(params.out, zlo, dx[1]),
                                     "HallRZ lob/hib/out must lie exactly on cell faces.");
#endif
}

std::unique_ptr<amrex::MultiFab>
HallRZPoissonSolver::MakeNodalGeometry (Params const& params,
                                        amrex::Geometry const& geom,
                                        amrex::BoxArray const& cell_grids,
                                        amrex::DistributionMapping const& dmap,
                                        int nr_cells,
                                        int nz_cells,
                                        int ngrow)
{
    ValidateGeometry(params, geom);

    amrex::BoxArray nodal_grids = amrex::convert(cell_grids, amrex::IntVect::TheNodeVector());
    auto eb_node_geom = std::make_unique<amrex::MultiFab>(nodal_grids, dmap, 6, ngrow);
    eb_node_geom->setVal(amrex::Real(0.0));

    int const nr = (nr_cells > 0) ? nr_cells : geom.Domain().length(0);
    int const nz = (nz_cells > 0) ? nz_cells : geom.Domain().length(1);
    HallDomain const h = makeHallDomain(params, geom);

    for (amrex::MFIter mfi(*eb_node_geom, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const& g = eb_node_geom->array(mfi);

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                NodeGeom const ng = analyticNodeGeom(h, nr, nz, i, j);
                g(i,j,k,0) = ng.v2d;
                g(i,j,k,1) = ng.arm;
                g(i,j,k,2) = ng.arp;
                g(i,j,k,3) = ng.azm;
                g(i,j,k,4) = ng.azp;
                g(i,j,k,5) = ng.aeb;
            });
    }
    eb_node_geom->FillBoundary(geom.periodicity());
    return eb_node_geom;
}

void
HallRZPoissonSolver::ComputePhiAndE (amrex::MultiFab const& rho,
                                     amrex::MultiFab& phi,
                                     amrex::Array<amrex::MultiFab*,3> const& Efield,
                                     amrex::Geometry const& geom,
                                     amrex::BoxArray const& cell_grids,
                                     amrex::DistributionMapping const& dmap,
                                     amrex::EBFArrayBoxFactory const& eb_factory,
                                     int const max_coarsening_level_override,
                                     int const amr_level)
{
    Params const params = ReadParameters();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(params.enabled,
                                     "HallRZPoissonSolver::ComputePhiAndE requires warpx.hall_rz_enable=1.");
    ValidateGeometry(params, geom);

    amrex::MultiFab rho_before(rho.boxArray(), rho.DistributionMap(), 1, 0);
    amrex::MultiFab::Copy(rho_before, rho, 0, 0, 1, 0);

    int const requested_mcl = amrex::max(
        0,
        (max_coarsening_level_override >= 0)
            ? max_coarsening_level_override
            : params.max_coarsening_level);
    amrex::LPInfo info;
    info.setMaxCoarseningLevel(requested_mcl);

    amrex::Vector<amrex::Geometry> pgeom{geom};
    amrex::Vector<amrex::BoxArray> pgrids{cell_grids};
    amrex::Vector<amrex::DistributionMapping> pdmap{dmap};
    amrex::Vector<amrex::EBFArrayBoxFactory const*> pfac{&eb_factory};

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!(s_python_robin_rlo_active && s_python_dirichlet_rlo_active),
        "Python HallRZ r-lo Robin and Dirichlet arrays are mutually exclusive.");
    RLoBoundary rlo_bc = RLoBoundary::Auto;
    if (!isAxisRLo(geom) && params.bc_lo_r == RLoBoundary::Auto && s_python_robin_rlo_active) {
        rlo_bc = RLoBoundary::Robin;
    } else if (!isAxisRLo(geom) && params.bc_lo_r == RLoBoundary::Auto && s_python_dirichlet_rlo_active) {
        rlo_bc = RLoBoundary::Dirichlet;
    } else {
        rlo_bc = effectiveRLoBoundary(params, geom);
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!s_python_robin_rlo_active || rlo_bc == RLoBoundary::Robin,
        "Python HallRZ r-lo Robin arrays are active but lo-r BC is not Robin.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!s_python_dirichlet_rlo_active || rlo_bc == RLoBoundary::Dirichlet,
        "Python HallRZ r-lo Dirichlet arrays are active but lo-r BC is not Dirichlet.");

    amrex::MLEBNodeFVLaplacian linop(pgeom, pgrids, pdmap, info, pfac);
    amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> lobc{
        AMREX_D_DECL(linOpBC(rlo_bc),
                     amrex::LinOpBCType::Dirichlet,
                     amrex::LinOpBCType::Dirichlet)};
    amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> hibc{
        AMREX_D_DECL(amrex::LinOpBCType::Robin,
                     amrex::LinOpBCType::Robin,
                     amrex::LinOpBCType::Dirichlet)};
    linop.setDomainBC(lobc, hibc);
    linop.setSigma({AMREX_D_DECL(amrex::Real(1.0), amrex::Real(1.0), amrex::Real(1.0))});
    linop.setRZ(true);

    amrex::BoxArray const nodal_grids = amrex::convert(cell_grids, amrex::IntVect::TheNodeVector());
    amrex::MultiFab rhs(nodal_grids, dmap, 1, 0, amrex::MFInfo{}, eb_factory);
    amrex::MultiFab neumann_bc(nodal_grids, dmap, 2*AMREX_SPACEDIM, 0, amrex::MFInfo{}, eb_factory);
    amrex::MultiFab robin_a(nodal_grids, dmap, 2*AMREX_SPACEDIM, 0, amrex::MFInfo{}, eb_factory);
    amrex::MultiFab robin_b(nodal_grids, dmap, 2*AMREX_SPACEDIM, 0, amrex::MFInfo{}, eb_factory);
    amrex::MultiFab robin_f(nodal_grids, dmap, 2*AMREX_SPACEDIM, 0, amrex::MFInfo{}, eb_factory);
    amrex::MultiFab eb_gn_cc(cell_grids, dmap, 1, 0, amrex::MFInfo{}, eb_factory);
    amrex::MultiFab eb_a_cc(cell_grids, dmap, 1, 0, amrex::MFInfo{}, eb_factory);
    amrex::MultiFab eb_b_cc(cell_grids, dmap, 1, 0, amrex::MFInfo{}, eb_factory);
    amrex::MultiFab eb_f_cc(cell_grids, dmap, 1, 0, amrex::MFInfo{}, eb_factory);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho.ixType() == rhs.ixType(),
        "HallRZPoissonSolver expects rho_fp to have the same nodal centering as phi/rhs.");

    rhs.setVal(amrex::Real(0.0));
    neumann_bc.setVal(amrex::Real(0.0));
    robin_a.setVal(amrex::Real(0.0));
    robin_b.setVal(amrex::Real(0.0));
    robin_f.setVal(amrex::Real(0.0));
    eb_gn_cc.setVal(amrex::Real(0.0));
    eb_a_cc.setVal(amrex::Real(0.0));
    eb_b_cc.setVal(amrex::Real(1.0));
    eb_f_cc.setVal(amrex::Real(0.0));
    // Keep phi from the previous PIC step as the MLMG initial guess.

    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    amrex::Box const nd = amrex::surroundingNodes(geom.Domain());
    HallDomain const hd = makeHallDomain(params, geom);
    int const nr0 = geom.Domain().length(0);
    int const nz0 = geom.Domain().length(1);
    if (s_python_robin_zhi_active) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(s_python_robin_zhi_nrp1 == nr0 + 1,
            "Python HallRZ z-hi Robin arrays must have shape (Nr+1).");
        for (int i = 0; i <= nr0; ++i) {
            amrex::Real const a = s_python_robin_zhi_a[static_cast<std::size_t>(i)];
            amrex::Real const b = s_python_robin_zhi_b[static_cast<std::size_t>(i)];
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::Math::abs(a) + amrex::Math::abs(b) >
                                             amrex::Real(0.0),
                "Invalid Python HallRZ z-hi Robin data: a and b cannot both be zero.");
        }
    }
    if (s_python_robin_rhi_active) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(s_python_robin_rhi_nzp1 == nz0 + 1,
            "Python HallRZ r-hi Robin arrays must have shape (Nz+1).");
        for (int j = 0; j <= nz0; ++j) {
            amrex::Real const z = plo[1] + amrex::Real(j) * dx[1];
            bool const active_rhi = contains(hd.out, geom.ProbHi(1), z, hd.align_tol);
            if (active_rhi) {
                amrex::Real const a = s_python_robin_rhi_a[static_cast<std::size_t>(j)];
                amrex::Real const b = s_python_robin_rhi_b[static_cast<std::size_t>(j)];
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::Math::abs(a) + amrex::Math::abs(b) >
                                                 amrex::Real(0.0),
                    "Invalid Python HallRZ r-hi Robin data: a and b cannot both be zero "
                    "on an active node.");
            }
        }
    }
    if (s_python_robin_rlo_active) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(s_python_robin_rlo_nzp1 == nz0 + 1,
            "Python HallRZ r-lo Robin arrays must have shape (Nz+1).");
        for (int j = 0; j <= nz0; ++j) {
            amrex::Real const z = plo[1] + amrex::Real(j) * dx[1];
            bool const active_rlo = contains(hd.out, geom.ProbHi(1), z, hd.align_tol);
            if (active_rlo) {
                amrex::Real const a = s_python_robin_rlo_a[static_cast<std::size_t>(j)];
                amrex::Real const b = s_python_robin_rlo_b[static_cast<std::size_t>(j)];
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::Math::abs(a) + amrex::Math::abs(b) >
                                                 amrex::Real(0.0),
                    "Invalid Python HallRZ r-lo Robin data: a and b cannot both be zero "
                    "on an active node.");
            }
        }
    }
    if (s_python_dirichlet_rlo_active) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(s_python_dirichlet_rlo_nzp1 == nz0 + 1,
            "Python HallRZ r-lo Dirichlet array must have shape (Nz+1).");
    }
    if (rlo_bc == RLoBoundary::Robin) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::Math::abs(params.robin_lo_r_a) +
                                         amrex::Math::abs(params.robin_lo_r_b) >
                                         amrex::Real(0.0),
            "Invalid scalar HallRZ r-lo Robin data: a and b cannot both be zero.");
    }
    if (s_python_inlet_dirichlet_active) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(s_python_inlet_dirichlet_nrp1 == nr0 + 1,
            "Python HallRZ inlet Dirichlet array must have shape (Nr+1).");
    }

    amrex::Gpu::DeviceVector<amrex::Real> d_robin_zhi_a;
    amrex::Gpu::DeviceVector<amrex::Real> d_robin_zhi_b;
    amrex::Gpu::DeviceVector<amrex::Real> d_robin_zhi_f;
    amrex::Gpu::DeviceVector<amrex::Real> d_robin_rhi_a;
    amrex::Gpu::DeviceVector<amrex::Real> d_robin_rhi_b;
    amrex::Gpu::DeviceVector<amrex::Real> d_robin_rhi_f;
    amrex::Gpu::DeviceVector<amrex::Real> d_robin_rlo_a;
    amrex::Gpu::DeviceVector<amrex::Real> d_robin_rlo_b;
    amrex::Gpu::DeviceVector<amrex::Real> d_robin_rlo_f;
    amrex::Gpu::DeviceVector<amrex::Real> d_dirichlet_rlo_phi;
    amrex::Gpu::DeviceVector<amrex::Real> d_inlet_phi;

    auto copy_to_device = [] (std::vector<amrex::Real> const& src,
                              amrex::Gpu::DeviceVector<amrex::Real>& dst)
    {
        dst.resize(src.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, src.begin(), src.end(), dst.begin());
    };

    bool const use_robin_zhi = s_python_robin_zhi_active;
    bool const use_robin_rhi = s_python_robin_rhi_active;
    bool const use_robin_rlo = s_python_robin_rlo_active;
    bool const use_dirichlet_rlo = s_python_dirichlet_rlo_active;
    bool const use_inlet_dirichlet = s_python_inlet_dirichlet_active;
    if (use_robin_zhi) {
        copy_to_device(s_python_robin_zhi_a, d_robin_zhi_a);
        copy_to_device(s_python_robin_zhi_b, d_robin_zhi_b);
        copy_to_device(s_python_robin_zhi_f, d_robin_zhi_f);
    }
    if (use_robin_rhi) {
        copy_to_device(s_python_robin_rhi_a, d_robin_rhi_a);
        copy_to_device(s_python_robin_rhi_b, d_robin_rhi_b);
        copy_to_device(s_python_robin_rhi_f, d_robin_rhi_f);
    }
    if (use_robin_rlo) {
        copy_to_device(s_python_robin_rlo_a, d_robin_rlo_a);
        copy_to_device(s_python_robin_rlo_b, d_robin_rlo_b);
        copy_to_device(s_python_robin_rlo_f, d_robin_rlo_f);
    }
    if (use_dirichlet_rlo) {
        copy_to_device(s_python_dirichlet_rlo_phi, d_dirichlet_rlo_phi);
    }
    if (use_inlet_dirichlet) {
        copy_to_device(s_python_inlet_dirichlet_phi, d_inlet_phi);
    }

    amrex::Real const* const robin_zhi_a = use_robin_zhi ? d_robin_zhi_a.data() : nullptr;
    amrex::Real const* const robin_zhi_b = use_robin_zhi ? d_robin_zhi_b.data() : nullptr;
    amrex::Real const* const robin_zhi_f = use_robin_zhi ? d_robin_zhi_f.data() : nullptr;
    amrex::Real const* const robin_rhi_a = use_robin_rhi ? d_robin_rhi_a.data() : nullptr;
    amrex::Real const* const robin_rhi_b = use_robin_rhi ? d_robin_rhi_b.data() : nullptr;
    amrex::Real const* const robin_rhi_f = use_robin_rhi ? d_robin_rhi_f.data() : nullptr;
    amrex::Real const* const robin_rlo_a = use_robin_rlo ? d_robin_rlo_a.data() : nullptr;
    amrex::Real const* const robin_rlo_b = use_robin_rlo ? d_robin_rlo_b.data() : nullptr;
    amrex::Real const* const robin_rlo_f = use_robin_rlo ? d_robin_rlo_f.data() : nullptr;
    amrex::Real const* const dirichlet_rlo_phi = use_dirichlet_rlo ? d_dirichlet_rlo_phi.data() : nullptr;
    amrex::Real const* const inlet_phi = use_inlet_dirichlet ? d_inlet_phi.data() : nullptr;

    amrex::ReduceOps<amrex::ReduceOpMin, amrex::ReduceOpMax, amrex::ReduceOpMin,
                     amrex::ReduceOpMax, amrex::ReduceOpSum> reduce_ops;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real,
                      amrex::Real, amrex::Real> reduce_data(reduce_ops);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    amrex::Real const zlo = geom.ProbLo(1);
    amrex::Real const zhi = geom.ProbHi(1);
    amrex::Real const inv_eps0 = amrex::Real(1.0) / PhysConst::epsilon_0;
    int const rlo_node = nd.smallEnd(0);
    int const rhi_node = nd.bigEnd(0);
    int const zlo_node = nd.smallEnd(1);
    int const zhi_node = nd.bigEnd(1);

    for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const& rarr = rhs.array(mfi);
        auto const& rhoarr = rho.const_array(mfi);
        auto const& parr = phi.array(mfi);
        auto const& ra = robin_a.array(mfi);
        auto const& rb = robin_b.array(mfi);
        auto const& rf = robin_f.array(mfi);

        reduce_ops.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
            {
                amrex::Real const r = plo[0] + amrex::Real(i)*dx[0];
                amrex::Real const z = plo[1] + amrex::Real(j)*dx[1];
                amrex::Real const rho_src = params.static_rho_test
                    ? staticRho(params, zlo, zhi, r, z)
                    : rhoarr(i,j,k);
                amrex::Real const rhs_val = -rho_src * inv_eps0;
                rarr(i,j,k) = rhs_val;
                NodeGeom const ng = analyticNodeGeom(hd, nr0, nz0, i, j);
                if (i == rlo_node) {
                    if (rlo_bc == RLoBoundary::Robin) {
                        bool const active_rlo = contains(hd.out, zhi, z, hd.align_tol);
                        amrex::Real a = params.robin_lo_r_a;
                        amrex::Real b = params.robin_lo_r_b;
                        amrex::Real f = params.robin_lo_r_f;
                        if (use_robin_rlo && active_rlo) {
                            a = robin_rlo_a[j];
                            b = robin_rlo_b[j];
                            f = robin_rlo_f[j];
                        }
                        ra(i,j,k,0) = a;
                        rb(i,j,k,0) = b;
                        rf(i,j,k,0) = f;
                    } else if (rlo_bc == RLoBoundary::Dirichlet) {
                        parr(i,j,k) = use_dirichlet_rlo
                            ? dirichlet_rlo_phi[j]
                            : params.potential_lo_r;
                    }
                }
                if (i == rhi_node) {
                    bool const active_rhi = contains(hd.out, zhi, z, hd.align_tol);
                    amrex::Real a = amrex::Real(1.0);
                    amrex::Real b = amrex::Real(1.0);
                    amrex::Real f = amrex::Real(0.0);
                    if (use_robin_rhi && active_rhi) {
                        a = robin_rhi_a[j];
                        b = robin_rhi_b[j];
                        f = robin_rhi_f[j];
                    }
                    ra(i,j,k,1) = a;
                    rb(i,j,k,1) = b;
                    rf(i,j,k,1) = f;
                }
                if (j == zhi_node) {
                    amrex::Real a = amrex::Real(1.0);
                    amrex::Real b = amrex::Real(1.0);
                    amrex::Real f = amrex::Real(0.0);
                    if (use_robin_zhi) {
                        a = robin_zhi_a[i];
                        b = robin_zhi_b[i];
                        f = robin_zhi_f[i];
                    }
                    ra(i,j,k,3) = a;
                    rb(i,j,k,3) = b;
                    rf(i,j,k,3) = f;
                }
                if (use_inlet_dirichlet && j == zlo_node &&
                    contains(hd.lob, hd.hib, r, hd.align_tol))
                {
                    parr(i,j,k) = inlet_phi[i];
                }

                return {rho_src, rho_src, rhs_val, rhs_val, rho_src * ng.vrz};
            });
    }

    amrex::Real rho_min = amrex::get<0>(reduce_data.value());
    amrex::Real rho_max = amrex::get<1>(reduce_data.value());
    amrex::Real rhs_min = amrex::get<2>(reduce_data.value());
    amrex::Real rhs_max = amrex::get<3>(reduce_data.value());
    amrex::Real rho_sum_vr = amrex::get<4>(reduce_data.value());
    amrex::ParallelDescriptor::ReduceRealMin(rho_min);
    amrex::ParallelDescriptor::ReduceRealMax(rho_max);
    amrex::ParallelDescriptor::ReduceRealMin(rhs_min);
    amrex::ParallelDescriptor::ReduceRealMax(rhs_max);
    amrex::ParallelDescriptor::ReduceRealSum(rho_sum_vr);

    EBFluxStats eb_flux;
    EBRobinStats eb_robin_stats;
    bool const eb_robin_mode = params.eb_bc_mode == EBBCMode::Robin;
    if (eb_robin_mode) {
        eb_robin_stats = fillEBRobin(params, geom, eb_factory, eb_a_cc, eb_b_cc, eb_f_cc,
                                     amr_level);
    } else {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!anyPythonEBDataActive(s_python_eb_robin),
            "hallrz.set_eb_robin(a,b,f) is active but warpx.hall_rz_eb_bc_mode is not robin.");
        eb_flux = fillEBNeumann(params, geom, eb_factory, eb_gn_cc, amr_level);
    }

    linop.setLevelBC(0, &neumann_bc, &robin_a, &robin_b, &robin_f);
    if (eb_robin_mode) {
        linop.setEBFVMRobin(0, eb_a_cc, eb_b_cc, eb_f_cc);
    } else {
        linop.setEBInhomogNeumann(0, eb_gn_cc);
    }
    auto eb_node_geom = MakeNodalGeometry(params, geom, cell_grids, dmap, nr0, nz0, 1);
    linop.setEBNodalGeometry(0, 0, *eb_node_geom);

    amrex::MLMG mlmg(linop);
    mlmg.setVerbose(params.verbose);
    mlmg.setBottomVerbose(0);
    mlmg.setMaxIter(params.max_iter);

    amrex::Vector<amrex::MultiFab*> pphi{&phi};
    amrex::Vector<amrex::MultiFab const*> prhs{&rhs};
    amrex::Real const resid = mlmg.solve(pphi, prhs, params.rel_tol, params.abs_tol);
    int const actual_nmg_levels = linop.NMGLevels(0);

    amrex::MultiFab eb_rhs_diag(nodal_grids, dmap, 1, 0, amrex::MFInfo{}, eb_factory);
    eb_rhs_diag.setVal(amrex::Real(0.0));
    linop.diagnosticApplyInhomogNeumannTerm(0, eb_rhs_diag);
    amrex::Real const q_eb_added = sumOperatorRZWeightedOwner(eb_rhs_diag, geom, hd, nr0, nz0);
    amrex::Real const q_eb_report_total = eb_robin_mode ? q_eb_added : eb_flux.total;
    amrex::Real const q_eb_diff = eb_robin_mode ? amrex::Real(0.0) : eb_flux.total - q_eb_added;
    amrex::Real const q_eb_eps = eb_robin_mode ? amrex::Real(0.0) :
        amrex::Math::abs(q_eb_diff) /
        (amrex::Math::abs(eb_flux.total) + amrex::Real(1.0e-300));

    ComputeStaggeredE(phi, Efield, geom, eb_factory);

    amrex::MultiFab rho_after_diff(rho.boxArray(), rho.DistributionMap(), 1, 0);
    amrex::MultiFab::Copy(rho_after_diff, rho, 0, 0, 1, 0);
    amrex::MultiFab::Subtract(rho_after_diff, rho_before, 0, 0, 1, 0);
    amrex::Real const rho_mutation = rho_after_diff.norminf(0, 0, false);

    auto const [phi_nan, phi_inf] = countNaNInf(phi);
    auto const [er_nan, er_inf] = countNaNInf(*Efield[0]);
    auto const [ez_nan, ez_inf] = countNaNInf(*Efield[2]);
    long const n_nan = phi_nan + er_nan + ez_nan;
    long const n_inf = phi_inf + er_inf + ez_inf;

    amrex::Real const phi_min = phi.min(0);
    amrex::Real const phi_max = phi.max(0);
    amrex::Real const phi_mean = phi.sum(0, false) / amrex::max(amrex::Real(1.0), amrex::Real(phi.boxArray().numPts()));
    amrex::Real const er_min = Efield[0]->min(0);
    amrex::Real const er_max = Efield[0]->max(0);
    amrex::Real const ez_min = Efield[2]->min(0);
    amrex::Real const ez_max = Efield[2]->max(0);
    amrex::IntVect const er_type = Efield[0]->ixType().toIntVect();
    amrex::IntVect const ez_type = Efield[2]->ixType().toIntVect();

    int const step = WarpX::GetInstance().getistep(0);
    int const diag_interval = std::max(params.diag_interval, 1);
    bool const do_summary = (step % diag_interval == 0) || (params.verbose >= 1);
    if (do_summary) {
        amrex::Print() << "HallRZ Poisson solve: step=" << step
                       << ", amr_level=" << amr_level
                       << ", source=" << (params.static_rho_test ? "static_rho_test" : "rho_fp")
                       << ", eb_bc_mode=" << (eb_robin_mode ? "robin" : "neumann")
                       << ", requested_mcl=" << requested_mcl
                       << ", actual_nmg_levels=" << actual_nmg_levels
                       << ", Final Iter=" << mlmg.getNumIters()
                       << ", resid/resid0=" << resid
                       << ", q_EB_total_RZ=" << q_eb_report_total
                       << ", q_EB_added_RZ=" << q_eb_added
                       << ", q_EB_diff_RZ=" << q_eb_diff
                       << ", epsilon_q=" << q_eb_eps
                       << ", phi_min=" << phi_min
                       << ", phi_max=" << phi_max
                       << ", Er_min=" << er_min
                       << ", Er_max=" << er_max
                       << ", Ez_min=" << ez_min
                       << ", Ez_max=" << ez_max
                       << ", N_NaN=" << n_nan
                       << ", N_Inf=" << n_inf;
        if (params.verbose >= 2) {
            amrex::Print() << ", q_EB_seg0_total_RZ=" << eb_flux.segment[0]
                           << ", q_EB_seg1_total_RZ=" << eb_flux.segment[1]
                           << ", q_EB_seg2_total_RZ=" << eb_flux.segment[2]
                           << ", q_EB_seg3_total_RZ=" << eb_flux.segment[3]
                           << ", EB_Robin_active_cut_cells=" << eb_robin_stats.active
                           << ", EB_Robin_seg0_cut_cells=" << eb_robin_stats.segment[0]
                           << ", EB_Robin_seg1_cut_cells=" << eb_robin_stats.segment[1]
                           << ", EB_Robin_seg2_cut_cells=" << eb_robin_stats.segment[2]
                           << ", EB_Robin_seg3_cut_cells=" << eb_robin_stats.segment[3]
                           << ", rho_min=" << rho_min
                           << ", rho_max=" << rho_max
                           << ", rho_sum_Vr=" << rho_sum_vr
                           << ", rhs_min=" << rhs_min
                           << ", rhs_max=" << rhs_max
                           << ", rho_mutation_inf=" << rho_mutation
                           << ", phi_mean=" << phi_mean
                           << ", Efield_fp[0].ixType()=" << er_type
                           << ", Efield_fp[2].ixType()=" << ez_type;
        }
        amrex::Print() << "\n";
    }
}

void
HallRZPoissonSolver::ComputeStaggeredE (amrex::MultiFab const& phi,
                                        amrex::Array<amrex::MultiFab*,3> const& Efield,
                                        amrex::Geometry const& geom,
                                        amrex::EBFArrayBoxFactory const& eb_factory)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Efield[0] != nullptr && Efield[2] != nullptr,
                                     "HallRZ staggered E writeback requires Efield[0] and Efield[2].");

    amrex::IntVect const er_type = Efield[0]->ixType().toIntVect();
    amrex::IntVect const ez_type = Efield[2]->ixType().toIntVect();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(er_type == amrex::IntVect(AMREX_D_DECL(0,1,0)),
                                     "HallRZ expects Efield_fp[0] centering IntVect(0,1) for E_r.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ez_type == amrex::IntVect(AMREX_D_DECL(1,0,0)),
                                     "HallRZ expects Efield_fp[2] centering IntVect(1,0) for E_z.");

    amrex::Real const dr = geom.CellSize(0);
    amrex::Real const dz = geom.CellSize(1);
    auto const& levset = eb_factory.getLevelSet();

    for (amrex::MFIter mfi(*Efield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const& er = Efield[0]->array(mfi);
        auto const& ph = phi.const_array(mfi);
        auto const& ls = levset.const_array(mfi);

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                bool const fluid_edge = (ls(i,j,0) <= amrex::Real(0.0)) &&
                                        (ls(i+1,j,0) <= amrex::Real(0.0));
                er(i,j,k) = fluid_edge
                    ? -(ph(i+1,j,k) - ph(i,j,k)) / dr
                    : amrex::Real(0.0);
            });
    }

    for (amrex::MFIter mfi(*Efield[2], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const& ez = Efield[2]->array(mfi);
        auto const& ph = phi.const_array(mfi);
        auto const& ls = levset.const_array(mfi);

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                bool const fluid_edge = (ls(i,j,0) <= amrex::Real(0.0)) &&
                                        (ls(i,j+1,0) <= amrex::Real(0.0));
                ez(i,j,k) = fluid_edge
                    ? -(ph(i,j+1,k) - ph(i,j,k)) / dz
                    : amrex::Real(0.0);
            });
    }

    if (HallRZPoissonSolver::ReadParameters().verbose >= 1) {
        amrex::Real const er_min = Efield[0]->min(0);
        amrex::Real const er_max = Efield[0]->max(0);
        amrex::Real const ez_min = Efield[2]->min(0);
        amrex::Real const ez_max = Efield[2]->max(0);
        amrex::Print() << "HallRZ staggered E writeback: "
                       << "E_r ixType=" << er_type
                       << ", E_z ixType=" << ez_type
                       << ", E_r[min,max]=[" << er_min << ", " << er_max << "]"
                       << ", E_z[min,max]=[" << ez_min << ", " << ez_max << "]\n";
    }
}
