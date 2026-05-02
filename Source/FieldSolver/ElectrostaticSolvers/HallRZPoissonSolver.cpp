/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "HallRZPoissonSolver.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_Box.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLEBNodeFVLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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

bool s_python_eb_neumann_active = false;
int s_python_eb_neumann_nr = 0;
int s_python_eb_neumann_nz = 0;
std::vector<amrex::Real> s_python_eb_neumann;

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

bool s_python_inlet_dirichlet_active = false;
int s_python_inlet_dirichlet_nrp1 = 0;
std::vector<amrex::Real> s_python_inlet_dirichlet_phi;

amrex::Real
overlap (amrex::Real alo, amrex::Real ahi, amrex::Real blo, amrex::Real bhi) noexcept
{
    return amrex::max(amrex::Real(0.0), amrex::min(ahi,bhi) - amrex::max(alo,blo));
}

bool
contains (amrex::Real lo, amrex::Real hi, amrex::Real x, amrex::Real eps) noexcept
{
    return x >= lo-eps && x <= hi+eps;
}

int
hallSegment (HallDomain const& h, amrex::Real r, amrex::Real z) noexcept
{
    amrex::Real const eps = amrex::max(h.align_tol, amrex::Real(1.0e-12));
    if (std::abs(r - h.lob) <= eps && contains(h.zlo, h.out, z, eps)) {
        return 0;
    }
    if (std::abs(r - h.hib) <= eps && contains(h.zlo, h.out, z, eps)) {
        return 1;
    }
    if (std::abs(z - h.out) <= eps && contains(h.rlo, h.lob, r, eps)) {
        return 2;
    }
    if (std::abs(z - h.out) <= eps && contains(h.hib, h.rhi, r, eps)) {
        return 3;
    }
    return -1;
}

amrex::Real
rectArea (Rect const& a, Rect const& b) noexcept
{
    return overlap(a.rlo,a.rhi,b.rlo,b.rhi) * overlap(a.zlo,a.zhi,b.zlo,b.zhi);
}

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

amrex::Real
verticalFluidRZArea (HallDomain const& h,
                     amrex::Real r,
                     amrex::Real za,
                     amrex::Real zb) noexcept
{
    return r * verticalFluidLength(h, r, za, zb);
}

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

amrex::Real
ebLengthInNode (HallDomain const& h, int seg, Rect const& d) noexcept
{
    if (seg == 0) { // r=lob, zlo..out
        return contains(d.rlo, d.rhi, h.lob, h.align_tol)
            ? overlap(d.zlo,d.zhi,h.zlo,h.out) : amrex::Real(0.0);
    } else if (seg == 1) { // r=hib, zlo..out
        return contains(d.rlo, d.rhi, h.hib, h.align_tol)
            ? overlap(d.zlo,d.zhi,h.zlo,h.out) : amrex::Real(0.0);
    } else if (seg == 2) { // z=out, rlo..lob
        return contains(d.zlo, d.zhi, h.out, h.align_tol)
            ? overlap(d.rlo,d.rhi,h.rlo,h.lob) : amrex::Real(0.0);
    } else { // z=out, hib..rhi
        return contains(d.zlo, d.zhi, h.out, h.align_tol)
            ? overlap(d.rlo,d.rhi,h.hib,h.rhi) : amrex::Real(0.0);
    }
}

amrex::Real
ebRZAreaInNode (HallDomain const& h, int seg, Rect const& d) noexcept
{
    if (seg == 0) { // r=lob, zlo..out
        return contains(d.rlo, d.rhi, h.lob, h.align_tol)
            ? h.lob * overlap(d.zlo,d.zhi,h.zlo,h.out) : amrex::Real(0.0);
    } else if (seg == 1) { // r=hib, zlo..out
        return contains(d.rlo, d.rhi, h.hib, h.align_tol)
            ? h.hib * overlap(d.zlo,d.zhi,h.zlo,h.out) : amrex::Real(0.0);
    } else if (seg == 2) { // z=out, rlo..lob
        return contains(d.zlo, d.zhi, h.out, h.align_tol)
            ? radialMomentOverlap(d.rlo,d.rhi,h.rlo,h.lob) : amrex::Real(0.0);
    } else { // z=out, hib..rhi
        return contains(d.zlo, d.zhi, h.out, h.align_tol)
            ? radialMomentOverlap(d.rlo,d.rhi,h.hib,h.rhi) : amrex::Real(0.0);
    }
}

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

amrex::Real
staticRho (HallRZPoissonSolver::Params const& params,
           amrex::Geometry const& geom,
           amrex::Real r, amrex::Real z) noexcept
{
    if (params.static_rho_mode == 0) {
        return amrex::Real(0.0);
    }
    if (params.static_rho_mode == 2) {
        return params.static_rho_amp;
    }

    amrex::Real const rc = amrex::Real(0.5) * (params.lob + params.hib);
    amrex::Real const zc = amrex::Real(0.5) * (geom.ProbLo(1) + params.out);
    amrex::Real const lr = params.static_rho_lambda_r;
    amrex::Real const lz = params.static_rho_lambda_z;
    amrex::Real const ar = (r - rc) / lr;
    amrex::Real const az = (z - zc) / lz;
    return params.static_rho_amp * std::exp(-(ar*ar + az*az));
}

std::pair<long,long>
countNaNInf (amrex::MultiFab const& mf)
{
    long nnan = 0;
    long ninf = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.validbox();
        auto const& a = mf.const_array(mfi);
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                amrex::Real const v = a(i,j,0);
                if (std::isnan(v)) { ++nnan; }
                if (std::isinf(v)) { ++ninf; }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceLongSum(nnan);
    amrex::ParallelDescriptor::ReduceLongSum(ninf);
    return {nnan, ninf};
}

EBFluxStats
fillEBNeumann (HallRZPoissonSolver::Params const& params,
               amrex::Geometry const& geom,
               amrex::EBFArrayBoxFactory const& eb_factory,
               amrex::MultiFab& eb_gn_cc)
{
    EBFluxStats stats;
    eb_gn_cc.setVal(amrex::Real(0.0));

    bool const use_python = s_python_eb_neumann_active;
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
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geom.Domain().length(0) == s_python_eb_neumann_nr &&
                                         geom.Domain().length(1) == s_python_eb_neumann_nz,
            "Python HallRZ EB Neumann array shape does not match the level-0 cell domain.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geom.Domain().smallEnd(0) == 0 &&
                                         geom.Domain().smallEnd(1) == 0,
            "Python HallRZ EB Neumann array currently requires a zero-based cell domain.");
    }

    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    HallDomain const hd{geom.ProbLo(0), geom.ProbHi(0), geom.ProbLo(1), geom.ProbHi(1),
                        params.lob, params.hib, params.out,
                        amrex::max(params.align_tol,
                                   amrex::Real(1.0e-8) * amrex::max(dx[0], dx[1]))};

    for (amrex::MFIter mfi(eb_gn_cc); mfi.isValid(); ++mfi) {
        if (!eb_factory.getBndryArea().ok(mfi)) {
            continue;
        }

        amrex::Box ccbx = eb_gn_cc[mfi].box();
        ccbx &= geom.Domain();

        auto const& gn = eb_gn_cc.array(mfi);
        auto const& ba = eb_factory.getBndryArea().const_array(mfi);
        auto const& bc = eb_factory.getBndryCent().const_array(mfi);
        auto const& bn = eb_factory.getBndryNormal().const_array(mfi);
        auto const& flg = eb_factory.getMultiEBCellFlagFab().const_array(mfi);

        for (int j = ccbx.smallEnd(1); j <= ccbx.bigEnd(1); ++j) {
            for (int i = ccbx.smallEnd(0); i <= ccbx.bigEnd(0); ++i) {
                if (!flg(i,j,0).isSingleValued()) {
                    continue;
                }
                amrex::Real const area = ba(i,j,0);
                if (area <= amrex::Real(0.0)) {
                    continue;
                }

                amrex::Real const fx = amrex::max(amrex::Real(0.0),
                    amrex::min(amrex::Real(1.0), amrex::Real(0.5) + bc(i,j,0,0)));
                amrex::Real const fy = amrex::max(amrex::Real(0.0),
                    amrex::min(amrex::Real(1.0), amrex::Real(0.5) + bc(i,j,0,1)));
                amrex::Real const rbc = plo[0] + (amrex::Real(i) + fx) * dx[0];
                amrex::Real const zbc = plo[1] + (amrex::Real(j) + fy) * dx[1];
                int const seg = hallSegment(hd, rbc, zbc);
                if (seg < 0) {
                    continue;
                }
                if (params.eb_wall_flux_segment >= 0 &&
                    params.eb_wall_flux_segment != seg)
                {
                    continue;
                }

                amrex::Real const geb = use_python
                    ? s_python_eb_neumann[static_cast<std::size_t>(i) * s_python_eb_neumann_nz
                                          + static_cast<std::size_t>(j)]
                    : params.eb_g0;
                gn(i,j,0,0) = geb;

                amrex::Real const len_scale = std::hypot(dx[0]*bn(i,j,0,1),
                                                         dx[1]*bn(i,j,0,0));
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

                stats.total += q;
                stats.segment[seg] += q;
            }
        }
    }

    amrex::ParallelDescriptor::ReduceRealSum(stats.total);
    for (auto& s : stats.segment) {
        amrex::ParallelDescriptor::ReduceRealSum(s);
    }
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
    amrex::Real sum = amrex::Real(0.0);
    amrex::Real const dr = geom.CellSize(0);
    int const ilo = amrex::surroundingNodes(geom.Domain()).smallEnd(0);
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.validbox();
        auto const& a = mf.const_array(mfi);
        auto const& om = owner_mask->const_array(mfi);
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if (om(i,j,0) == 0) {
                    continue;
                }
                NodeGeom const ng = analyticNodeGeom(hd, nr0, nz0, i, j);
                amrex::Real const rnode = geom.ProbLo(0) + amrex::Real(i) * dr;
                amrex::Real const vfac = (geom.ProbLo(0) == amrex::Real(0.0) && i == ilo)
                    ? amrex::Real(0.25) * dr
                    : amrex::max(rnode, amrex::Real(0.25) * dr);
                sum += ng.v2d * vfac * a(i,j,0);
            }
        }
    }
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
    pp_warpx.query("hall_rz_allow_nonaligned_coarse_geom",
                   params.allow_nonaligned_coarse_geom);
    pp_warpx.query("hall_rz_rz_weighted_geom", params.rz_weighted_geom);
    pp_warpx.query("hall_rz_static_rho_test", params.static_rho_test);
    pp_warpx.query("hall_rz_static_rho_mode", params.static_rho_mode);
    pp_warpx.query("hall_rz_static_rho_amp", params.static_rho_amp);
    pp_warpx.query("hall_rz_static_rho_lambda_r", params.static_rho_lambda_r);
    pp_warpx.query("hall_rz_static_rho_lambda_z", params.static_rho_lambda_z);
    bool have_eb_g0 = pp_warpx.query("hall_rz_eb_g0", params.eb_g0);
    pp_warpx.query("hall_rz_eb_wall_flux", params.eb_wall_flux);
    pp_warpx.query("hall_rz_eb_wall_flux_segment", params.eb_wall_flux_segment);
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
HallRZPoissonSolver::SetPythonEBNeumann (int nr, int nz, std::vector<amrex::Real> data)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nr > 0 && nz > 0,
        "HallRZ Python EB Neumann array dimensions must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(data.size() == static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz),
        "HallRZ Python EB Neumann array data size does not match (nr,nz).");

    s_python_eb_neumann_nr = nr;
    s_python_eb_neumann_nz = nz;
    s_python_eb_neumann = std::move(data);
    s_python_eb_neumann_active = true;
}

void
HallRZPoissonSolver::ClearPythonEBNeumann ()
{
    s_python_eb_neumann_active = false;
    s_python_eb_neumann_nr = 0;
    s_python_eb_neumann_nz = 0;
    s_python_eb_neumann.clear();
}

bool
HallRZPoissonSolver::HasPythonEBNeumann ()
{
    return s_python_eb_neumann_active;
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

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(rlo) < 10.0 * std::numeric_limits<amrex::Real>::epsilon(),
                                     "HallRZ requires r_min=0.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rlo < params.lob && params.lob < params.hib && params.hib < rhi,
                                     "HallRZ requires r_min < lob < hib < r_max.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(zlo < params.out && params.out < zhi,
                                     "HallRZ requires z_min < out < z_max.");

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

int
HallRZPoissonSolver::AlignedMaxCoarseningLevel (Params const& params,
                                                amrex::Geometry const& geom,
                                                int requested_max)
{
    if (!params.enabled) { return requested_max; }
    ValidateGeometry(params, geom);

    int const nr0 = geom.Domain().length(0);
    int const nz0 = geom.Domain().length(1);
    amrex::Real const rlo = geom.ProbLo(0);
    amrex::Real const rhi = geom.ProbHi(0);
    amrex::Real const zlo = geom.ProbLo(1);
    amrex::Real const zhi = geom.ProbHi(1);

    int nr = nr0;
    int nz = nz0;
    int aligned_max = 0;
    for (int lev = 1; lev <= requested_max; ++lev) {
        if ((nr % 2) != 0 || (nz % 2) != 0) { break; }
        nr /= 2;
        nz /= 2;
        amrex::Real const dr = (rhi-rlo) / amrex::Real(nr);
        amrex::Real const dz = (zhi-zlo) / amrex::Real(nz);
        auto aligned = [&params] (amrex::Real x, amrex::Real xlo, amrex::Real h) noexcept {
            amrex::Real const idx = (x - xlo) / h;
            return std::abs(idx - std::round(idx)) < params.align_tol;
        };
        if (!aligned(params.lob, rlo, dr) ||
            !aligned(params.hib, rlo, dr) ||
            !aligned(params.out, zlo, dz)) {
            break;
        }
        aligned_max = lev;
    }
    return amrex::min(requested_max, aligned_max);
}

std::unique_ptr<amrex::MultiFab>
HallRZPoissonSolver::MakeNodalGeometry (Params const& params,
                                        amrex::Geometry const& geom,
                                        amrex::BoxArray const& cell_grids,
                                        amrex::DistributionMapping const& dmap,
                                        int nr_cells,
                                        int nz_cells,
                                        int ngrow,
                                        bool rz_weighted)
{
    ValidateGeometry(params, geom);

    amrex::BoxArray nodal_grids = amrex::convert(cell_grids, amrex::IntVect::TheNodeVector());
    auto eb_node_geom = std::make_unique<amrex::MultiFab>(nodal_grids, dmap, 6, ngrow);
    eb_node_geom->setVal(amrex::Real(0.0));

    int const nr = (nr_cells > 0) ? nr_cells : geom.Domain().length(0);
    int const nz = (nz_cells > 0) ? nz_cells : geom.Domain().length(1);
    HallDomain const h{geom.ProbLo(0), geom.ProbHi(0),
                       geom.ProbLo(1), geom.ProbHi(1),
                       params.lob, params.hib, params.out, params.align_tol};

    for (amrex::MFIter mfi(*eb_node_geom); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.validbox();
        auto const& g = eb_node_geom->array(mfi);
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                NodeGeom const ng = analyticNodeGeom(h, nr, nz, i, j);
                if (rz_weighted) {
                    g(i,j,0,0) = ng.vrz;
                    g(i,j,0,1) = ng.arm_rz;
                    g(i,j,0,2) = ng.arp_rz;
                    g(i,j,0,3) = ng.azm_rz;
                    g(i,j,0,4) = ng.azp_rz;
                    g(i,j,0,5) = ng.aeb_rz;
                } else {
                    g(i,j,0,0) = ng.v2d;
                    g(i,j,0,1) = ng.arm;
                    g(i,j,0,2) = ng.arp;
                    g(i,j,0,3) = ng.azm;
                    g(i,j,0,4) = ng.azp;
                    g(i,j,0,5) = ng.aeb;
                }
            }
        }
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
                                     amrex::EBFArrayBoxFactory const& eb_factory)
{
    Params const params = ReadParameters();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(params.enabled,
                                     "HallRZPoissonSolver::ComputePhiAndE requires warpx.hall_rz_enable=1.");
    ValidateGeometry(params, geom);

    amrex::MultiFab rho_before(rho.boxArray(), rho.DistributionMap(), 1, 0);
    amrex::MultiFab::Copy(rho_before, rho, 0, 0, 1, 0);

    int const requested_mcl = amrex::max(0, params.max_coarsening_level);
    int const aligned_max = AlignedMaxCoarseningLevel(params, geom, requested_mcl);
    int const effective_mcl = params.allow_nonaligned_coarse_geom
        ? requested_mcl
        : aligned_max;
    amrex::LPInfo info;
    info.setMaxCoarseningLevel(effective_mcl);

    amrex::Vector<amrex::Geometry> pgeom{geom};
    amrex::Vector<amrex::BoxArray> pgrids{cell_grids};
    amrex::Vector<amrex::DistributionMapping> pdmap{dmap};
    amrex::Vector<amrex::EBFArrayBoxFactory const*> pfac{&eb_factory};

    amrex::MLEBNodeFVLaplacian linop(pgeom, pgrids, pdmap, info, pfac);
    amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> lobc{
        AMREX_D_DECL(amrex::LinOpBCType::inhomogNeumann,
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

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho.ixType() == rhs.ixType(),
        "HallRZPoissonSolver expects rho_fp to have the same nodal centering as phi/rhs.");

    rhs.setVal(amrex::Real(0.0));
    neumann_bc.setVal(amrex::Real(0.0));
    robin_a.setVal(amrex::Real(0.0));
    robin_b.setVal(amrex::Real(0.0));
    robin_f.setVal(amrex::Real(0.0));
    eb_gn_cc.setVal(amrex::Real(0.0));
    phi.setVal(amrex::Real(0.0));

    auto const dx = geom.CellSizeArray();
    auto const plo = geom.ProbLoArray();
    amrex::Box const nd = amrex::surroundingNodes(geom.Domain());
    HallDomain const hd{geom.ProbLo(0), geom.ProbHi(0), geom.ProbLo(1), geom.ProbHi(1),
                        params.lob, params.hib, params.out, params.align_tol};
    int const nr0 = geom.Domain().length(0);
    int const nz0 = geom.Domain().length(1);
    if (s_python_robin_zhi_active) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(s_python_robin_zhi_nrp1 == nr0 + 1,
            "Python HallRZ z-hi Robin arrays must have shape (Nr+1).");
    }
    if (s_python_robin_rhi_active) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(s_python_robin_rhi_nzp1 == nz0 + 1,
            "Python HallRZ r-hi Robin arrays must have shape (Nz+1).");
    }
    if (s_python_inlet_dirichlet_active) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(s_python_inlet_dirichlet_nrp1 == nr0 + 1,
            "Python HallRZ inlet Dirichlet array must have shape (Nr+1).");
    }

    amrex::Real rho_min = std::numeric_limits<amrex::Real>::max();
    amrex::Real rho_max = -std::numeric_limits<amrex::Real>::max();
    amrex::Real rhs_min = std::numeric_limits<amrex::Real>::max();
    amrex::Real rhs_max = -std::numeric_limits<amrex::Real>::max();
    amrex::Real rho_sum_vr = amrex::Real(0.0);

    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.validbox();
        auto const& rarr = rhs.array(mfi);
        auto const& rhoarr = rho.const_array(mfi);
        auto const& parr = phi.array(mfi);
        auto const& ra = robin_a.array(mfi);
        auto const& rb = robin_b.array(mfi);
        auto const& rf = robin_f.array(mfi);
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                amrex::Real const r = plo[0] + amrex::Real(i)*dx[0];
                amrex::Real const z = plo[1] + amrex::Real(j)*dx[1];
                amrex::Real const rho_src = params.static_rho_test
                    ? staticRho(params, geom, r, z)
                    : rhoarr(i,j,0);
                amrex::Real const rhs_val = -rho_src / PhysConst::epsilon_0;
                rarr(i,j,0) = rhs_val;
                NodeGeom const ng = analyticNodeGeom(hd, nr0, nz0, i, j);
                rho_min = amrex::min(rho_min, rho_src);
                rho_max = amrex::max(rho_max, rho_src);
                rhs_min = amrex::min(rhs_min, rhs_val);
                rhs_max = amrex::max(rhs_max, rhs_val);
                rho_sum_vr += rho_src * ng.vrz;
                if (i == nd.bigEnd(0)) {
                    bool const active_rhi = contains(params.out, geom.ProbHi(1), z, params.align_tol);
                    amrex::Real a = amrex::Real(1.0);
                    amrex::Real b = amrex::Real(1.0);
                    amrex::Real f = amrex::Real(0.0);
                    if (s_python_robin_rhi_active && active_rhi) {
                        a = s_python_robin_rhi_a[static_cast<std::size_t>(j)];
                        b = s_python_robin_rhi_b[static_cast<std::size_t>(j)];
                        f = s_python_robin_rhi_f[static_cast<std::size_t>(j)];
                        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(a) + std::abs(b) > amrex::Real(0.0),
                            "Invalid Python HallRZ r-hi Robin data: a and b cannot both be zero on an active node.");
                    }
                    ra(i,j,0,1) = a;
                    rb(i,j,0,1) = b;
                    rf(i,j,0,1) = f;
                }
                if (j == nd.bigEnd(1)) {
                    amrex::Real a = amrex::Real(1.0);
                    amrex::Real b = amrex::Real(1.0);
                    amrex::Real f = amrex::Real(0.0);
                    if (s_python_robin_zhi_active) {
                        a = s_python_robin_zhi_a[static_cast<std::size_t>(i)];
                        b = s_python_robin_zhi_b[static_cast<std::size_t>(i)];
                        f = s_python_robin_zhi_f[static_cast<std::size_t>(i)];
                        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(a) + std::abs(b) > amrex::Real(0.0),
                            "Invalid Python HallRZ z-hi Robin data: a and b cannot both be zero on an active node.");
                    }
                    ra(i,j,0,3) = a;
                    rb(i,j,0,3) = b;
                    rf(i,j,0,3) = f;
                }
                if (s_python_inlet_dirichlet_active && j == nd.smallEnd(1) &&
                    contains(params.lob, params.hib, r, params.align_tol))
                {
                    parr(i,j,0) = s_python_inlet_dirichlet_phi[static_cast<std::size_t>(i)];
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealMin(rho_min);
    amrex::ParallelDescriptor::ReduceRealMax(rho_max);
    amrex::ParallelDescriptor::ReduceRealMin(rhs_min);
    amrex::ParallelDescriptor::ReduceRealMax(rhs_max);
    amrex::ParallelDescriptor::ReduceRealSum(rho_sum_vr);

    EBFluxStats const eb_flux = fillEBNeumann(params, geom, eb_factory, eb_gn_cc);

    linop.setLevelBC(0, &neumann_bc, &robin_a, &robin_b, &robin_f);
    linop.setEBFVMForce(true);
    linop.setEBFVMRZMetricMode(1);
    linop.setEBFVMRZGeomWeighted(params.rz_weighted_geom);
    linop.setEBInhomogNeumann(0, eb_gn_cc);
    linop.setEBInhomogNeumannLengthMode(1);
    linop.setEBInhomogNeumannFluxScale(amrex::Real(1.0));

    int const actual_nmg_levels = linop.NMGLevels(0);
    int const nr_fine = geom.Domain().length(0);
    int const nz_fine = geom.Domain().length(1);
    for (int mglev = 0; mglev < actual_nmg_levels; ++mglev) {
        auto const& level_geom = linop.Geom(0, mglev);
        int const nr_l = level_geom.Domain().length(0);
        int const nz_l = level_geom.Domain().length(1);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nr_l > 0 && nz_l > 0,
            "HallRZ MG geometry level has invalid domain size.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE((nr_fine % nr_l) == 0 && (nz_fine % nz_l) == 0,
            "HallRZ MG geometry level is not an integer coarsening of the fine domain.");

        amrex::IntVect const ratio(AMREX_D_DECL(nr_fine / nr_l, nz_fine / nz_l, 1));
        amrex::BoxArray level_ba = amrex::coarsen(cell_grids, ratio);
        amrex::DistributionMapping level_dm(level_ba);
        auto level_node_geom = MakeNodalGeometry(params, geom, level_ba, level_dm,
                                                 nr_l, nz_l, 1,
                                                 params.rz_weighted_geom);
        linop.setEBNodalGeometry(0, mglev, *level_node_geom);
    }

    amrex::MLMG mlmg(linop);
    mlmg.setVerbose(params.verbose);
    mlmg.setBottomVerbose(0);
    mlmg.setMaxIter(params.max_iter);

    amrex::Vector<amrex::MultiFab*> pphi{&phi};
    amrex::Vector<amrex::MultiFab const*> prhs{&rhs};
    amrex::Real const resid = mlmg.solve(pphi, prhs, params.rel_tol, params.abs_tol);

    amrex::MultiFab eb_rhs_diag(nodal_grids, dmap, 1, 0, amrex::MFInfo{}, eb_factory);
    eb_rhs_diag.setVal(amrex::Real(0.0));
    linop.diagnosticApplyInhomogNeumannTerm(0, eb_rhs_diag);
    amrex::Real const q_eb_added = sumOperatorRZWeightedOwner(eb_rhs_diag, geom, hd, nr0, nz0);
    amrex::Real const q_eb_diff = eb_flux.total - q_eb_added;
    amrex::Real const q_eb_eps = amrex::Math::abs(q_eb_diff) /
        (amrex::Math::abs(eb_flux.total) + amrex::Real(1.0e-300));

    ComputeStaggeredE(phi, Efield, geom);

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
                       << ", source=" << (params.static_rho_test ? "static_rho_test" : "rho_fp")
                       << ", Final Iter=" << mlmg.getNumIters()
                       << ", resid/resid0=" << resid
                       << ", q_EB_total_RZ=" << eb_flux.total
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
                                        amrex::Geometry const& geom)
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

    for (amrex::MFIter mfi(*Efield[0]); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.validbox();
        auto const& er = Efield[0]->array(mfi);
        auto const& ph = phi.const_array(mfi);
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                er(i,j,0) = -(ph(i+1,j,0) - ph(i,j,0)) / dr;
            }
        }
    }

    for (amrex::MFIter mfi(*Efield[2]); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.validbox();
        auto const& ez = Efield[2]->array(mfi);
        auto const& ph = phi.const_array(mfi);
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                ez(i,j,0) = -(ph(i,j+1,0) - ph(i,j,0)) / dz;
            }
        }
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
