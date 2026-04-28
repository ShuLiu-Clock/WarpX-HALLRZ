/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HALL_RZ_POISSON_SOLVER_H_
#define WARPX_HALL_RZ_POISSON_SOLVER_H_

#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Array.H>

#include <memory>
#include <vector>

namespace amrex {
class EBFArrayBoxFactory;
}

class HallRZPoissonSolver
{
public:
    struct Params
    {
        bool enabled = false;
        amrex::Real lob = 0.0;
        amrex::Real hib = 0.0;
        amrex::Real out = 0.0;
        amrex::Real align_tol = amrex::Real(1.0e-10);
        int verbose = 0;
        int diag_interval = 10;
        int max_iter = 200;
        amrex::Real rel_tol = amrex::Real(1.0e-10);
        amrex::Real abs_tol = 0.0;
        int max_coarsening_level = 30;
        bool static_rho_test = false;
        int static_rho_mode = 0;
        amrex::Real static_rho_amp = 0.0;
        amrex::Real static_rho_lambda_r = 0.004;
        amrex::Real static_rho_lambda_z = 0.006;
        bool eb_wall_flux = false;
        amrex::Real eb_g0 = 0.0;
        int eb_wall_flux_segment = -1; // -1: all Hall EB segments; 0..3: one segment only.
    };

    /** Read warpx.hall_rz_* parameters. */
    static Params ReadParameters ();

    /** Convenience query for warpx.hall_rz_enable. */
    static bool Enabled ();

    /** Set a persistent Python-provided cell-centered EB Neumann array.
     *
     * The array is global, row-major with shape (nr,nz), and stores
     * g_EB(i,j)=dphi/dn on cell centers.  Only EB2 single-valued cells consume
     * these values during the solve.
     */
    static void SetPythonEBNeumann (int nr, int nz, std::vector<amrex::Real> data);

    /** Clear the Python-provided EB Neumann array and return to scalar/default path. */
    static void ClearPythonEBNeumann ();

    /** True if a Python-provided EB Neumann array is currently active. */
    static bool HasPythonEBNeumann ();

    /** Set persistent Python-provided node-centered z-hi Robin arrays. */
    static void SetPythonRobinZHi (int nrp1, std::vector<amrex::Real> a,
                                   std::vector<amrex::Real> b,
                                   std::vector<amrex::Real> f);

    /** Clear Python-provided z-hi Robin arrays and return to scalar/default path. */
    static void ClearPythonRobinZHi ();

    /** True if Python-provided z-hi Robin arrays are active. */
    static bool HasPythonRobinZHi ();

    /** Set persistent Python-provided node-centered r-hi Robin arrays. */
    static void SetPythonRobinRHi (int nzp1, std::vector<amrex::Real> a,
                                   std::vector<amrex::Real> b,
                                   std::vector<amrex::Real> f);

    /** Clear Python-provided r-hi Robin arrays and return to scalar/default path. */
    static void ClearPythonRobinRHi ();

    /** True if Python-provided r-hi Robin arrays are active. */
    static bool HasPythonRobinRHi ();

    /** Set persistent Python-provided node-centered z-lo inlet Dirichlet values. */
    static void SetPythonInletDirichlet (int nrp1, std::vector<amrex::Real> phi);

    /** Clear Python-provided inlet Dirichlet values and return to scalar/default path. */
    static void ClearPythonInletDirichlet ();

    /** True if Python-provided inlet Dirichlet values are active. */
    static bool HasPythonInletDirichlet ();

    /** Validate RZ geometry and cell-face alignment on a given WarpX Geometry. */
    static void ValidateGeometry (Params const& params, amrex::Geometry const& geom);

    /** Return the deepest aligned MG level allowed by Hall grid-aligned geometry. */
    static int AlignedMaxCoarseningLevel (Params const& params,
                                          amrex::Geometry const& geom,
                                          int requested_max);

    /**
     * Build user-provided planar nodal EB-FVM geometry for a HallRZ level.
     *
     * Components are:
     *   0: V2D, 1: A_r-, 2: A_r+, 3: A_z-, 4: A_z+, 5: A_EB.
     *
     * The returned MultiFab is nodal and uses the input BoxArray converted to
     * nodal index type.  It is intentionally not RZ-weighted; the AMReX
     * MLEBNodeFVLaplacian must apply the RZ metric internally.
     */
    static std::unique_ptr<amrex::MultiFab>
    MakeNodalGeometry (Params const& params,
                       amrex::Geometry const& geom,
                       amrex::BoxArray const& cell_grids,
                       amrex::DistributionMapping const& dmap,
                       int nr_cells = -1,
                       int nz_cells = -1,
                       int ngrow = 1);

    /**
     * Write electrostatic field from nodal phi into WarpX staggered/Yee RZ layout.
     *
     * Required layout:
     *   E[0] = E_r on IntVect(0,1), using -(phi(i+1,j)-phi(i,j))/dr.
     *   E[2] = E_z on IntVect(1,0), using -(phi(i,j+1)-phi(i,j))/dz.
     *   E[1] is not modified here.
     */
    static void ComputeStaggeredE (amrex::MultiFab const& phi,
                                   amrex::Array<amrex::MultiFab*,3> const& Efield,
                                   amrex::Geometry const& geom);

    /**
     * HallRZ Poisson solve and WarpX staggered E writeback.
     *
     * By default this consumes the input rho as a source via rhs=-rho/epsilon0.
     * When warpx.hall_rz_static_rho_test=1, it instead uses an internal
     * analytic static source for W3/W4 smoke tests.  In both modes, the input
     * rho MultiFab is never modified.
     */
    static void ComputePhiAndE (amrex::MultiFab const& rho,
                                amrex::MultiFab& phi,
                                amrex::Array<amrex::MultiFab*,3> const& Efield,
                                amrex::Geometry const& geom,
                                amrex::BoxArray const& cell_grids,
                                amrex::DistributionMapping const& dmap,
                                amrex::EBFArrayBoxFactory const& eb_factory);
};

#endif
