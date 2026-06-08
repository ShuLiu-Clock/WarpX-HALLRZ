/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald, Arianna Formenti, Revathi Jambunathan
 *
 * License: BSD-3-Clause-LBNL
 */
#include "LabFrameExplicitES.H"
#include "HallRZPoissonSolver.H"
#include "Fluids/MultiFluidContainer_fwd.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Python/callbacks.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/Interpolate.H>
#include <ablastr/utils/Communication.H>

#include <AMReX_GpuUtility.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParticleReduce.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

using namespace amrex;

namespace {

bool
HallRZParticleDiagEnabled ()
{
    int enabled = 0;
    amrex::ParmParse pp("warpx");
    pp.query("hall_rz_particle_diag", enabled);
    return enabled != 0;
}

void
PrintHallRZExternalBParticleDiagnostics ()
{
    amrex::ParmParse pp_particles("particles");

    std::string b_style = "none";
    pp_particles.query("B_ext_particle_init_style", b_style);
    std::transform(b_style.begin(), b_style.end(), b_style.begin(), ::tolower);

    amrex::Vector<amrex::Real> b_ext(3, amrex::Real(0.0));
    if (b_style == "constant") {
        pp_particles.queryarr("B_external_particle", b_ext, 0, 3);
    }

    amrex::Real const br_min = (b_style == "constant") ? b_ext[0] : amrex::Real(0.0);
    amrex::Real const br_max = br_min;
    amrex::Real const bt_min = (b_style == "constant") ? b_ext[1] : amrex::Real(0.0);
    amrex::Real const bt_max = bt_min;
    amrex::Real const bz_min = (b_style == "constant") ? b_ext[2] : amrex::Real(0.0);
    amrex::Real const bz_max = bz_min;
    amrex::Real const bmag = std::sqrt(b_ext[0]*b_ext[0] + b_ext[1]*b_ext[1] + b_ext[2]*b_ext[2]);
    amrex::Real const dt = WarpX::GetInstance().getdt(0);
    amrex::Real const omega_ce_dt = PhysConst::q_e * bmag / PhysConst::m_e * dt;
    amrex::Real const omega_ci_dt = PhysConst::q_e * bmag / PhysConst::m_p * dt;

    amrex::Print() << "HallRZ external particle B diagnostics: style=" << b_style
                   << ", Br_min=" << br_min << ", Br_max=" << br_max
                   << ", Btheta_min=" << bt_min << ", Btheta_max=" << bt_max
                   << ", Bz_min=" << bz_min << ", Bz_max=" << bz_max
                   << ", |B|_max=" << bmag
                   << ", omega_ce_dt=" << omega_ce_dt
                   << ", omega_ci_dt=" << omega_ci_dt
                   << "\n";
}

void
PrintHallRZParticleDiagnostics (MultiParticleContainer& mpc,
                                HallRZPoissonSolver::Params const& params,
                                amrex::Geometry const& geom)
{
    if (!HallRZParticleDiagEnabled()) { return; }

    auto const species_names = mpc.GetSpeciesNames();
    auto const n_species = mpc.nSpecies();
    auto const step = WarpX::GetInstance().getistep(0);
    int interval = 1;
    amrex::ParmParse pp("warpx");
    pp.query("hall_rz_particle_diag_interval", interval);
    interval = std::max(interval, 1);
    if (step % interval != 0) { return; }
    bool const full_particle_diag = (params.verbose >= 2);

    amrex::Long total_np = 0;
    amrex::Long total_nan = 0;
    amrex::Long total_inf = 0;
    amrex::Long total_out_of_domain = 0;
    amrex::Long total_inside_covered = 0;

    if (full_particle_diag) {
        amrex::Print() << "HallRZ particle diagnostics: step=" << step
                       << ", n_species=" << n_species << "\n";
        PrintHallRZExternalBParticleDiagnostics();
    }

    for (int is = 0; is < n_species; ++is)
    {
        auto& pc = mpc.GetParticleContainer(is);
        amrex::Long const np = pc.TotalNumberOfParticles();
        total_np += np;

        if (np == 0)
        {
            if (full_particle_diag) {
                amrex::Print() << "HallRZ particle species=" << species_names[is]
                               << ": Np=0\n";
            }
            continue;
        }

        using PType = typename WarpXParticleContainer::SuperParticleType;
        using OpMin = amrex::ReduceOpMin;
        using OpMax = amrex::ReduceOpMax;
        using OpSum = amrex::ReduceOpSum;

        amrex::Real const rlo = geom.ProbLo(0);
        amrex::Real const rhi = geom.ProbHi(0);
        amrex::Real const zlo = geom.ProbLo(1);
        amrex::Real const zhi = geom.ProbHi(1);
        amrex::Real const lob = params.lob;
        amrex::Real const hib = params.hib;
        amrex::Real const out = params.out;
        amrex::Real const mass = pc.getMass();
        amrex::Real constexpr inv_c2 = amrex::Real(1.0) / (PhysConst::c * PhysConst::c);

        amrex::ReduceOps<OpMin, OpMin, OpMin, OpMin, OpMin, OpMin,
                         OpMax, OpMax, OpMax, OpMax, OpMax, OpMax,
                         OpSum, OpSum, OpSum, OpSum, OpSum, OpSum, OpSum, OpSum> reduce_ops;

        auto const reduced = amrex::ParticleReduce<
            amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                              amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                              amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                              amrex::Real, amrex::Real, amrex::Real>>(
            pc,
            [=] AMREX_GPU_DEVICE (PType const& p) noexcept
                -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                                   amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                                   amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                                   amrex::Real, amrex::Real, amrex::Real>
            {
                amrex::Real const r = p.pos(PIdx::x);
                amrex::Real const z = p.pos(PIdx::z);
                amrex::Real const w = p.rdata(PIdx::w);
                amrex::Real const ur = p.rdata(PIdx::ux);
                amrex::Real const ut = p.rdata(PIdx::uy);
                amrex::Real const uz = p.rdata(PIdx::uz);
                amrex::Real const gamma = std::sqrt(amrex::Real(1.0) +
                    (ur*ur + ut*ut + uz*uz) * inv_c2);
                amrex::Real const vr = ur / gamma;
                amrex::Real const vt = ut / gamma;
                amrex::Real const vz = uz / gamma;
                amrex::Real const kinetic = amrex::Real(0.5) * mass * w *
                    (vr*vr + vt*vt + vz*vz);

                bool const bad_nan =
                    amrex::isnan(r) || amrex::isnan(z) ||
                    amrex::isnan(ur) || amrex::isnan(ut) || amrex::isnan(uz);
                bool const bad_inf =
                    amrex::isinf(r) || amrex::isinf(z) ||
                    amrex::isinf(ur) || amrex::isinf(ut) || amrex::isinf(uz);
                bool const out_of_domain = (r < rlo) || (r > rhi) || (z < zlo) || (z > zhi);
                bool const inside_covered = (!out_of_domain) && (z < out) && ((r < lob) || (r > hib));

                return {r, z, vr, vt, vz, gamma,
                        r, z, vr, vt, vz, gamma,
                        bad_nan ? amrex::Real(1.0) : amrex::Real(0.0),
                        bad_inf ? amrex::Real(1.0) : amrex::Real(0.0),
                        out_of_domain ? amrex::Real(1.0) : amrex::Real(0.0),
                        inside_covered ? amrex::Real(1.0) : amrex::Real(0.0),
                        kinetic, vr, vt, vz};
            },
            reduce_ops);

        amrex::Real rmin = amrex::get<0>(reduced);
        amrex::Real zmin = amrex::get<1>(reduced);
        amrex::Real vrmin = amrex::get<2>(reduced);
        amrex::Real vtmin = amrex::get<3>(reduced);
        amrex::Real vzmin = amrex::get<4>(reduced);
        amrex::Real gammamin = amrex::get<5>(reduced);
        amrex::Real rmax = amrex::get<6>(reduced);
        amrex::Real zmax = amrex::get<7>(reduced);
        amrex::Real vrmax = amrex::get<8>(reduced);
        amrex::Real vtmax = amrex::get<9>(reduced);
        amrex::Real vzmax = amrex::get<10>(reduced);
        amrex::Real gammamax = amrex::get<11>(reduced);
        amrex::Real nnan = amrex::get<12>(reduced);
        amrex::Real ninf = amrex::get<13>(reduced);
        amrex::Real nout = amrex::get<14>(reduced);
        amrex::Real ncovered = amrex::get<15>(reduced);
        amrex::Real kinetic_sum = amrex::get<16>(reduced);
        amrex::Real vr_sum = amrex::get<17>(reduced);
        amrex::Real vt_sum = amrex::get<18>(reduced);
        amrex::Real vz_sum = amrex::get<19>(reduced);

        amrex::ParallelDescriptor::ReduceRealMin({rmin, zmin, vrmin, vtmin, vzmin, gammamin});
        amrex::ParallelDescriptor::ReduceRealMax({rmax, zmax, vrmax, vtmax, vzmax, gammamax});
        amrex::ParallelDescriptor::ReduceRealSum({nnan, ninf, nout, ncovered, kinetic_sum,
                                                  vr_sum, vt_sum, vz_sum});

        auto const nan_count = static_cast<amrex::Long>(std::llround(nnan));
        auto const inf_count = static_cast<amrex::Long>(std::llround(ninf));
        auto const out_count = static_cast<amrex::Long>(std::llround(nout));
        auto const covered_count = static_cast<amrex::Long>(std::llround(ncovered));

        total_nan += nan_count;
        total_inf += inf_count;
        total_out_of_domain += out_count;
        total_inside_covered += covered_count;

        if (full_particle_diag) {
            amrex::Print() << "HallRZ particle species=" << species_names[is]
                           << ": Np=" << np
                           << ", r_min=" << rmin << ", r_max=" << rmax
                           << ", z_min=" << zmin << ", z_max=" << zmax
                           << ", vr_min=" << vrmin << ", vr_max=" << vrmax
                           << ", vr_mean=" << vr_sum / static_cast<amrex::Real>(np)
                           << ", vtheta_min=" << vtmin << ", vtheta_max=" << vtmax
                           << ", vtheta_mean=" << vt_sum / static_cast<amrex::Real>(np)
                           << ", vz_min=" << vzmin << ", vz_max=" << vzmax
                           << ", vz_mean=" << vz_sum / static_cast<amrex::Real>(np)
                           << ", gamma_min=" << gammamin << ", gamma_max=" << gammamax
                           << ", Ktot_nonrel=" << kinetic_sum
                           << ", N_particle_NaN=" << nan_count
                           << ", N_particle_Inf=" << inf_count
                           << ", N_out_of_domain=" << out_count
                           << ", N_inside_covered_EB=" << covered_count
                           << "\n";
        }
    }

    amrex::Print() << "HallRZ particle summary: step=" << step
                   << ", Np_total=" << total_np
                   << ", N_particle_NaN=" << total_nan
                   << ", N_particle_Inf=" << total_inf
                   << ", N_out_of_domain=" << total_out_of_domain
                   << ", N_inside_covered_EB=" << total_inside_covered
                   << "\n";
}

amrex::Vector<int>
ReadHallRZAMRRequestedMCLs (int const max_level)
{
    amrex::ParmParse pp("warpx");
    amrex::Vector<int> mcls;
    bool const have_per_level = pp.queryarr("hall_rz_max_coarsening_levels", mcls);
    if (!have_per_level) {
        int scalar_mcl = 30;
        pp.query("hall_rz_max_coarsening_level", scalar_mcl);
        mcls.assign(max_level + 1, std::max(0, scalar_mcl));
    }
    if (static_cast<int>(mcls.size()) != max_level + 1) {
        amrex::Abort(
            "warpx.hall_rz_max_coarsening_levels must have one entry per HallRZ AMR level.");
    }
    for (int& mcl : mcls) {
        mcl = std::max(0, mcl);
    }
    return mcls;
}

amrex::Real
ReadHallRZAMRPhiInitWeight ()
{
    amrex::ParmParse pp("warpx");
    amrex::Real weight = amrex::Real(1.0);
    pp.query("hall_rz_amr_phi_init_weight", weight);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        weight >= amrex::Real(0.0) && weight <= amrex::Real(1.0),
        "warpx.hall_rz_amr_phi_init_weight must be in [0, 1].");
    return weight;
}

void
InterpolateHallRZPhiBetweenLevels (
    amrex::MultiFab const& crse_phi,
    amrex::MultiFab& fine_phi,
    amrex::Geometry const& crse_geom,
    amrex::Geometry const& fine_geom,
    amrex::IntVect const& refratio,
    amrex::Real const phi_init_weight,
    bool const fine_phi_history_valid,
    bool const do_single_precision_comms)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(crse_phi.nComp() == 1 && fine_phi.nComp() == 1,
        "HallRZ AMR phi interpolation expects one phi component.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(crse_phi.ixType().nodeCentered() &&
                                     fine_phi.ixType().nodeCentered(),
        "HallRZ AMR phi interpolation expects nodal phi MultiFabs.");

    amrex::BoxArray crse_ba = fine_phi.boxArray();
    crse_ba.coarsen(refratio);
    amrex::MultiFab crse_on_fine_dm(crse_ba, fine_phi.DistributionMap(), 1, 0);
    amrex::MultiFab crse_interp(fine_phi.boxArray(), fine_phi.DistributionMap(), 1, 0);

    ablastr::utils::communication::ParallelCopy(
        crse_on_fine_dm,
        crse_phi,
        0,
        0,
        1,
        amrex::IntVect(0),
        amrex::IntVect(0),
        do_single_precision_comms,
        crse_geom.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(crse_interp, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const fine = crse_interp.array(mfi);
        auto const crse = crse_on_fine_dm.const_array(mfi);
        ablastr::fields::details::PoissonInterpCPtoFP const interp(fine, crse, refratio);
        amrex::ParallelFor(mfi.tilebox(), interp);
    }

    amrex::BoxArray fine_cell_ba = amrex::convert(
        fine_phi.boxArray(), amrex::IntVect::TheCellVector());
    amrex::Box const fine_valid_cell = fine_cell_ba.minimalBox();
    amrex::Box const fine_domain = fine_geom.Domain();
    bool const force_rhi = fine_valid_cell.bigEnd(0) < fine_domain.bigEnd(0);
    bool const force_zhi = fine_valid_cell.bigEnd(1) < fine_domain.bigEnd(1);
    int const rhi_node = fine_valid_cell.bigEnd(0) + 1;
    int const zhi_node = fine_valid_cell.bigEnd(1) + 1;
    amrex::Real const w = fine_phi_history_valid ? phi_init_weight : amrex::Real(0.0);
    amrex::Real const one_minus_w = amrex::Real(1.0) - w;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(fine_phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const fine = fine_phi.array(mfi);
        auto const interp = crse_interp.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                bool const artificial_boundary =
                    (force_rhi && i == rhi_node) || (force_zhi && j == zhi_node);
                amrex::Real const interp_phi = interp(i,j,k);
                if (artificial_boundary || w == amrex::Real(0.0)) {
                    fine(i,j,k) = interp_phi;
                } else if (w != amrex::Real(1.0)) {
                    fine(i,j,k) = w * fine(i,j,k) + one_minus_w * interp_phi;
                }
            });
    }
}

std::string
HallRZFaceRole (bool const touches_physical, bool const is_low_side)
{
    if (touches_physical) {
        return is_low_side ? "physical-domain-lo" : "physical-domain-hi";
    }
    return is_low_side ? "interior-valid-region-lo" : "amr-artificial-hi-coarse-dirichlet";
}

void
PrintHallRZAMRDriverSetup (
    ablastr::fields::MultiLevelScalarField const& rho_fp,
    ablastr::fields::MultiLevelScalarField const& phi_fp,
    int const max_level)
{
    auto& warpx = WarpX::GetInstance();
    auto const params = HallRZPoissonSolver::ReadParameters();
    auto const requested_mcls = ReadHallRZAMRRequestedMCLs(max_level);
    amrex::Real const phi_init_weight = ReadHallRZAMRPhiInitWeight();
    int const step = warpx.getistep(0);

    amrex::Print() << "HallRZ AMR driver setup: step=" << step
                   << ", max_level=" << max_level
                   << ", solve_order=coarse-to-fine"
                   << ", level_by_level_poisson=stage3_level_by_level_enabled"
                   << ", phi_init_weight=" << phi_init_weight
                   << "\n";

    for (int lev = 0; lev <= max_level; ++lev) {
        amrex::Geometry const& geom = warpx.Geom(lev);
        amrex::Box const domain = geom.Domain();
        amrex::BoxArray const cell_ba =
            amrex::convert(phi_fp[lev]->boxArray(), amrex::IntVect::TheCellVector());
        amrex::Box const valid_region = cell_ba.minimalBox();
        bool const rlo_physical = valid_region.smallEnd(0) == domain.smallEnd(0);
        bool const rhi_physical = valid_region.bigEnd(0) == domain.bigEnd(0);
        bool const zlo_physical = valid_region.smallEnd(1) == domain.smallEnd(1);
        bool const zhi_physical = valid_region.bigEnd(1) == domain.bigEnd(1);

        amrex::Print() << "HallRZ AMR driver setup level=" << lev
                       << ": requested_mcl=" << requested_mcls[lev]
                       << ", actual_nmg_levels=reported_by_level_solve"
                       << ", rho_boxes=" << rho_fp[lev]->boxArray().size()
                       << ", phi_boxes=" << phi_fp[lev]->boxArray().size()
                       << ", cell_valid_region=" << valid_region
                       << ", domain=" << domain
                       << ", rlo_face=" << HallRZFaceRole(rlo_physical, true)
                       << ", rhi_face=" << HallRZFaceRole(rhi_physical, false)
                       << ", zlo_face=" << HallRZFaceRole(zlo_physical, true)
                       << ", zhi_face=" << HallRZFaceRole(zhi_physical, false)
                       << ", eb_boundary=HallRZ-EB-FVM-fluid-to-covered"
                       << ", internal_box_faces=decomposition-only-not-BC"
                       << "\n";
    }

    amrex::ignore_unused(params);
}

void
ValidateHallRZAMRBoundaryData (HallRZPoissonSolver::Params const& params, int const max_level)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !HallRZPoissonSolver::HasPythonRobinZHi() &&
        !HallRZPoissonSolver::HasPythonRobinRHi() &&
        !HallRZPoissonSolver::HasPythonRobinRLo() &&
        !HallRZPoissonSolver::HasPythonDirichletRLo() &&
        !HallRZPoissonSolver::HasPythonInletDirichlet(),
        "Stage-3.3 HallRZ AMR level-by-level solve supports per-level Python EB "
        "Neumann/Robin arrays only. Physical domain Python boundary arrays are still "
        "single-level data and are not supported on the AMR path.");

    if (params.eb_bc_mode == HallRZPoissonSolver::EBBCMode::Robin) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !HallRZPoissonSolver::HasPythonEBNeumann(),
            "warpx.hall_rz_eb_bc_mode=robin is mutually exclusive with "
            "hallrz.set_eb_neumann(...).");
        for (int lev = 0; lev <= max_level; ++lev) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                HallRZPoissonSolver::HasPythonEBRobin(lev),
                "HallRZ AMR EB Robin mode requires hallrz.set_eb_robin(a,b,f,lev=...) "
                "for every active AMR level.");
        }
    } else {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !HallRZPoissonSolver::HasPythonEBRobin(),
            "hallrz.set_eb_robin(a,b,f,...) is active but warpx.hall_rz_eb_bc_mode is "
            "not robin.");
        if (HallRZPoissonSolver::HasPythonEBNeumann()) {
            for (int lev = 0; lev <= max_level; ++lev) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    HallRZPoissonSolver::HasPythonEBNeumann(lev),
                    "HallRZ AMR Python EB Neumann data must be provided for every AMR "
                    "level, or cleared to use scalar/default EB Neumann data.");
            }
        }
    }
}

} // namespace

void LabFrameExplicitES::InitData() {
    auto & warpx = WarpX::GetInstance();
    m_poisson_boundary_handler->DefinePhiBCs(warpx.Geom(0));
}

void LabFrameExplicitES::ComputeSpaceChargeField (
    ablastr::fields::MultiFabRegister& fields,
    MultiParticleContainer& mpc,
    MultiFluidContainer* mfl,
    int max_level)
{
    using ablastr::fields::MultiLevelScalarField;
    using ablastr::fields::MultiLevelVectorField;
    using warpx::fields::FieldType;

    bool const skip_lev0_coarse_patch = true;

    const MultiLevelScalarField rho_fp = fields.get_mr_levels(FieldType::rho_fp, max_level);
    const MultiLevelScalarField rho_cp = fields.get_mr_levels(FieldType::rho_cp, max_level, skip_lev0_coarse_patch);
    const MultiLevelScalarField phi_fp = fields.get_mr_levels(FieldType::phi_fp, max_level);
    const MultiLevelVectorField Efield_fp = fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);

    mpc.DepositCharge(rho_fp, 0.0_rt);
    if (mfl) {
        const int lev = 0;
        mfl->DepositCharge(fields, *rho_fp[lev], lev);
    }

    // Apply filter, perform MPI exchange, interpolate across levels
    const Vector<std::unique_ptr<MultiFab> > rho_buf(num_levels);
    auto & warpx = WarpX::GetInstance();
    warpx.SyncRho( rho_fp, rho_cp, amrex::GetVecOfPtrs(rho_buf) );

    if (HallRZPoissonSolver::Enabled()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(max_level >= 0 && max_level <= 2,
            "HallRZ AMR currently supports amr.max_level=0, 1, or static 2.");
        if (max_level > 0) {
            ValidateHallRZAMRBoundaryData(HallRZPoissonSolver::ReadParameters(), max_level);
            PrintHallRZAMRDriverSetup(rho_fp, phi_fp, max_level);
            amrex::Vector<int> const requested_mcls = ReadHallRZAMRRequestedMCLs(max_level);
            amrex::Real const phi_init_weight = ReadHallRZAMRPhiInitWeight();
            for (int lev = 0; lev <= max_level; ++lev) {
                amrex::BoxArray cell_grids = amrex::convert(
                    phi_fp[lev]->boxArray(), amrex::IntVect::TheCellVector());
                HallRZPoissonSolver::ComputePhiAndE(
                    *rho_fp[lev],
                    *phi_fp[lev],
                    Efield_fp[lev],
                    warpx.Geom(lev),
                    cell_grids,
                    phi_fp[lev]->DistributionMap(),
                    warpx.fieldEBFactory(lev),
                    requested_mcls[lev],
                    lev);
                if (lev < max_level) {
                    InterpolateHallRZPhiBetweenLevels(
                        *phi_fp[lev],
                        *phi_fp[lev+1],
                        warpx.Geom(lev),
                        warpx.Geom(lev+1),
                        WarpX::RefRatio(lev),
                        phi_init_weight,
                        warpx.getistep(0) > 0,
                        WarpX::do_single_precision_comms);
                }
            }
            PrintHallRZParticleDiagnostics(
                mpc, HallRZPoissonSolver::ReadParameters(), warpx.Geom(max_level));
            return;
        }
        amrex::BoxArray cell_grids = amrex::convert(phi_fp[0]->boxArray(), amrex::IntVect::TheCellVector());
        HallRZPoissonSolver::ComputePhiAndE(*rho_fp[0], *phi_fp[0], Efield_fp[0],
                                            warpx.Geom(0), cell_grids,
                                            phi_fp[0]->DistributionMap(),
                                            warpx.fieldEBFactory(0),
                                            -1, 0);
        PrintHallRZParticleDiagnostics(mpc, HallRZPoissonSolver::ReadParameters(), warpx.Geom(0));
        return;
    }

#ifndef WARPX_DIM_RZ
    for (int lev = 0; lev < num_levels; lev++) {
        // Reflect density over PEC boundaries, if needed.
        warpx.ApplyRhofieldBoundary(lev, rho_fp[lev], PatchType::fine);
    }
#endif
    // beta is zero in lab frame
    // Todo: use simpler finite difference form with beta=0
    const std::array<Real, 3> beta = {0._rt};

    // set the boundary potentials appropriately
    setPhiBC(phi_fp, warpx.gett_new(0));

    // Compute the potential phi, by solving the Poisson equation
    if (IsPythonCallbackInstalled("poissonsolver")) {

        // Use the Python level solver (user specified)
        ExecutePythonCallback("poissonsolver");

    } else {

#if defined(WARPX_DIM_1D_Z)
        // Use the tridiag solver with 1D
        computePhiTriDiagonal(rho_fp, phi_fp);
#else
        // Use the AMREX MLMG or the FFT (IGF) solver otherwise
        computePhi(rho_fp, phi_fp, beta, self_fields_required_precision,
                   self_fields_absolute_tolerance, self_fields_max_iters,
                   self_fields_verbosity, is_igf_2d_slices, Efield_fp);
#endif

    }

    // Compute the electric field. Note that if an EB is used the electric
    // field will be calculated in the computePhi call.
    if (!EB::enabled()) { computeE( Efield_fp, phi_fp, beta ); }
    else {
        if (IsPythonCallbackInstalled("poissonsolver")) { computeE(Efield_fp, phi_fp, beta); }
    }
}

/* \brief Compute the potential by solving Poisson's equation with
          a 1D tridiagonal solve.

   \param[in] rho The charge density a given species
   \param[out] phi The potential to be computed by this function
*/
void LabFrameExplicitES::computePhiTriDiagonal (
    const ablastr::fields::MultiLevelScalarField& rho,
    const ablastr::fields::MultiLevelScalarField& phi)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(num_levels == 1,
    "The tridiagonal solver cannot be used with mesh refinement");

    auto field_boundary_lo0 = WarpX::field_boundary_lo[0];
    auto field_boundary_hi0 = WarpX::field_boundary_hi[0];

    if (field_boundary_lo0 == FieldBoundaryType::Periodic) {
        computePhiTriDiagonal_periodic(rho, phi);
        return;
    }

    const int lev = 0;
    auto & warpx = WarpX::GetInstance();

    const amrex::Real* dx = warpx.Geom(lev).CellSize();
    const amrex::Real xmin = warpx.Geom(lev).ProbLo(0);
    const amrex::Real xmax = warpx.Geom(lev).ProbHi(0);
    const int nx_full_domain = static_cast<int>( (xmax - xmin)/dx[0] + 0.5_rt );

    int nx_solve_min = 1;
    int nx_solve_max = nx_full_domain - 1;

    if (field_boundary_lo0 == FieldBoundaryType::Neumann) {
        // Solve for the point on the lower boundary
        nx_solve_min = 0;
    }
    if (field_boundary_hi0 == FieldBoundaryType::Neumann) {
        // Solve for the point on the upper boundary
        nx_solve_max = nx_full_domain;
    }

    // Create a 1-D MultiFab that covers all of x.
    // The tridiag solve will be done in this MultiFab and then copied out afterwards.
    const amrex::IntVect lo_full_domain(AMREX_D_DECL(0,0,0));
    const amrex::IntVect hi_full_domain(AMREX_D_DECL(nx_full_domain,0,0));
    const amrex::Box box_full_domain_node(lo_full_domain, hi_full_domain, amrex::IntVect::TheNodeVector());
    const BoxArray ba_full_domain_node(box_full_domain_node);
    const amrex::Vector<int> pmap = {0}; // The data will only be on processor 0
    const amrex::DistributionMapping dm_full_domain(pmap);

    // Put the data in the pinned arena since the tridiag solver will be done on the CPU, but have
    // the data readily accessible from the GPU.
    auto phi1d_mf = MultiFab(ba_full_domain_node, dm_full_domain, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));
    auto zwork1d_mf = MultiFab(ba_full_domain_node, dm_full_domain, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));
    auto rho1d_mf = MultiFab(ba_full_domain_node, dm_full_domain, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));

    if (field_boundary_lo0 == FieldBoundaryType::PEC || field_boundary_hi0 == FieldBoundaryType::PEC) {
        // Copy from phi to get the boundary values
        phi1d_mf.ParallelCopy(*phi[lev], 0, 0, 1);
    }
    rho1d_mf.ParallelCopy(*rho[lev], 0, 0, 1);

    // Multiplier on the charge density
    const amrex::Real norm = dx[0]*dx[0]/PhysConst::epsilon_0;
    rho1d_mf.mult(norm);

    // Use the MFIter loop since when parallel, only process zero has a FAB.
    // This skips the loop on all other processors.
    for (MFIter mfi(phi1d_mf); mfi.isValid(); ++mfi) {

        const auto& phi1d_arr = phi1d_mf[mfi].array();
        const auto& zwork1d_arr = zwork1d_mf[mfi].array();
        const auto& rho1d_arr = rho1d_mf[mfi].array();

        // The loops are always performed on the CPU

        amrex::Real diag = 2._rt;

        // The initial values depend on the boundary condition
        if (field_boundary_lo0 == FieldBoundaryType::PEC) {

            phi1d_arr(1,0,0) = (phi1d_arr(0,0,0) + rho1d_arr(1,0,0))/diag;

        } else if (field_boundary_lo0 == FieldBoundaryType::Neumann) {

            // Neumann boundary condition
            phi1d_arr(0,0,0) = rho1d_arr(0,0,0)/diag;

            zwork1d_arr(1,0,0) = 2._rt/diag;
            diag = 2._rt - zwork1d_arr(1,0,0);
            phi1d_arr(1,0,0) = (rho1d_arr(1,0,0) - (-1._rt)*phi1d_arr(1-1,0,0))/diag;

        }

        // Loop upward, calculating the Gaussian elimination multipliers and right hand sides
        for (int i_up = 2 ; i_up < nx_solve_max ; i_up++) {

            zwork1d_arr(i_up,0,0) = 1._rt/diag;
            diag = 2._rt - zwork1d_arr(i_up,0,0);
            phi1d_arr(i_up,0,0) = (rho1d_arr(i_up,0,0) - (-1._rt)*phi1d_arr(i_up-1,0,0))/diag;

        }

        // The last value depend on the boundary condition
        if (field_boundary_hi0 == FieldBoundaryType::PEC) {

            int const nxm1 = nx_full_domain - 1;
            zwork1d_arr(nxm1,0,0) = 1._rt/diag;
            diag = 2._rt - zwork1d_arr(nxm1,0,0);
            phi1d_arr(nxm1,0,0) = (phi1d_arr(nxm1+1,0,0) + rho1d_arr(nxm1,0,0) - (-1._rt)*phi1d_arr(nxm1-1,0,0))/diag;

        } else if (field_boundary_hi0 == FieldBoundaryType::Neumann) {

            // Neumann boundary condition
            zwork1d_arr(nx_full_domain,0,0) = 1._rt/diag;
            diag = 2._rt - 2._rt*zwork1d_arr(nx_full_domain,0,0);
            if (diag == 0._rt) {
                // This happens if the lower boundary is also Neumann.
                // It this case, the potential is relative to an arbitrary constant,
                // so set the upper boundary to zero to force a value.
                phi1d_arr(nx_full_domain,0,0) = 0.;
            } else {
                phi1d_arr(nx_full_domain,0,0) = (rho1d_arr(nx_full_domain,0,0) - (-1._rt)*phi1d_arr(nx_full_domain-1,0,0))/diag;
            }

        }


        for (int i_down = nx_solve_max-1 ; i_down >= nx_solve_min ; i_down--) {
            phi1d_arr(i_down,0,0) = phi1d_arr(i_down,0,0) + zwork1d_arr(i_down+1,0,0)*phi1d_arr(i_down+1,0,0);
        }

    }

    // Copy phi1d to phi
    phi[lev]->ParallelCopy(phi1d_mf, 0, 0, 1);
}

/* \brief Compute the potential by solving Poisson's equation
 *        with periodic boundaries using
          a 1D tridiagonal solve.
          This makes use of the Sherman–Morrison formula.
          The code is based on the code given in the Wikipedia page,
          https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm#Variants.

   \param[in] rho The charge density a given species
   \param[out] phi The potential to be computed by this function
*/
void LabFrameExplicitES::computePhiTriDiagonal_periodic (
    const ablastr::fields::MultiLevelScalarField& rho,
    const ablastr::fields::MultiLevelScalarField& phi)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(num_levels == 1,
    "The tridiagonal solver cannot be used with mesh refinement");

    const int lev = 0;
    auto & warpx = WarpX::GetInstance();

    const amrex::Real* dx = warpx.Geom(lev).CellSize();
    const amrex::Real xmin = warpx.Geom(lev).ProbLo(0);
    const amrex::Real xmax = warpx.Geom(lev).ProbHi(0);
    const int nx = static_cast<int>( (xmax - xmin)/dx[0] + 0.5_rt );


    // Create a 1-D MultiFab that covers all of x.
    // The tridiag solve will be done in this MultiFab and then copied out afterwards.
    const amrex::IntVect lo_full_domain(AMREX_D_DECL(0,0,0));
    const amrex::IntVect hi_full_domain(AMREX_D_DECL(nx,0,0));
    const amrex::Box box_full_domain_node(lo_full_domain, hi_full_domain, amrex::IntVect::TheNodeVector());
    const BoxArray ba_full_domain_node(box_full_domain_node);
    const amrex::Vector<int> pmap = {0}; // The data will only be on processor 0
    const amrex::DistributionMapping dm_full_domain(pmap);

    // Put the data in the pinned arena since the tridiag solver will be done on the CPU, but have
    // the data readily accessible from the GPU.
    auto phi1d_mf = MultiFab(ba_full_domain_node, dm_full_domain, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));

    // Work arrays
    auto cmod_mf = MultiFab(ba_full_domain_node, dm_full_domain, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));
    auto u_mf = MultiFab(ba_full_domain_node, dm_full_domain, 1, 0, MFInfo().SetArena(The_Pinned_Arena()));

    // Copy rho into phi1d_mf to start
    phi1d_mf.ParallelCopy(*rho[lev], 0, 0, 1);

    // Multiplier on the charge density
    const amrex::Real norm = dx[0]*dx[0]/PhysConst::epsilon_0;
    phi1d_mf.mult(norm);

    // Use the MFIter loop since when parallel, only process zero has a FAB.
    // This skips the loop on all other processors.
    for (MFIter mfi(phi1d_mf); mfi.isValid(); ++mfi) {

        const auto& x = phi1d_mf[mfi].array();
        const auto& cmod = cmod_mf[mfi].array();
        const auto& u = u_mf[mfi].array();

        // The loops are always performed on the CPU

        {

        // This code is adapted from the Wikipedia page on the Tridiagonal matrix algorithm
        // https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm#Variants.
        // Licensed under CC BY-SA 4.0
        // Modifications: The a, b, and c inputs are replaced with the fixed values.

        const amrex::Real alpha = -1.0_rt;
        const amrex::Real beta = -1.0_rt;

        /* arbitrary, but chosen such that division by zero is avoided */
        const amrex::Real gamma = -2.0_rt;

        cmod(0,0,0) = -1.0_rt / (2.0_rt - gamma);
        u(0,0,0) = gamma / (2.0_rt - gamma);
        x(0,0,0) /= (2.0_rt - gamma);

        /* loop from 1 to nx - 2 inclusive */
        for (int ix = 1; ix + 1 < nx; ix++) {
            const amrex::Real m = 1.00_rt / (2.0_rt - -1.0_rt * cmod(ix - 1,0,0));
            cmod(ix,0,0) = -1.0_rt * m;
            u(ix,0,0) = (0.0f  - -1.0_rt * u(ix - 1,0,0)) * m;
            x(ix,0,0) = (x(ix,0,0) - -1.0_rt * x(ix - 1,0,0)) * m;
        }

        /* handle nx - 1 */
        const amrex::Real m = 1.00_rt / (2.0_rt - alpha * beta / gamma - -1.0_rt * cmod(nx - 2,0,0));
        u(nx - 1,0,0) = (alpha    - -1.0_rt * u(nx - 2,0,0)) * m;
        x(nx - 1,0,0) = (x(nx - 1,0,0) - -1.0_rt * x(nx - 2,0,0)) * m;

        /* loop from nx - 2 to 0 inclusive */
        for (int ix = nx - 2; ix >= 0; ix--) {
            u(ix,0,0) -= cmod(ix,0,0) * u(ix + 1,0,0);
            x(ix,0,0) -= cmod(ix,0,0) * x(ix + 1,0,0);
        }

        const amrex::Real fact = (x(0,0,0) + x(nx - 1,0,0) * alpha / gamma) / (1.00 + u(0,0,0) + u(nx - 1,0,0) * alpha / gamma);

        /* loop from 0 to nx - 1 inclusive */
        for (int ix = 0; ix < nx; ix++)
            x(ix,0,0) -= fact * u(ix,0,0);

        }

        x(nx,0,0) = x(0,0,0);

        // In a test case, this was giving an relative residual of around 1.e-10.
        // A dozen or so SOR iterations could improve that by a factor of 10.
        // Is it worth it?

    }

    // Copy phi1d to phi
    phi[lev]->ParallelCopy(phi1d_mf, 0, 0, 1);
}
