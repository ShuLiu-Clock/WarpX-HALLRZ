/* Copyright 2021 Lorenzo Giacomel
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarpX.H"

#include "EmbeddedBoundary/Enabled.H"
#ifdef AMREX_USE_EB
#  include "Fields.H"
#  include "Utils/Parser/ParserUtils.H"
#  include "Utils/TextMsg.H"

#   include <AMReX_BLProfiler.H>
#   include <AMReX_BoxArray.H>
#   include <AMReX_Config.H>
#   include <AMReX_EB2.H>
#   include <AMReX_EB2_GeometryShop.H>
#   include <AMReX_EB2_IF_Base.H>
#   include <AMReX_EB2_IF_Box.H>
#   include <AMReX_EB2_IF_Union.H>
#   include <AMReX_EB_utils.H>
#   include <AMReX_GpuQualifiers.H>
#   include <AMReX_ParmParse.H>
#   include <AMReX_REAL.H>
#   include <AMReX_SPACE.H>

#  include <cstdlib>
#  include <algorithm>
#  include <cmath>
#  include <limits>
#  include <string>

using namespace ablastr::fields;

#endif

#ifdef AMREX_USE_EB
namespace {
    class ParserIF
        : public amrex::GPUable
    {
    public:
        ParserIF (const amrex::ParserExecutor<3>& a_parser)
            : m_parser(a_parser)
            {}

        ParserIF (const ParserIF& rhs) noexcept = default;
        ParserIF (ParserIF&& rhs) noexcept = default;
        ParserIF& operator= (const ParserIF& rhs) = delete;
        ParserIF& operator= (ParserIF&& rhs) = delete;

        ~ParserIF() = default;

        AMREX_GPU_HOST_DEVICE inline
        amrex::Real operator() (AMREX_D_DECL(amrex::Real x, amrex::Real y,
                                             amrex::Real z)) const noexcept {
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            return m_parser(x,amrex::Real(0.0),y);
#else
            return m_parser(x,y,z);
#endif
        }

        inline amrex::Real operator() (const amrex::RealArray& p) const noexcept {
            return this->operator()(AMREX_D_DECL(p[0],p[1],p[2]));
        }

    private:
        amrex::ParserExecutor<3> m_parser; //! function parser with three arguments (x,y,z)
    };
}
#endif

void
WarpX::InitEB ()
{
    if (!EB::enabled()) {
        throw std::runtime_error("InitEB only works when EBs are enabled at runtime");
    }

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("EBs only implemented in 2D and 3D");
#endif

#ifdef AMREX_USE_EB
    BL_PROFILE("InitEB");

    const amrex::ParmParse pp_warpx("warpx");
    bool hall_rz_enabled = false;
    pp_warpx.query("hall_rz_enable", hall_rz_enabled);

    if (hall_rz_enabled) {
#if !defined(WARPX_DIM_RZ)
        WARPX_ABORT_WITH_MESSAGE("warpx.hall_rz_enable requires a RZ build.");
#else
        std::string impf_check;
        amrex::ParmParse pp_eb2_check("eb2");
        bool const has_eb_implicit_function = pp_warpx.query("eb_implicit_function", impf_check);
        bool const has_eb2_geom_type = pp_eb2_check.contains("geom_type");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !has_eb_implicit_function && !has_eb2_geom_type,
            "warpx.hall_rz_enable must be the only EB geometry source. "
            "Do not also set warpx.eb_implicit_function or eb2.geom_type.");

        amrex::Real lob = 0.0;
        amrex::Real hib = 0.0;
        amrex::Real out = 0.0;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pp_warpx.query("hall_rz_lob", lob),
                                         "warpx.hall_rz_lob is required when hall_rz_enable=1.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pp_warpx.query("hall_rz_hib", hib),
                                         "warpx.hall_rz_hib is required when hall_rz_enable=1.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pp_warpx.query("hall_rz_out", out),
                                         "warpx.hall_rz_out is required when hall_rz_enable=1.");

        auto const& eb_geom = Geom(maxLevel());
        amrex::Real const rlo = eb_geom.ProbLo(0);
        amrex::Real const rhi = eb_geom.ProbHi(0);
        amrex::Real const zlo = eb_geom.ProbLo(1);
        amrex::Real const zhi = eb_geom.ProbHi(1);

        amrex::Real align_tol = amrex::Real(1.0e-10);
        pp_warpx.query("hall_rz_align_tol", align_tol);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rlo >= amrex::Real(0.0),
                                         "HallRZ requires r_min >= 0.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rlo <= lob + align_tol &&
                                         lob < hib && hib < rhi,
                                         "HallRZ requires r_min <= lob < hib < r_max.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(zlo < out && out < zhi,
                                         "HallRZ requires z_min < out < z_max.");

        auto is_aligned = [align_tol] (amrex::Real x, amrex::Real xlo, amrex::Real dx) noexcept {
            amrex::Real const idx = (x - xlo) / dx;
            return std::abs(idx - std::round(idx)) < align_tol;
        };
        amrex::Real const* dx = eb_geom.CellSize();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(is_aligned(lob, rlo, dx[0]) &&
                                         is_aligned(hib, rlo, dx[0]) &&
                                         is_aligned(out, zlo, dx[1]),
                                         "HallRZ lob/hib/out must lie exactly on cell faces.");
        auto snap_to_face = [] (amrex::Real x, amrex::Real xlo, amrex::Real h) noexcept {
            return xlo + std::round((x - xlo) / h) * h;
        };
        amrex::Real const lob_face = snap_to_face(lob, rlo, dx[0]);
        amrex::Real const hib_face = snap_to_face(hib, rlo, dx[0]);
        amrex::Real const out_face = snap_to_face(out, zlo, dx[1]);

        amrex::Real delta_box = std::max(rhi-rlo, zhi-zlo);
        pp_warpx.query("hall_rz_box_delta", delta_box);
        delta_box = std::max(delta_box, amrex::Real(10.0) * std::max(dx[0], dx[1]));

        amrex::EB2::BoxIF high(
            {AMREX_D_DECL(hib_face,      zlo-delta_box, amrex::Real(-1.0))},
            {AMREX_D_DECL(rhi+delta_box, out_face,       amrex::Real( 1.0))}, false);
        bool const low_block_exists = lob_face > rlo + align_tol;
        if (low_block_exists) {
            amrex::EB2::BoxIF low(
                {AMREX_D_DECL(rlo-delta_box, zlo-delta_box, amrex::Real(-1.0))},
                {AMREX_D_DECL(lob_face,      out_face,       amrex::Real( 1.0))}, false);
            auto hall_body = amrex::EB2::makeUnion(low, high);
            auto hall_shop = amrex::EB2::makeShop(hall_body);
            amrex::EB2::Build(hall_shop, Geom(maxLevel()), maxLevel(), maxLevel()+20);
        } else {
            auto hall_shop = amrex::EB2::makeShop(high);
            amrex::EB2::Build(hall_shop, Geom(maxLevel()), maxLevel(), maxLevel()+20);
        }

        if (Verbose()) {
            amrex::Print() << "HallRZ EB initialized by WarpX::InitEB: lob=" << lob
                           << ", hib=" << hib << ", out=" << out
                           << ", lob_face=" << lob_face
                           << ", hib_face=" << hib_face
                           << ", out_face=" << out_face
                           << ", low_block_exists=" << low_block_exists
                           << ", delta_box=" << delta_box << "\n";
        }
        return;
#endif
    }

    std::string impf;
    pp_warpx.query("eb_implicit_function", impf);
    if (! impf.empty()) {
        auto eb_if_parser = utils::parser::makeParser(impf, {"x", "y", "z"});
        ParserIF const pif(eb_if_parser.compile<3>());
        auto gshop = amrex::EB2::makeShop(pif, eb_if_parser);
         // The last argument of amrex::EB2::Build is the maximum coarsening level
         // to which amrex should try to coarsen the EB.  It will stop after coarsening
         // as much as it can, if it cannot coarsen to that level.  Here we use a big
         // number (e.g., maxLevel()+20) for multigrid solvers.  Because the coarse
         // level has only 1/8 of the cells on the fine level, the memory usage should
         // not be an issue.
        amrex::EB2::Build(gshop, Geom(maxLevel()), maxLevel(), maxLevel()+20);
    } else {
        amrex::ParmParse pp_eb2("eb2");
        if (!pp_eb2.contains("geom_type")) {
            std::string const geom_type = "all_regular";
            pp_eb2.add("geom_type", geom_type); // use all_regular by default
        }
        // See the comment above on amrex::EB2::Build for the hard-wired number 20.
        amrex::EB2::Build(Geom(maxLevel()), maxLevel(), maxLevel()+20);
    }
#endif
}

void
WarpX::ComputeDistanceToEB ()
{
    if (!EB::enabled()) {
        throw std::runtime_error("ComputeDistanceToEB only works when EBs are enabled at runtime");
    }
#ifdef AMREX_USE_EB
    BL_PROFILE("ComputeDistanceToEB");
    using warpx::fields::FieldType;
    const amrex::EB2::IndexSpace& eb_is = amrex::EB2::IndexSpace::top();
    for (int lev=0; lev<=maxLevel(); lev++) {
        const amrex::EB2::Level& eb_level = eb_is.getLevel(Geom(lev));
        auto const eb_fact = fieldEBFactory(lev);
        amrex::FillSignedDistance(*m_fields.get(FieldType::distance_to_eb, lev), eb_level, eb_fact, 1);
    }
#endif
}
