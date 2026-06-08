/* Copyright 2021-2023 The WarpX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyWarpX.H"
#include "callbacks.H"

#if defined(WARPX_DIM_RZ)
#   include <FieldSolver/ElectrostaticSolvers/HallRZPoissonSolver.H>
#endif

#include <WarpX.H>  // todo: move this out to Python/WarpX.cpp
#include <Utils/WarpXUtil.H>  // todo: move to its own Python/Utils.cpp
#include <Utils/WarpXVersion.H>
#include <Initialization/WarpXAMReXInit.H>

#include <tuple>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#define CONCAT_NAME(PRE, SUF) PRE ## SUF

// see: CMakeLists.txt, setup.py and __init__.py
#if defined(WARPX_DIM_1D_Z)
#  define PYWARPX_MODULE_NAME CONCAT_NAME(warpx_pybind_, 1d)
#elif defined(WARPX_DIM_XZ)
#  define PYWARPX_MODULE_NAME CONCAT_NAME(warpx_pybind_, 2d)
#elif defined(WARPX_DIM_RZ)
#  define PYWARPX_MODULE_NAME CONCAT_NAME(warpx_pybind_, rz)
#elif defined(WARPX_DIM_RCYLINDER)
#  define PYWARPX_MODULE_NAME CONCAT_NAME(warpx_pybind_, rcylinder)
#elif defined(WARPX_DIM_RSPHERE)
#  define PYWARPX_MODULE_NAME CONCAT_NAME(warpx_pybind_, rsphere)
#elif defined(WARPX_DIM_3D)
#  define PYWARPX_MODULE_NAME CONCAT_NAME(warpx_pybind_, 3d)
#endif

//using namespace warpx;


// forward declarations of exposed classes
void init_BoundaryBufferParIter (py::module&);
void init_MultiParticleContainer (py::module&);
void init_MultiFabRegister (py::module&);
void init_ParticleBoundaryBuffer (py::module&);
void init_WarpXParIter (py::module&);
void init_WarpXParticleContainer (py::module&);
void init_WarpX(py::module&);

PYBIND11_MODULE(PYWARPX_MODULE_NAME, m) {
    // make sure AMReX types are known
#if defined(WARPX_DIM_3D)
    auto amr = py::module::import("amrex.space3d");
#elif defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    auto amr = py::module::import("amrex.space1d");
#else
    auto amr = py::module::import("amrex.space2d");
#endif

    m.doc() = R"pbdoc(
            warpx_pybind
            --------------
            .. currentmodule:: warpx_pybind_(1d|2d|3d|rz|rcylinder|rsphere)

            .. autosummary::
               :toctree: _generate
               WarpX
    )pbdoc";

    // note: order from parent to child classes
    init_MultiFabRegister(m);
    init_WarpXParticleContainer(m);
    init_WarpXParIter(m);
    init_BoundaryBufferParIter(m);
    init_ParticleBoundaryBuffer(m);
    init_MultiParticleContainer(m);
    init_WarpX(m);

    // expose our amrex module
    m.attr("amr") = amr;

    // API runtime version
    //   note PEP-440 syntax: x.y.zaN but x.y.z.devN
#ifdef PYWARPX_VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(PYWARPX_VERSION_INFO);
#else
    // note: not necessarily PEP-440 compliant
    m.attr("__version__") = WarpX::Version();
#endif

    // authors
    m.attr("__author__") =
        "Jean-Luc Vay, David P. Grote, Maxence Thevenet, Remi Lehe, Andrew Myers, Weiqun Zhang, Axel Huebl, et al.";

    // API runtime build-time feature variants
    // m.attr("variants") = warpx::getVariants();
    // TODO allow to query runtime versions of all dependencies

    // license SPDX identifier
    m.attr("__license__") = "BSD-3-Clause-LBNL";

    // TODO broken numpy if not at least v1.15.0: raise warning
    // auto numpy = py::module::import("numpy");
    // auto npversion = numpy.attr("__version__");
    // std::cout << "numpy version: " << py::str(npversion) << "\n";

    m.def("amrex_init",
        [](const py::list args) {
            amrex::Vector<std::string> cargs;
            amrex::Vector<char*> argv;

            // Populate the "command line"
            for (const auto& v: args)
                cargs.push_back(v.cast<std::string>());
            for (auto& v: cargs)
                argv.push_back(&v[0]);
            int argc = argv.size();

            // note: +1 since there is an extra char-string array element,
            //       that ANSII C requires to be a simple NULL entry
            //       https://stackoverflow.com/a/39096006/2719194
            argv.push_back(NULL);
            char** tmp = argv.data();

            const bool build_parm_parse = (cargs.size() > 1);
            // TODO: handle version with MPI
            return warpx::initialization::amrex_init(argc, tmp, build_parm_parse);
        }, py::return_value_policy::reference,
        "Initialize AMReX library");
    m.def("amrex_finalize", [] () { amrex::Finalize(); },
        "Close out the amrex related data");

    // Expose functions to get the processor number
    m.def("getNProcs", [](){return amrex::ParallelDescriptor::NProcs();} );
    m.def("getMyProc", [](){return amrex::ParallelDescriptor::MyProc();} );

    // Expose the python callback function installation and removal functions
    m.def("add_python_callback", &InstallPythonCallback);
    m.def("remove_python_callback", &ClearPythonCallback);
    m.def("execute_python_callback", &ExecutePythonCallback, py::arg("name"));

#if defined(WARPX_DIM_RZ)
    m.def("hallrz_set_eb_neumann",
        [](py::array const& arr, int const lev)
        {
            if (arr.ndim() != 2) {
                throw std::runtime_error("hallrz.set_eb_neumann expects a 2D NumPy array with shape (Nr,Nz).");
            }
            if (!arr.dtype().is(py::dtype::of<amrex::Real>())) {
                throw std::runtime_error("hallrz.set_eb_neumann expects dtype float64.");
            }
            if ((arr.flags() & py::array::c_style) == 0) {
                throw std::runtime_error("hallrz.set_eb_neumann expects a C-contiguous array.");
            }

            int const nr = static_cast<int>(arr.shape(0));
            int const nz = static_cast<int>(arr.shape(1));
            auto const g = py::array_t<amrex::Real>(arr).unchecked<2>();
            std::vector<amrex::Real> data(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz));
            for (int i = 0; i < nr; ++i) {
                for (int j = 0; j < nz; ++j) {
                    data[static_cast<std::size_t>(i) * static_cast<std::size_t>(nz) + static_cast<std::size_t>(j)] = g(i,j);
                }
            }
            HallRZPoissonSolver::SetPythonEBNeumann(nr, nz, std::move(data), lev);
        },
        py::arg("g_eb"), py::arg("lev") = 0,
        "Set persistent HallRZ cell-centered EB Neumann data for one AMR level.");

    m.def("hallrz_clear_eb_neumann",
        [](int const lev) { HallRZPoissonSolver::ClearPythonEBNeumann(lev); },
        py::arg("lev") = -1,
        "Clear persistent HallRZ Python EB Neumann data. lev=-1 clears all levels.");

    m.def("hallrz_has_eb_neumann",
        [](int const lev) { return HallRZPoissonSolver::HasPythonEBNeumann(lev); },
        py::arg("lev") = -1,
        "Return True if persistent HallRZ Python EB Neumann data is active.");

    auto copy_2d_real_array = [] (py::array const& arr, char const* name)
    {
        if (arr.ndim() != 2) {
            throw std::runtime_error(std::string(name) + " expects a 2D NumPy array with shape (Nr,Nz).");
        }
        if (!arr.dtype().is(py::dtype::of<amrex::Real>())) {
            throw std::runtime_error(std::string(name) + " expects dtype float64.");
        }
        if ((arr.flags() & py::array::c_style) == 0) {
            throw std::runtime_error(std::string(name) + " expects a C-contiguous array.");
        }
        int const nr = static_cast<int>(arr.shape(0));
        int const nz = static_cast<int>(arr.shape(1));
        auto const a = py::array_t<amrex::Real>(arr).unchecked<2>();
        std::vector<amrex::Real> data(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz));
        for (int i = 0; i < nr; ++i) {
            for (int j = 0; j < nz; ++j) {
                data[static_cast<std::size_t>(i) * static_cast<std::size_t>(nz) + static_cast<std::size_t>(j)] = a(i,j);
            }
        }
        return std::tuple<int, int, std::vector<amrex::Real>>(nr, nz, std::move(data));
    };

    m.def("hallrz_set_eb_robin",
        [copy_2d_real_array](py::array const& a, py::array const& b, py::array const& f,
                             int const lev)
        {
            auto [nr, nz, avec] = copy_2d_real_array(a, "hallrz.set_eb_robin(a)");
            auto [bnr, bnz, bvec] = copy_2d_real_array(b, "hallrz.set_eb_robin(b)");
            auto [fnr, fnz, fvec] = copy_2d_real_array(f, "hallrz.set_eb_robin(f)");
            if (bnr != nr || bnz != nz || fnr != nr || fnz != nz) {
                throw std::runtime_error("hallrz.set_eb_robin expects a, b, and f to have the same shape.");
            }
            HallRZPoissonSolver::SetPythonEBRobin(
                nr, nz, std::move(avec), std::move(bvec), std::move(fvec), lev);
        },
        py::arg("a"), py::arg("b"), py::arg("f"), py::arg("lev") = 0,
        "Set persistent HallRZ EB-FVM Robin data for one AMR level.");

    m.def("hallrz_clear_eb_robin",
        [](int const lev) { HallRZPoissonSolver::ClearPythonEBRobin(lev); },
        py::arg("lev") = -1,
        "Clear persistent HallRZ Python EB Robin data. lev=-1 clears all levels.");

    m.def("hallrz_has_eb_robin",
        [](int const lev) { return HallRZPoissonSolver::HasPythonEBRobin(lev); },
        py::arg("lev") = -1,
        "Return True if persistent HallRZ Python EB Robin data is active.");

    auto copy_1d_real_array = [] (py::array const& arr, char const* name)
    {
        if (arr.ndim() != 1) {
            throw std::runtime_error(std::string(name) + " expects a 1D NumPy array.");
        }
        if (!arr.dtype().is(py::dtype::of<amrex::Real>())) {
            throw std::runtime_error(std::string(name) + " expects dtype float64.");
        }
        if ((arr.flags() & py::array::c_style) == 0) {
            throw std::runtime_error(std::string(name) + " expects a C-contiguous array.");
        }
        int const n = static_cast<int>(arr.shape(0));
        auto const a = py::array_t<amrex::Real>(arr).unchecked<1>();
        std::vector<amrex::Real> data(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            data[static_cast<std::size_t>(i)] = a(i);
        }
        return data;
    };

    m.def("hallrz_set_robin_zhi",
        [copy_1d_real_array](py::array const& a, py::array const& b, py::array const& f)
        {
            auto avec = copy_1d_real_array(a, "hallrz.set_robin_zhi(a)");
            int const n = static_cast<int>(avec.size());
            auto bvec = copy_1d_real_array(b, "hallrz.set_robin_zhi(b)");
            auto fvec = copy_1d_real_array(f, "hallrz.set_robin_zhi(f)");
            if (static_cast<int>(bvec.size()) != n || static_cast<int>(fvec.size()) != n) {
                throw std::runtime_error("hallrz.set_robin_zhi expects a, b, and f to have the same shape.");
            }
            HallRZPoissonSolver::SetPythonRobinZHi(n, std::move(avec), std::move(bvec), std::move(fvec));
        },
        py::arg("a"), py::arg("b"), py::arg("f"),
        "Set persistent HallRZ z-hi Robin data from float64 C-contiguous arrays with shape (Nr+1).");

    m.def("hallrz_clear_robin_zhi",
        []() { HallRZPoissonSolver::ClearPythonRobinZHi(); },
        "Clear persistent HallRZ Python z-hi Robin data.");

    m.def("hallrz_has_robin_zhi",
        []() { return HallRZPoissonSolver::HasPythonRobinZHi(); },
        "Return True if persistent HallRZ Python z-hi Robin data is active.");

    m.def("hallrz_set_robin_rhi",
        [copy_1d_real_array](py::array const& a, py::array const& b, py::array const& f)
        {
            auto avec = copy_1d_real_array(a, "hallrz.set_robin_rhi(a)");
            int const n = static_cast<int>(avec.size());
            auto bvec = copy_1d_real_array(b, "hallrz.set_robin_rhi(b)");
            auto fvec = copy_1d_real_array(f, "hallrz.set_robin_rhi(f)");
            if (static_cast<int>(bvec.size()) != n || static_cast<int>(fvec.size()) != n) {
                throw std::runtime_error("hallrz.set_robin_rhi expects a, b, and f to have the same shape.");
            }
            HallRZPoissonSolver::SetPythonRobinRHi(n, std::move(avec), std::move(bvec), std::move(fvec));
        },
        py::arg("a"), py::arg("b"), py::arg("f"),
        "Set persistent HallRZ r-hi Robin data from float64 C-contiguous arrays with shape (Nz+1).");

    m.def("hallrz_clear_robin_rhi",
        []() { HallRZPoissonSolver::ClearPythonRobinRHi(); },
        "Clear persistent HallRZ Python r-hi Robin data.");

    m.def("hallrz_has_robin_rhi",
        []() { return HallRZPoissonSolver::HasPythonRobinRHi(); },
        "Return True if persistent HallRZ Python r-hi Robin data is active.");

    m.def("hallrz_set_robin_rlo",
        [copy_1d_real_array](py::array const& a, py::array const& b, py::array const& f)
        {
            auto avec = copy_1d_real_array(a, "hallrz.set_robin_rlo(a)");
            int const n = static_cast<int>(avec.size());
            auto bvec = copy_1d_real_array(b, "hallrz.set_robin_rlo(b)");
            auto fvec = copy_1d_real_array(f, "hallrz.set_robin_rlo(f)");
            if (static_cast<int>(bvec.size()) != n || static_cast<int>(fvec.size()) != n) {
                throw std::runtime_error("hallrz.set_robin_rlo expects a, b, and f to have the same shape.");
            }
            HallRZPoissonSolver::SetPythonRobinRLo(n, std::move(avec), std::move(bvec), std::move(fvec));
        },
        py::arg("a"), py::arg("b"), py::arg("f"),
        "Set persistent HallRZ r-lo Robin data from float64 C-contiguous arrays with shape (Nz+1).");

    m.def("hallrz_clear_robin_rlo",
        []() { HallRZPoissonSolver::ClearPythonRobinRLo(); },
        "Clear persistent HallRZ Python r-lo Robin data.");

    m.def("hallrz_has_robin_rlo",
        []() { return HallRZPoissonSolver::HasPythonRobinRLo(); },
        "Return True if persistent HallRZ Python r-lo Robin data is active.");

    m.def("hallrz_set_dirichlet_rlo",
        [copy_1d_real_array](py::array const& phi)
        {
            auto data = copy_1d_real_array(phi, "hallrz.set_dirichlet_rlo(phi)");
            int const n = static_cast<int>(data.size());
            HallRZPoissonSolver::SetPythonDirichletRLo(n, std::move(data));
        },
        py::arg("phi"),
        "Set persistent HallRZ r-lo Dirichlet data from a float64 C-contiguous array with shape (Nz+1).");

    m.def("hallrz_clear_dirichlet_rlo",
        []() { HallRZPoissonSolver::ClearPythonDirichletRLo(); },
        "Clear persistent HallRZ Python r-lo Dirichlet data.");

    m.def("hallrz_has_dirichlet_rlo",
        []() { return HallRZPoissonSolver::HasPythonDirichletRLo(); },
        "Return True if persistent HallRZ Python r-lo Dirichlet data is active.");

    m.def("hallrz_set_inlet_dirichlet",
        [copy_1d_real_array](py::array const& phi)
        {
            auto data = copy_1d_real_array(phi, "hallrz.set_inlet_dirichlet(phi)");
            int const n = static_cast<int>(data.size());
            HallRZPoissonSolver::SetPythonInletDirichlet(n, std::move(data));
        },
        py::arg("phi"),
        "Set persistent HallRZ z-lo inlet Dirichlet data from a float64 C-contiguous array with shape (Nr+1).");

    m.def("hallrz_clear_inlet_dirichlet",
        []() { HallRZPoissonSolver::ClearPythonInletDirichlet(); },
        "Clear persistent HallRZ Python inlet Dirichlet data.");

    m.def("hallrz_has_inlet_dirichlet",
        []() { return HallRZPoissonSolver::HasPythonInletDirichlet(); },
        "Return True if persistent HallRZ Python inlet Dirichlet data is active.");
#endif
}
