#!/usr/bin/env python3
"""HallRZ PICMI core smoke.

This Python smoke keeps all HallRZ geometry construction in C++ and provides
runtime boundary data through ``pywarpx.hallrz`` arrays by default.  Use
``--scalar-boundaries`` to reproduce the input-file scalar wall-flux path.
"""

import argparse

import numpy as np

import pywarpx
from pywarpx import hallrz
from pywarpx import picmi


def build_sim(max_steps=200, use_python_boundary_arrays=True):
    # Geometry and numerics match inputs.hall_rz_core_smoke.
    nr = 70
    nz = 80
    rmin = 0.0
    rmax = 0.070
    zmin = 0.0
    zmax = 0.080
    dt = 1.0e-11

    # HallRZ-specific scalar parameters.  Set them before constructing the
    # PICMI Simulation so both direct Python execution and write_input_file()
    # see the same values.  No EB implicit function and no nodal geometry are
    # passed from Python; C++ builds the EB2 and nodal FVM geometry.
    pywarpx.warpx.hall_rz_enable = 1
    pywarpx.warpx.hall_rz_static_rho_test = 0
    pywarpx.warpx.hall_rz_lob = 0.035
    pywarpx.warpx.hall_rz_hib = 0.050
    pywarpx.warpx.hall_rz_out = 0.040
    pywarpx.warpx.hall_rz_verbose = 0
    pywarpx.warpx.hall_rz_rel_tol = 1.0e-6
    pywarpx.warpx.hall_rz_abs_tol = 0.0
    pywarpx.warpx.hall_rz_max_iter = 100
    pywarpx.warpx.hall_rz_diag_interval = 50
    pywarpx.warpx.hall_rz_particle_diag = 1
    pywarpx.warpx.hall_rz_particle_diag_interval = 50
    pywarpx.warpx.hall_rz_eb_wall_flux = 0 if use_python_boundary_arrays else 1
    pywarpx.warpx.hall_rz_eb_g0 = 0.0 if use_python_boundary_arrays else 1.0
    pywarpx.warpx.hall_rz_eb_wall_flux_segment = -1
    pywarpx.warpx.abort_on_warning_threshold = "high"

    grid = picmi.CylindricalGrid(
        number_of_cells=[nr, nz],
        n_azimuthal_modes=1,
        lower_bound=[rmin, zmin],
        upper_bound=[rmax, zmax],
        lower_boundary_conditions=["none", "dirichlet"],
        upper_boundary_conditions=["dirichlet", "dirichlet"],
        lower_boundary_conditions_particles=["none", "absorbing"],
        upper_boundary_conditions_particles=["absorbing", "absorbing"],
        warpx_blocking_factor=2,
        warpx_max_grid_size=70,
        warpx_potential_lo_r=0.0,
        warpx_potential_hi_r=0.0,
        warpx_potential_lo_z=0.0,
        warpx_potential_hi_z=0.0,
    )

    solver = picmi.ElectrostaticSolver(
        grid=grid,
        method="Multigrid",
        required_precision=1.0e-10,
        warpx_absolute_tolerance=0.0,
        maximum_iterations=100,
        warpx_self_fields_verbosity=0,
    )

    electron_dist = picmi.UniformDistribution(
        density=6.1166788925e10,
        lower_bound=[0.005, None, 0.045],
        upper_bound=[0.065, None, 0.075],
        directed_velocity=[0.0, 0.0, 0.0],
    )
    proton_dist = picmi.UniformDistribution(
        density=6.241509074e10,
        lower_bound=[0.005, None, 0.045],
        upper_bound=[0.065, None, 0.075],
        directed_velocity=[0.0, 0.0, 0.0],
    )

    electrons = picmi.Species(
        particle_type="electron",
        name="electrons",
        initial_distribution=electron_dist,
    )
    protons = picmi.Species(
        particle_type="proton",
        name="protons",
        initial_distribution=proton_dist,
    )

    layout = picmi.GriddedLayout(n_macroparticle_per_cell=[1, 1, 1], grid=grid)

    # External particle B in RZ component order used by WarpX: Br, Btheta, Bz.
    external_b = picmi.ConstantAppliedField(Bx=1.0e-3, By=0.0, Bz=0.0)

    sim = picmi.Simulation(
        solver=solver,
        max_steps=max_steps,
        time_step_size=dt,
        verbose=0,
        particle_shape="linear",
        warpx_grid_type="staggered",
        warpx_use_filter=0,
        warpx_amr_check_input=0,
        warpx_serialize_initial_conditions=0,
        warpx_do_dynamic_scheduling=0,
    )
    sim.add_species(electrons, layout=layout)
    sim.add_species(protons, layout=layout)
    sim.add_applied_field(external_b)

    return sim


def make_eb_neumann_array(g0, segment=None, phase=1.0):
    """Build a cell-centered HallRZ EB Neumann array.

    The array is physical data, not geometry.  Non-EB cells are harmless: C++
    only consumes values on EB2 single-valued cells.  Segment IDs match the
    C++ HallRZ convention:
      0: r=lob wall, 1: r=hib wall, 2: z=out low-r block, 3: z=out high-r block.
    """
    nr = 70
    nz = 80
    ilob = 35
    ihib = 50
    jout = 40

    g = np.zeros((nr, nz), dtype=np.float64)
    value = np.float64(g0 * phase)
    if segment is None or segment < 0:
        g[:, :] = value
    elif segment == 0:
        # Fluid is on the high-r side of the low channel wall.
        g[ilob, :jout] = value
    elif segment == 1:
        # Fluid is on the low-r side of the high channel wall.
        g[ihib - 1, :jout] = value
    elif segment == 2:
        # Plume is on the high-z side of the low-r covered block.
        g[:ilob, jout] = value
    elif segment == 3:
        # Plume is on the high-z side of the high-r covered block.
        g[ihib:, jout] = value
    else:
        raise ValueError("--python-eb-segment must be -1, 0, 1, 2, or 3")
    return g


def set_default_physical_boundary_arrays():
    hallrz.set_robin_zhi(
        np.ones(71, dtype=np.float64),
        np.ones(71, dtype=np.float64),
        np.zeros(71, dtype=np.float64),
    )
    hallrz.set_robin_rhi(
        np.ones(81, dtype=np.float64),
        np.ones(81, dtype=np.float64),
        np.zeros(81, dtype=np.float64),
    )
    phi_inlet = np.zeros(71, dtype=np.float64)
    hallrz.set_inlet_dirichlet(phi_inlet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-input", metavar="FILE", default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--scalar-boundaries",
        action="store_true",
        help="Use scalar C++ boundary defaults instead of Python boundary arrays.",
    )
    parser.add_argument(
        "--python-eb-g0",
        type=float,
        default=None,
        help="Use a Python-provided constant cell-centered EB Neumann array instead of scalar wall flux.",
    )
    parser.add_argument(
        "--python-eb-segment",
        type=int,
        default=-1,
        help="Restrict Python EB Neumann data to one Hall segment: -1 all, or 0..3.",
    )
    parser.add_argument(
        "--python-eb-time-varying",
        action="store_true",
        help="Update Python EB Neumann data before every step with a smooth sinusoidal factor.",
    )
    parser.add_argument(
        "--python-eb-clear-after",
        type=int,
        default=None,
        help="Clear Python EB Neumann data after this many one-step advances; scalar wall flux remains disabled.",
    )
    args = parser.parse_args()

    use_python_arrays = (not args.scalar_boundaries) and (args.write_input is None)
    sim = build_sim(max_steps=args.max_steps, use_python_boundary_arrays=use_python_arrays)
    if args.write_input:
        if args.python_eb_g0 is not None or args.python_eb_time_varying or args.python_eb_clear_after is not None:
            raise RuntimeError("Python EB Neumann arrays are runtime data and cannot be represented in an inputs file.")
        sim.write_input_file(args.write_input)
    else:
        sim.initialize_inputs()
        if use_python_arrays:
            eb_g0 = 1.0 if args.python_eb_g0 is None else args.python_eb_g0
            hallrz.set_eb_neumann(make_eb_neumann_array(eb_g0, args.python_eb_segment))
            set_default_physical_boundary_arrays()
        sim.initialize_warpx()
        if use_python_arrays and (args.python_eb_time_varying or args.python_eb_clear_after is not None):
            eb_g0 = 1.0 if args.python_eb_g0 is None else args.python_eb_g0
            for n in range(args.max_steps):
                if args.python_eb_clear_after is not None and n >= args.python_eb_clear_after:
                    if hallrz.has_eb_neumann():
                        hallrz.clear_eb_neumann()
                elif args.python_eb_time_varying:
                    phase = 1.0 + 0.1 * np.sin(2.0 * np.pi * n / max(args.max_steps, 1))
                    hallrz.set_eb_neumann(make_eb_neumann_array(eb_g0, args.python_eb_segment, phase))
                sim.step(1)
        else:
            sim.step(args.max_steps)


if __name__ == "__main__":
    main()
