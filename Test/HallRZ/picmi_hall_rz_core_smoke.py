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


NR = 70
NZ = 80
RMAX = 0.070
ZMIN = 0.0
ZMAX = 0.080
LOB = 0.035
HIB = 0.050
OUT = 0.040


def build_sim(
    max_steps=200,
    use_python_boundary_arrays=True,
    rmin=0.0,
    rmax=RMAX,
    zmax=ZMAX,
    nr=NR,
    nz=NZ,
    lob=LOB,
    hib=HIB,
    out=OUT,
    rlo_bc="auto",
    warpx_max_grid_size=70,
    eb_bc_mode="neumann",
    particle_mode="grid",
    fixed_particle_count_r=60,
    fixed_particle_count_z=30,
    static_rho_test=False,
    static_rho_mode=0,
    static_rho_amp=0.0,
    static_rho_lambda_r=0.004,
    static_rho_lambda_z=0.006,
):
    # Geometry and numerics match inputs.hall_rz_core_smoke.
    zmin = ZMIN
    dt = 1.0e-11

    if rmin == 0.0:
        lo_r_field_bc = "none"
        lo_r_particle_bc = "none"
    elif rlo_bc in ("auto", "dirichlet"):
        lo_r_field_bc = "dirichlet"
        lo_r_particle_bc = "absorbing"
    elif rlo_bc in ("neumann", "robin"):
        lo_r_field_bc = "neumann"
        lo_r_particle_bc = "absorbing"
    else:
        lo_r_field_bc = "none"
        lo_r_particle_bc = "none"

    pywarpx.warpx.abort_on_warning_threshold = "high"

    grid = picmi.CylindricalGrid(
        number_of_cells=[nr, nz],
        n_azimuthal_modes=1,
        lower_bound=[rmin, zmin],
        upper_bound=[rmax, zmax],
        lower_boundary_conditions=[lo_r_field_bc, "dirichlet"],
        upper_boundary_conditions=["dirichlet", "dirichlet"],
        lower_boundary_conditions_particles=[lo_r_particle_bc, "absorbing"],
        upper_boundary_conditions_particles=["absorbing", "absorbing"],
        warpx_blocking_factor=2,
        warpx_max_grid_size=warpx_max_grid_size,
        warpx_potential_lo_r=0.0,
        warpx_potential_hi_r=0.0,
        warpx_potential_lo_z=0.0,
        warpx_potential_hi_z=0.0,
    )

    use_scalar_eb = (not use_python_boundary_arrays) and eb_bc_mode == "neumann"

    solver = picmi.ElectrostaticSolver(
        grid=grid,
        method="Multigrid",
        required_precision=1.0e-10,
        warpx_absolute_tolerance=0.0,
        maximum_iterations=100,
        warpx_self_fields_verbosity=0,
        warpx_hall_rz_enable=True,
        warpx_hall_rz_static_rho_test=int(static_rho_test),
        warpx_hall_rz_static_rho_mode=static_rho_mode,
        warpx_hall_rz_static_rho_amp=static_rho_amp,
        warpx_hall_rz_static_rho_lambda_r=static_rho_lambda_r,
        warpx_hall_rz_static_rho_lambda_z=static_rho_lambda_z,
        warpx_hall_rz_lob=lob,
        warpx_hall_rz_hib=hib,
        warpx_hall_rz_out=out,
        warpx_hall_rz_verbose=0,
        warpx_hall_rz_rel_tol=1.0e-6,
        warpx_hall_rz_abs_tol=0.0,
        warpx_hall_rz_max_iter=100,
        warpx_hall_rz_diag_interval=50,
        warpx_hall_rz_particle_diag=1,
        warpx_hall_rz_particle_diag_interval=50,
        warpx_hall_rz_eb_bc_mode=eb_bc_mode,
        warpx_hall_rz_eb_wall_flux=1 if use_scalar_eb else 0,
        warpx_hall_rz_eb_g0=1.0 if use_scalar_eb else 0.0,
        warpx_hall_rz_eb_wall_flux_segment=-1,
        warpx_hall_rz_bc_lo_r=rlo_bc,
        warpx_hall_rz_potential_lo_r=0.0,
        warpx_hall_rz_robin_lo_r_a=1.0,
        warpx_hall_rz_robin_lo_r_b=1.0,
        warpx_hall_rz_robin_lo_r_f=0.0,
    )

    electron_density = 6.1166788925e10
    proton_density = 6.241509074e10
    particle_rlo = rmin + 0.005
    particle_rhi = rmax - 0.005
    particle_zlo = 0.045
    particle_zhi = 0.075
    if particle_mode == "grid":
        electron_dist = picmi.UniformDistribution(
            density=electron_density,
            lower_bound=[particle_rlo, None, particle_zlo],
            upper_bound=[particle_rhi, None, particle_zhi],
            directed_velocity=[0.0, 0.0, 0.0],
        )
        proton_dist = picmi.UniformDistribution(
            density=proton_density,
            lower_bound=[particle_rlo, None, particle_zlo],
            upper_bound=[particle_rhi, None, particle_zhi],
            directed_velocity=[0.0, 0.0, 0.0],
        )
    elif particle_mode == "fixed":
        electron_dist = None
        proton_dist = None
    else:
        raise ValueError("particle_mode must be grid or fixed")

    if particle_mode == "fixed":
        electrons = picmi.Species(particle_type="electron", name="electrons")
        protons = picmi.Species(particle_type="proton", name="protons")
    else:
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
    if particle_mode == "fixed":
        sim.add_species(electrons, layout=None)
        sim.add_species(protons, layout=None)
    else:
        sim.add_species(electrons, layout=layout)
        sim.add_species(protons, layout=layout)
    sim.add_applied_field(external_b)

    return sim


def make_eb_neumann_array(
    g0,
    segment=None,
    phase=1.0,
    rmin=0.0,
    lob=LOB,
    hib=HIB,
    out=OUT,
    nr=NR,
    nz=NZ,
    rmax=RMAX,
    zmin=ZMIN,
    zmax=ZMAX,
):
    """Build a cell-centered HallRZ EB Neumann array.

    The array is physical data, not geometry.  Non-EB cells are harmless: C++
    only consumes values on EB2 single-valued cells.  Segment IDs match the
    C++ HallRZ convention:
      0: r=lob wall, 1: r=hib wall, 2: z=out low-r block, 3: z=out high-r block.
    """
    dr = (rmax - rmin) / nr
    dz = (zmax - zmin) / nz
    ilob = int(round((lob - rmin) / dr))
    ihib = int(round((hib - rmin) / dr))
    jout = int(round((out - zmin) / dz))
    low_block_exists = lob > rmin + 1.0e-12
    if (segment == 0 or segment == 2) and not low_block_exists:
        raise ValueError("HallRZ segment 0/2 do not exist when lob == rmin")

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


def make_neumann_equivalent_eb_robin(
    g0,
    segment=None,
    phase=1.0,
    rmin=0.0,
    lob=LOB,
    hib=HIB,
    out=OUT,
    nr=NR,
    nz=NZ,
    rmax=RMAX,
    zmin=ZMIN,
    zmax=ZMAX,
):
    """Build EB Robin arrays equivalent to dphi/dn_EB=g0."""
    f = make_eb_neumann_array(
        g0, segment, phase, rmin, lob, hib, out, nr, nz, rmax, zmin, zmax
    )
    a = np.zeros_like(f)
    b = np.ones_like(f)
    return a, b, f


def make_fixed_particle_arrays(rlo, rhi, zlo, zhi, nr=60, nz=30):
    if nr <= 0 or nz <= 0:
        raise ValueError("fixed particle counts must be positive")
    r = rlo + (np.arange(nr, dtype=np.float64) + 0.5) * (rhi - rlo) / nr
    z = zlo + (np.arange(nz, dtype=np.float64) + 0.5) * (zhi - zlo) / nz
    rr, zz = np.meshgrid(r, z, indexing="ij")
    x = np.ascontiguousarray(rr.ravel())
    y = np.zeros_like(x)
    zarr = np.ascontiguousarray(zz.ravel())
    ux = np.zeros_like(x)
    uy = np.zeros_like(x)
    uz = np.zeros_like(x)
    return x, y, zarr, ux, uy, uz


def make_fixed_particle_payload(
    rmin=0.0,
    rmax=RMAX,
    zmax=ZMAX,
    particle_count_r=60,
    particle_count_z=30,
):
    electron_density = 6.1166788925e10
    proton_density = 6.241509074e10
    particle_rlo = rmin + 0.005
    particle_rhi = rmax - 0.005
    particle_zlo = 0.045
    particle_zhi = min(0.075, zmax - 0.005)
    x, y, z, ux, uy, uz = make_fixed_particle_arrays(
        rlo=particle_rlo,
        rhi=particle_rhi,
        zlo=particle_zlo,
        zhi=particle_zhi,
        nr=particle_count_r,
        nz=particle_count_z,
    )
    volume = np.pi * (particle_rhi**2 - particle_rlo**2) * (particle_zhi - particle_zlo)
    npart = max(int(x.size), 1)
    return {
        "electrons": {
            "x": x,
            "y": y,
            "z": z,
            "ux": ux,
            "uy": uy,
            "uz": uz,
            "w": np.full(npart, electron_density * volume / npart, dtype=np.float64),
        },
        "protons": {
            "x": x,
            "y": y,
            "z": z,
            "ux": ux,
            "uy": uy,
            "uz": uz,
            "w": np.full(npart, proton_density * volume / npart, dtype=np.float64),
        },
    }


def add_fixed_particles(
    sim,
    rmin=0.0,
    rmax=RMAX,
    zmax=ZMAX,
    particle_count_r=60,
    particle_count_z=30,
):
    payload = make_fixed_particle_payload(
        rmin=rmin,
        rmax=rmax,
        zmax=zmax,
        particle_count_r=particle_count_r,
        particle_count_z=particle_count_z,
    )
    for name, data in payload.items():
        sim.particles.get(name).add_particles(unique_particles=False, **data)


class FixedParticleLoader:
    def __init__(
        self,
        sim,
        rmin=0.0,
        rmax=RMAX,
        zmax=ZMAX,
        particle_count_r=60,
        particle_count_z=30,
    ):
        self.sim = sim
        self.rmin = rmin
        self.rmax = rmax
        self.zmax = zmax
        self.particle_count_r = particle_count_r
        self.particle_count_z = particle_count_z
        self.loaded = False

    def __call__(self):
        if self.loaded:
            return
        add_fixed_particles(
            self.sim,
            rmin=self.rmin,
            rmax=self.rmax,
            zmax=self.zmax,
            particle_count_r=self.particle_count_r,
            particle_count_z=self.particle_count_z,
        )
        self.loaded = True

    def ensure_loaded(self):
        self()


def install_fixed_particle_loader(
    sim,
    rmin=0.0,
    rmax=RMAX,
    zmax=ZMAX,
    particle_count_r=60,
    particle_count_z=30,
):
    from pywarpx import callbacks

    loader = FixedParticleLoader(
        sim,
        rmin=rmin,
        rmax=rmax,
        zmax=zmax,
        particle_count_r=particle_count_r,
        particle_count_z=particle_count_z,
    )
    callbacks.installbeforeInitEsolve(loader.ensure_loaded)
    return loader


def set_default_physical_boundary_arrays(nr=NR, nz=NZ):
    hallrz.set_robin_zhi(
        np.ones(nr + 1, dtype=np.float64),
        np.ones(nr + 1, dtype=np.float64),
        np.zeros(nr + 1, dtype=np.float64),
    )
    hallrz.set_robin_rhi(
        np.ones(nz + 1, dtype=np.float64),
        np.ones(nz + 1, dtype=np.float64),
        np.zeros(nz + 1, dtype=np.float64),
    )
    phi_inlet = np.zeros(nr + 1, dtype=np.float64)
    hallrz.set_inlet_dirichlet(phi_inlet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-input", metavar="FILE", default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--nr", type=int, default=NR)
    parser.add_argument("--nz", type=int, default=NZ)
    parser.add_argument(
        "--warpx-max-grid-size",
        type=int,
        default=70,
        help="PICMI warpx_max_grid_size. Use a smaller value for CPU MPI np=4 smoke tests.",
    )
    parser.add_argument(
        "--rmin",
        type=float,
        default=0.0,
        help="Lower radial domain bound. Use 0.02 with the default grid for an aligned rmin>0 smoke.",
    )
    parser.add_argument("--rmax", type=float, default=RMAX)
    parser.add_argument("--zmax", type=float, default=ZMAX)
    parser.add_argument("--lob", type=float, default=LOB)
    parser.add_argument("--hib", type=float, default=HIB)
    parser.add_argument("--out", type=float, default=OUT)
    parser.add_argument(
        "--rlo-bc",
        choices=("auto", "axis", "dirichlet", "neumann", "robin"),
        default="auto",
        help="HallRZ lo-r Poisson BC. axis is only valid for rmin=0.",
    )
    parser.add_argument(
        "--scalar-boundaries",
        action="store_true",
        help="Use scalar C++ boundary defaults instead of Python boundary arrays.",
    )
    parser.add_argument(
        "--eb-bc-mode",
        choices=("neumann", "robin"),
        default="neumann",
        help="HallRZ EB-FVM boundary mode. Robin uses a=0,b=1,f=gn by default.",
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
    parser.add_argument(
        "--particle-mode",
        choices=("grid", "fixed"),
        default="grid",
        help="Use grid-density particles or a fixed deterministic particle list.",
    )
    parser.add_argument("--static-rho-test", action="store_true")
    parser.add_argument("--static-rho-mode", type=int, default=0)
    parser.add_argument("--static-rho-amp", type=float, default=0.0)
    parser.add_argument("--static-rho-lambda-r", type=float, default=0.004)
    parser.add_argument("--static-rho-lambda-z", type=float, default=0.006)
    args = parser.parse_args()

    if args.eb_bc_mode == "robin" and args.scalar_boundaries:
        raise RuntimeError("--eb-bc-mode robin requires Python EB Robin arrays, not --scalar-boundaries")
    if args.eb_bc_mode == "robin" and args.write_input:
        raise RuntimeError("Python EB Robin arrays are runtime data and cannot be represented in an inputs file.")
    if args.eb_bc_mode == "robin" and args.python_eb_segment >= 0:
        raise RuntimeError("--python-eb-segment is only valid for EB Neumann mode in this smoke.")

    use_python_arrays = (not args.scalar_boundaries) and (args.write_input is None)
    sim = build_sim(
        max_steps=args.max_steps,
        use_python_boundary_arrays=use_python_arrays,
        rmin=args.rmin,
        rmax=args.rmax,
        zmax=args.zmax,
        nr=args.nr,
        nz=args.nz,
        lob=args.lob,
        hib=args.hib,
        out=args.out,
        rlo_bc=args.rlo_bc,
        warpx_max_grid_size=args.warpx_max_grid_size,
        eb_bc_mode=args.eb_bc_mode,
        particle_mode=args.particle_mode,
        static_rho_test=args.static_rho_test,
        static_rho_mode=args.static_rho_mode,
        static_rho_amp=args.static_rho_amp,
        static_rho_lambda_r=args.static_rho_lambda_r,
        static_rho_lambda_z=args.static_rho_lambda_z,
    )
    if args.write_input:
        if args.python_eb_g0 is not None or args.python_eb_time_varying or args.python_eb_clear_after is not None:
            raise RuntimeError("Python EB Neumann arrays are runtime data and cannot be represented in an inputs file.")
        sim.write_input_file(args.write_input)
    else:
        sim.initialize_inputs()
        if use_python_arrays:
            eb_g0 = 1.0 if args.python_eb_g0 is None else args.python_eb_g0
            if args.eb_bc_mode == "robin":
                hallrz.set_eb_robin(
                    *make_neumann_equivalent_eb_robin(
                        eb_g0,
                        rmin=args.rmin,
                        rmax=args.rmax,
                        zmax=args.zmax,
                        lob=args.lob,
                        hib=args.hib,
                        out=args.out,
                        nr=args.nr,
                        nz=args.nz,
                    )
                )
            else:
                hallrz.set_eb_neumann(
                    make_eb_neumann_array(
                        eb_g0,
                        args.python_eb_segment,
                        rmin=args.rmin,
                        rmax=args.rmax,
                        zmax=args.zmax,
                        lob=args.lob,
                        hib=args.hib,
                        out=args.out,
                        nr=args.nr,
                        nz=args.nz,
                    )
                )
            set_default_physical_boundary_arrays(args.nr, args.nz)
        fixed_particle_loader = None
        if args.particle_mode == "fixed":
            fixed_particle_loader = install_fixed_particle_loader(
                sim, rmin=args.rmin, rmax=args.rmax, zmax=args.zmax
            )
        sim.initialize_warpx()
        if fixed_particle_loader is not None:
            fixed_particle_loader.ensure_loaded()
        if use_python_arrays and (args.python_eb_time_varying or args.python_eb_clear_after is not None):
            if args.eb_bc_mode == "robin":
                raise RuntimeError("time-varying/clear-after EB smoke currently targets Neumann mode")
            eb_g0 = 1.0 if args.python_eb_g0 is None else args.python_eb_g0
            for n in range(args.max_steps):
                if args.python_eb_clear_after is not None and n >= args.python_eb_clear_after:
                    if hallrz.has_eb_neumann():
                        hallrz.clear_eb_neumann()
                elif args.python_eb_time_varying:
                    phase = 1.0 + 0.1 * np.sin(2.0 * np.pi * n / max(args.max_steps, 1))
                    hallrz.set_eb_neumann(
                        make_eb_neumann_array(
                            eb_g0,
                            args.python_eb_segment,
                            phase,
                            args.rmin,
                            args.lob,
                            args.hib,
                            args.out,
                            nr=args.nr,
                            nz=args.nz,
                            rmax=args.rmax,
                            zmax=args.zmax,
                        )
                    )
                sim.step(1)
        else:
            sim.step(args.max_steps)


if __name__ == "__main__":
    main()
