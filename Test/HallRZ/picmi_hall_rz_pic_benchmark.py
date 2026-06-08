#!/usr/bin/env python3
"""Stage-0 HallRZ PIC benchmark.

This benchmark reuses the HallRZ core-smoke setup and adds repeatable timing
and end-of-run correctness checks.  It is intentionally a Python/PICMI driver so
that GPU, CPU, and MPI runs use the same user-facing path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import pywarpx
from pywarpx import fields as warpx_fields
from pywarpx import hallrz
from pywarpx._libwarpx import libwarpx


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import picmi_hall_rz_core_smoke as core  # noqa: E402


FIELD_WRAPPERS = (
    ("rho_fp", warpx_fields.RhoFPWrapper),
    ("phi_fp", warpx_fields.PhiFPWrapper),
    ("Er_fp", warpx_fields.ExFPWrapper),
    ("Ez_fp", warpx_fields.EzFPWrapper),
)
COARSE_FINE_PHI_TOL = 1.0e-10


def _rank_info() -> tuple[int, int]:
    try:
        return int(libwarpx.libwarpx_so.getMyProc()), int(libwarpx.libwarpx_so.getNProcs())
    except Exception:
        return 0, 1


def _field_array(wrapper_factory, level: int):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The fields wrapper is now obsolete.*")
        wrapper = wrapper_factory(level=level)
    return np.asarray(wrapper[:])


def _array_stats(name: str, array: np.ndarray) -> dict:
    array = np.asarray(array)
    finite = np.isfinite(array)
    finite_count = int(np.count_nonzero(finite))
    total_count = int(array.size)
    stats = {
        "name": name,
        "shape": list(array.shape),
        "finite": bool(finite_count == total_count),
        "nan_count": int(np.count_nonzero(np.isnan(array))),
        "inf_count": int(np.count_nonzero(np.isinf(array))),
    }
    if finite_count > 0:
        finite_values = array[finite]
        stats.update(
            {
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "max_abs": float(np.max(np.abs(finite_values))),
                "l2": float(np.sqrt(np.mean(finite_values * finite_values))),
            }
        )
    else:
        stats.update({"min": None, "max": None, "max_abs": None, "l2": None})
    return stats


def collect_field_stats(level: int = 0) -> dict:
    stats = {}
    for name, wrapper_factory in FIELD_WRAPPERS:
        stats[name] = _array_stats(name, _field_array(wrapper_factory, level))
    return stats


def collect_field_stats_by_level(finest_level: int) -> list[dict]:
    return [
        {
            "level": lev,
            "fields": collect_field_stats(lev),
        }
        for lev in range(finest_level + 1)
    ]


def write_snapshot(path: Path, args, summary: dict, finest_level: int) -> None:
    payload = {
        "summary_json": np.array(json.dumps(summary, sort_keys=True)),
        "field_names": np.array([name for name, _ in FIELD_WRAPPERS]),
    }
    for lev in range(finest_level + 1):
        for name, wrapper_factory in FIELD_WRAPPERS:
            payload[f"lev{lev}_{name}"] = np.ascontiguousarray(
                _field_array(wrapper_factory, lev)
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _node_bilinear_interp(coarse_phi: np.ndarray, fine_i: np.ndarray, fine_j: np.ndarray):
    """Match AMReX node-bilinear interpolation for refinement ratio 2."""
    coarse_i = fine_i // 2
    coarse_j = fine_j // 2
    wr = 0.5 * (fine_i % 2)
    wz = 0.5 * (fine_j % 2)

    coarse_i1 = np.minimum(coarse_i + 1, coarse_phi.shape[0] - 1)
    coarse_j1 = np.minimum(coarse_j + 1, coarse_phi.shape[1] - 1)

    c00 = coarse_phi[coarse_i, coarse_j]
    c10 = coarse_phi[coarse_i1, coarse_j]
    c01 = coarse_phi[coarse_i, coarse_j1]
    c11 = coarse_phi[coarse_i1, coarse_j1]
    return (
        (1.0 - wr) * (1.0 - wz) * c00
        + wr * (1.0 - wz) * c10
        + (1.0 - wr) * wz * c01
        + wr * wz * c11
    )


def _error_stats(error: np.ndarray) -> dict:
    error = np.asarray(error)
    if error.size == 0:
        return {"count": 0, "max_abs": 0.0, "l2": 0.0}
    return {
        "count": int(error.size),
        "max_abs": float(np.max(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error * error))),
    }


def collect_coarse_fine_phi_continuity(args, finest_level: int) -> dict:
    if (not args.hall_rz_amr_enable) or finest_level < 1 or args.init_only:
        return {"enabled": False, "pass": True, "pairs": [], "tolerance": COARSE_FINE_PHI_TOL}

    phi = [_field_array(warpx_fields.PhiFPWrapper, lev) for lev in range(finest_level + 1)]
    pairs = []
    all_max = 0.0
    for fine_lev in range(1, finest_level + 1):
        coarse_phi = phi[fine_lev - 1]
        fine_phi = phi[fine_lev]

        j = np.arange(fine_phi.shape[1], dtype=np.int64)
        i = np.full_like(j, fine_phi.shape[0] - 1)
        rhi_error = fine_phi[-1, :] - _node_bilinear_interp(coarse_phi, i, j)

        i = np.arange(fine_phi.shape[0], dtype=np.int64)
        j = np.full_like(i, fine_phi.shape[1] - 1)
        zhi_error = fine_phi[:, -1] - _node_bilinear_interp(coarse_phi, i, j)

        rhi_stats = _error_stats(rhi_error)
        zhi_stats = _error_stats(zhi_error)
        pair_max = max(rhi_stats["max_abs"], zhi_stats["max_abs"])
        all_max = max(all_max, pair_max)
        pairs.append(
            {
                "coarse_level": fine_lev - 1,
                "fine_level": fine_lev,
                "rhi_artificial": rhi_stats,
                "zhi_artificial": zhi_stats,
                "max_abs": pair_max,
            }
        )

    return {
        "enabled": True,
        "pass": bool(all_max <= COARSE_FINE_PHI_TOL),
        "tolerance": COARSE_FINE_PHI_TOL,
        "max_abs": all_max,
        "pairs": pairs,
    }


def collect_particle_counts(sim) -> dict:
    counts = {}
    for name in ("electrons", "protons"):
        try:
            counts[name] = int(sim.particles.get(name).total_number_of_particles(True, False))
        except Exception as exc:
            counts[name] = {"error": str(exc)}
    return counts


def field_checks_pass(field_stats: dict) -> bool:
    return all(stats["finite"] for stats in field_stats.values())


def field_level_checks_pass(field_levels: list[dict]) -> bool:
    return all(field_checks_pass(level["fields"]) for level in field_levels)


def particle_checks_pass(particle_counts: dict) -> bool:
    for value in particle_counts.values():
        if isinstance(value, dict):
            return False
        if value <= 0:
            return False
    return True


def _safe_call(obj, name: str):
    try:
        attr = getattr(obj, name)
        return attr() if callable(attr) else attr
    except Exception as exc:
        return {"error": str(exc)}


def _safe_geom_vector(geom, name: str, ndim: int = 2) -> list[float] | dict:
    values = []
    try:
        for idir in range(ndim):
            values.append(float(getattr(geom, name)(idir)))
    except Exception as exc:
        return {"error": str(exc)}
    return values


def _box_summary(box) -> dict:
    return {
        "repr": str(box),
        "small_end": str(_safe_call(box, "small_end")),
        "big_end": str(_safe_call(box, "big_end")),
        "length": str(_safe_call(box, "length")),
    }


def collect_amr_grid_summary() -> dict:
    try:
        warpx = libwarpx.libwarpx_so.get_instance()
        max_level = int(warpx.max_level)
        finest_level = int(warpx.finest_level)
    except Exception as exc:
        return {"error": str(exc)}

    levels = []
    for lev in range(finest_level + 1):
        level = {"level": lev}
        try:
            geom = warpx.Geom(lev)
            level["prob_lo"] = _safe_geom_vector(geom, "ProbLo")
            level["prob_hi"] = _safe_geom_vector(geom, "ProbHi")
            level["cell_size"] = _safe_geom_vector(geom.data(), "CellSize")
        except Exception as exc:
            level["geom_error"] = str(exc)
        try:
            ba = warpx.boxArray(lev)
            num_boxes = _safe_call(ba, "size")
            level["box_array"] = str(ba)
            level["num_boxes"] = num_boxes
            level["minimal_box"] = _box_summary(_safe_call(ba, "minimal_box"))
            if isinstance(num_boxes, int):
                level["boxes"] = [_box_summary(ba.get(i)) for i in range(num_boxes)]
        except Exception as exc:
            level["box_array_error"] = str(exc)
        try:
            level["distribution_map"] = str(warpx.DistributionMap(lev))
        except Exception as exc:
            level["distribution_map_error"] = str(exc)
        levels.append(level)

    return {
        "max_level": max_level,
        "finest_level": finest_level,
        "levels": levels,
    }


def amr_grid_checks_pass(args, amr_grid: dict) -> bool:
    if not args.hall_rz_amr_enable:
        return True
    expected_max_level = args.hall_rz_amr_max_level
    return (
        not amr_grid.get("error")
        and amr_grid.get("max_level") == expected_max_level
        and amr_grid.get("finest_level") == expected_max_level
        and len(amr_grid.get("levels", [])) == expected_max_level + 1
    )


def active_hall_rz_amr_levels(args) -> range:
    if args.hall_rz_amr_enable:
        return range(args.hall_rz_amr_max_level + 1)
    return range(1)


def make_summary(args, sim, init_wall_time: float, chunk_records: list[dict]) -> dict:
    particle_counts = collect_particle_counts(sim)
    amr_grid = collect_amr_grid_summary()
    finest_level = int(amr_grid.get("finest_level", 0)) if not amr_grid.get("error") else 0
    if args.init_only and args.hall_rz_amr_enable:
        field_levels = []
        field_stats = {}
    else:
        field_levels = collect_field_stats_by_level(finest_level)
        field_stats = field_levels[0]["fields"] if field_levels else {}
    coarse_fine_phi_continuity = collect_coarse_fine_phi_continuity(args, finest_level)
    total_step_wall_time = float(sum(record["wall_time_s"] for record in chunk_records))
    max_steps = int(sum(record["steps"] for record in chunk_records))
    chunk_times = [record["wall_time_s"] for record in chunk_records]
    avg_step_wall_time = total_step_wall_time / max(max_steps, 1)
    rank, nprocs = _rank_info()
    summary = {
        "case": args.case,
        "rank": rank,
        "nprocs": nprocs,
        "max_steps": max_steps,
        "chunk_steps": args.chunk_steps,
        "init_wall_time_s": init_wall_time,
        "total_step_wall_time_s": total_step_wall_time,
        "avg_step_wall_time_s": avg_step_wall_time,
        "chunk_records": chunk_records,
        "chunk_wall_time_s": {
            "min": float(min(chunk_times)) if chunk_times else 0.0,
            "max": float(max(chunk_times)) if chunk_times else 0.0,
            "mean": float(np.mean(chunk_times)) if chunk_times else 0.0,
            "p50": float(np.percentile(chunk_times, 50.0)) if chunk_times else 0.0,
            "p95": float(np.percentile(chunk_times, 95.0)) if chunk_times else 0.0,
        },
        "geometry": {
            "rmin": args.rmin,
            "rmax": args.rmax,
            "zmax": args.zmax,
            "lob": args.lob,
            "hib": args.hib,
            "out": args.out,
            "nr": args.nr,
            "nz": args.nz,
            "warpx_max_grid_size": args.warpx_max_grid_size,
            "hall_rz_amr_enable": args.hall_rz_amr_enable,
            "hall_rz_amr_max_level": args.hall_rz_amr_max_level,
            "hall_rz_amr_r1": args.hall_rz_amr_r1,
            "hall_rz_amr_z1": args.hall_rz_amr_z1,
            "hall_rz_amr_r2": args.hall_rz_amr_r2,
            "hall_rz_amr_z2": args.hall_rz_amr_z2,
        },
        "solver": {
            "requested_hall_rz_max_coarsening_level": args.hall_rz_max_coarsening_level,
            "requested_hall_rz_max_coarsening_levels": args.hall_rz_max_coarsening_levels,
            "hall_rz_amr_phi_init_weight": args.hall_rz_amr_phi_init_weight,
            "static_rho_test": args.static_rho_test,
            "static_rho_mode": args.static_rho_mode,
            "static_rho_amp": args.static_rho_amp,
            "static_rho_lambda_r": args.static_rho_lambda_r,
            "static_rho_lambda_z": args.static_rho_lambda_z,
        },
        "amr_grid": amr_grid,
        "boundary_mode": {
            "eb_bc_mode": args.eb_bc_mode,
            "scalar_boundaries": args.scalar_boundaries,
            "python_eb_levels": [] if args.scalar_boundaries else (
                list(active_hall_rz_amr_levels(args))
            ),
            "python_physical_boundary_arrays": (
                (not args.scalar_boundaries) and (not args.hall_rz_amr_enable)
            ),
            "python_eb_g0": args.python_eb_g0,
            "python_eb_segment": args.python_eb_segment,
            "rlo_bc": args.rlo_bc,
            "particle_mode": args.particle_mode,
        },
        "fields": field_stats,
        "field_levels": field_levels,
        "coarse_fine_phi_continuity": coarse_fine_phi_continuity,
        "particles": particle_counts,
        "checks": {
            "fields_finite": field_level_checks_pass(field_levels)
            if field_levels else field_checks_pass(field_stats),
            "field_levels_finite": field_level_checks_pass(field_levels),
            "coarse_fine_phi_continuity": coarse_fine_phi_continuity["pass"],
            "particles_positive": particle_checks_pass(particle_counts),
            "amr_grid": amr_grid_checks_pass(args, amr_grid),
        },
    }
    summary["checks"]["pass"] = all(summary["checks"].values())
    return summary


def _level_cell_shape(args, lev: int) -> tuple[int, int]:
    refinement = 2**lev
    return args.nr * refinement, args.nz * refinement


def _configure_python_eb_data_for_level(args, lev: int) -> None:
    nr, nz = _level_cell_shape(args, lev)
    if args.eb_bc_mode == "robin":
        hallrz.set_eb_robin(
            *core.make_neumann_equivalent_eb_robin(
                args.python_eb_g0,
                rmin=args.rmin,
                rmax=args.rmax,
                zmax=args.zmax,
                lob=args.lob,
                hib=args.hib,
                out=args.out,
                nr=nr,
                nz=nz,
            ),
            lev=lev,
        )
    else:
        hallrz.set_eb_neumann(
            core.make_eb_neumann_array(
                args.python_eb_g0,
                args.python_eb_segment,
                rmin=args.rmin,
                rmax=args.rmax,
                zmax=args.zmax,
                lob=args.lob,
                hib=args.hib,
                out=args.out,
                nr=nr,
                nz=nz,
            ),
            lev=lev,
        )


def configure_runtime_boundary_data(args):
    hallrz.clear_eb_neumann()
    hallrz.clear_eb_robin()
    if args.scalar_boundaries:
        return
    for lev in active_hall_rz_amr_levels(args):
        _configure_python_eb_data_for_level(args, lev)
    if not args.hall_rz_amr_enable:
        core.set_default_physical_boundary_arrays(args.nr, args.nz)


def configure_hall_rz_amr_inputs(args):
    if not args.hall_rz_amr_enable:
        if args.hall_rz_max_coarsening_levels is not None:
            raise ValueError(
                "--hall-rz-max-coarsening-levels is only valid with --hall-rz-amr-enable"
            )
        return

    pywarpx.amr.max_level = args.hall_rz_amr_max_level
    pywarpx.amr.ref_ratio = [2] * args.hall_rz_amr_max_level
    pywarpx.warpx.regrid_int = -1
    pywarpx.warpx.hall_rz_amr_enable = 1
    pywarpx.warpx.hall_rz_amr_r1 = args.hall_rz_amr_r1
    pywarpx.warpx.hall_rz_amr_z1 = args.hall_rz_amr_z1
    if args.hall_rz_amr_max_level == 2:
        pywarpx.warpx.hall_rz_amr_r2 = args.hall_rz_amr_r2
        pywarpx.warpx.hall_rz_amr_z2 = args.hall_rz_amr_z2
    pywarpx.warpx.hall_rz_amr_phi_init_weight = args.hall_rz_amr_phi_init_weight
    if args.init_only:
        pywarpx.warpx.do_electrostatic = "none"
    if args.hall_rz_max_coarsening_levels is not None:
        pywarpx.warpx.hall_rz_max_coarsening_levels = args.hall_rz_max_coarsening_levels


def run_case(args) -> dict:
    if not (args.init_only and args.hall_rz_amr_enable):
        pywarpx.warpx.abort_on_warning_threshold = "high"
    sim = core.build_sim(
        max_steps=args.max_steps,
        use_python_boundary_arrays=not args.scalar_boundaries,
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

    t0 = time.perf_counter()
    sim.initialize_inputs()
    if args.amrex_the_arena_init_size is not None:
        pywarpx.amrex.the_arena_init_size = args.amrex_the_arena_init_size
    pywarpx.warpx.hall_rz_max_coarsening_level = args.hall_rz_max_coarsening_level
    configure_hall_rz_amr_inputs(args)
    configure_runtime_boundary_data(args)
    fixed_particle_loader = None
    if args.particle_mode == "fixed":
        fixed_particle_loader = core.install_fixed_particle_loader(
            sim, rmin=args.rmin, rmax=args.rmax, zmax=args.zmax
        )
    sim.initialize_warpx()
    if fixed_particle_loader is not None:
        fixed_particle_loader.ensure_loaded()
    init_wall_time = time.perf_counter() - t0

    chunk_records = []
    remaining = args.max_steps
    step0 = 0
    while remaining > 0 and not args.init_only:
        nsteps = min(args.chunk_steps, remaining)
        t_chunk = time.perf_counter()
        sim.step(nsteps)
        wall_time = time.perf_counter() - t_chunk
        step0 += nsteps
        remaining -= nsteps
        chunk_records.append(
            {
                "end_step": step0,
                "steps": nsteps,
                "wall_time_s": wall_time,
                "avg_step_wall_time_s": wall_time / max(nsteps, 1),
            }
        )

    summary = make_summary(args, sim, init_wall_time, chunk_records)
    if args.snapshot_npz is not None:
        rank, _ = _rank_info()
        if rank == 0:
            amr_grid = summary.get("amr_grid", {})
            finest_level = int(amr_grid.get("finest_level", 0)) if not amr_grid.get("error") else 0
            write_snapshot(args.snapshot_npz, args, summary, finest_level)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="stage0_neumann_baseline")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--nr", type=int, default=core.NR)
    parser.add_argument("--nz", type=int, default=core.NZ)
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Initialize WarpX and write a summary without advancing PIC steps.",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--snapshot-npz", type=Path, default=None)
    parser.add_argument(
        "--warpx-max-grid-size",
        type=int,
        default=70,
        help="PICMI warpx_max_grid_size. Use a smaller value for CPU MPI np=4.",
    )
    parser.add_argument(
        "--amrex-the-arena-init-size",
        type=int,
        default=None,
        help="Optional amrex.the_arena_init_size override for short GPU smoke tests.",
    )
    parser.add_argument("--rmin", type=float, default=0.0)
    parser.add_argument("--rmax", type=float, default=core.RMAX)
    parser.add_argument("--zmax", type=float, default=core.ZMAX)
    parser.add_argument("--lob", type=float, default=core.LOB)
    parser.add_argument("--hib", type=float, default=core.HIB)
    parser.add_argument("--out", type=float, default=core.OUT)
    parser.add_argument(
        "--hall-rz-max-coarsening-level",
        type=int,
        default=30,
        help="warpx.hall_rz_max_coarsening_level passed to AMReX as the MCL upper bound.",
    )
    parser.add_argument(
        "--hall-rz-max-coarsening-levels",
        type=int,
        nargs="+",
        default=None,
        metavar="MCL",
        help="warpx.hall_rz_max_coarsening_levels, one entry per active HallRZ AMR level.",
    )
    parser.add_argument("--hall-rz-amr-enable", action="store_true")
    parser.add_argument(
        "--hall-rz-amr-max-level",
        type=int,
        choices=(1, 2),
        default=2,
        help="Static HallRZ AMR finest level when --hall-rz-amr-enable is used.",
    )
    parser.add_argument("--hall-rz-amr-r1", type=float, default=None)
    parser.add_argument("--hall-rz-amr-z1", type=float, default=None)
    parser.add_argument("--hall-rz-amr-r2", type=float, default=None)
    parser.add_argument("--hall-rz-amr-z2", type=float, default=None)
    parser.add_argument(
        "--hall-rz-amr-phi-init-weight",
        type=float,
        default=1.0,
        help="Fine-level AMR initial-guess weight for previous fine phi in [0, 1].",
    )
    parser.add_argument(
        "--rlo-bc",
        choices=("auto", "axis", "dirichlet", "neumann", "robin"),
        default="auto",
    )
    parser.add_argument(
        "--scalar-boundaries",
        action="store_true",
        help="Use scalar C++ EB Neumann data instead of Python EB Neumann arrays.",
    )
    parser.add_argument(
        "--eb-bc-mode",
        choices=("neumann", "robin"),
        default="neumann",
        help="HallRZ EB-FVM boundary mode. Robin uses a=0,b=1,f=gn.",
    )
    parser.add_argument("--python-eb-g0", type=float, default=1.0)
    parser.add_argument("--python-eb-segment", type=int, default=-1)
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

    if args.max_steps < 0 or (args.max_steps == 0 and not args.init_only):
        raise ValueError("--max-steps must be positive unless --init-only is used")
    if args.chunk_steps <= 0:
        raise ValueError("--chunk-steps must be positive")
    if args.nr <= 0 or args.nz <= 0:
        raise ValueError("--nr and --nz must be positive")
    if args.warpx_max_grid_size <= 0:
        raise ValueError("--warpx-max-grid-size must be positive")
    if args.amrex_the_arena_init_size is not None and args.amrex_the_arena_init_size < 0:
        raise ValueError("--amrex-the-arena-init-size must be non-negative")
    if args.hall_rz_max_coarsening_level < 0:
        raise ValueError("--hall-rz-max-coarsening-level must be non-negative")
    if args.hall_rz_max_coarsening_levels is not None:
        if any(mcl < 0 for mcl in args.hall_rz_max_coarsening_levels):
            raise ValueError("--hall-rz-max-coarsening-levels entries must be non-negative")
        if args.hall_rz_amr_enable:
            expected_mcls = args.hall_rz_amr_max_level + 1
            if len(args.hall_rz_max_coarsening_levels) != expected_mcls:
                raise ValueError(
                    "--hall-rz-max-coarsening-levels must have one entry per active "
                    f"HallRZ AMR level ({expected_mcls} entries for "
                    f"--hall-rz-amr-max-level {args.hall_rz_amr_max_level})"
                )
    if not math.isfinite(args.rmin):
        raise ValueError("--rmin must be finite")
    if not math.isfinite(args.rmax) or args.rmax <= args.rmin:
        raise ValueError("--rmax must be finite and greater than --rmin")
    if not math.isfinite(args.zmax) or args.zmax <= 0.0:
        raise ValueError("--zmax must be finite and positive")
    for name in ("lob", "hib", "out"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"--{name} must be finite")
    if not (args.rmin <= args.lob < args.hib < args.rmax):
        raise ValueError("--rmin <= --lob < --hib < --rmax is required")
    if not (0.0 < args.out < args.zmax):
        raise ValueError("0 < --out < --zmax is required")
    if args.hall_rz_amr_enable:
        for name in ("hall_rz_amr_r1", "hall_rz_amr_z1"):
            value = getattr(args, name)
            if value is None or not math.isfinite(value):
                raise ValueError(f"--{name.replace('_', '-')} is required and must be finite")
        if not (args.out < args.hall_rz_amr_z1 < args.zmax):
            raise ValueError("--out < --hall-rz-amr-z1 < --zmax is required")
        if not (args.hib < args.hall_rz_amr_r1 < args.rmax):
            raise ValueError("--hib < --hall-rz-amr-r1 < --rmax is required")
        if args.hall_rz_amr_max_level == 2:
            for name in ("hall_rz_amr_r2", "hall_rz_amr_z2"):
                value = getattr(args, name)
                if value is None or not math.isfinite(value):
                    raise ValueError(
                        f"--{name.replace('_', '-')} is required and must be finite"
                    )
            if not (args.out < args.hall_rz_amr_z2 < args.hall_rz_amr_z1):
                raise ValueError("--out < --hall-rz-amr-z2 < --hall-rz-amr-z1 is required")
            if not (args.hib < args.hall_rz_amr_r2 < args.hall_rz_amr_r1):
                raise ValueError("--hib < --hall-rz-amr-r2 < --hall-rz-amr-r1 is required")
    elif args.hall_rz_max_coarsening_levels is not None:
        raise ValueError("--hall-rz-max-coarsening-levels requires --hall-rz-amr-enable")
    if args.eb_bc_mode == "robin" and args.scalar_boundaries:
        raise ValueError("--eb-bc-mode robin requires Python EB Robin arrays, not --scalar-boundaries")
    if args.eb_bc_mode == "robin" and args.python_eb_segment >= 0:
        raise ValueError("--python-eb-segment is only valid for EB Neumann mode")
    if not math.isfinite(args.hall_rz_amr_phi_init_weight):
        raise ValueError("--hall-rz-amr-phi-init-weight must be finite")
    if args.hall_rz_amr_phi_init_weight < 0.0 or args.hall_rz_amr_phi_init_weight > 1.0:
        raise ValueError("--hall-rz-amr-phi-init-weight must be in [0, 1]")
    if args.static_rho_mode < 0:
        raise ValueError("--static-rho-mode must be non-negative")
    for name in ("static_rho_amp", "static_rho_lambda_r", "static_rho_lambda_z"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if args.static_rho_lambda_r <= 0.0 or args.static_rho_lambda_z <= 0.0:
        raise ValueError("--static-rho-lambda-r/z must be positive")
    return args


def main() -> None:
    args = parse_args()
    summary = run_case(args)
    rank = int(summary["rank"])
    text = json.dumps(summary, indent=2, sort_keys=True)
    if rank == 0:
        if args.summary_json is not None:
            args.summary_json.write_text(text + "\n", encoding="utf-8")
        print(text)
    if not summary["checks"]["pass"]:
        raise RuntimeError("HallRZ PIC benchmark checks failed")


if __name__ == "__main__":
    main()
