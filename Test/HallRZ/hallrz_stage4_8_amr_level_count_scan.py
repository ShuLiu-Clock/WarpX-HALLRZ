#!/usr/bin/env python3
"""Compare HallRZ 1/2/3-level AMR layouts on the same finest grid.

The scan is a design tool, not a CI test.  It runs a full-finest single-level
reference and compares two-level and three-level static rectangular AMR against
that reference.  Error statistics use the HallRZ computational-region mask and
exclude regions owned by finer AMR levels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import picmi_hall_rz_core_smoke as core


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
BENCHMARK = THIS_DIR / "picmi_hall_rz_pic_benchmark.py"

FIELD_CENTERING = {
    "phi_fp": (0.0, 0.0),
    "Er_fp": (0.5, 0.0),
    "Ez_fp": (0.0, 0.5),
}


@dataclass(frozen=True)
class GridSpec:
    label: str
    rmax: float
    zmax: float
    nr_full: int
    nz_full: int


@dataclass(frozen=True)
class PatchSpec:
    status: str
    reason: str
    requested_ratio: float
    model_ratio: float
    r1: float | None = None
    z1: float | None = None
    r2: float | None = None
    z2: float | None = None


def token(value) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace("x", "x")


def parse_grid(value: str) -> tuple[int, int]:
    text = value.lower().replace(",", "x")
    parts = text.split("x")
    if len(parts) != 2:
        raise ValueError(f"grid must be Nr x Nz, got {value}")
    nr, nz = int(parts[0]), int(parts[1])
    if nr <= 0 or nz <= 0:
        raise ValueError("grid sizes must be positive")
    if nr % 4 != 0 or nz % 4 != 0:
        raise ValueError("full grid sizes must be divisible by 4")
    return nr, nz


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def steady_step_time(summary: dict) -> float:
    records = summary.get("chunk_records", [])
    if len(records) <= 1:
        return float(summary["avg_step_wall_time_s"])
    steps = sum(int(record["steps"]) for record in records[1:])
    wall = sum(float(record["wall_time_s"]) for record in records[1:])
    return wall / max(steps, 1)


def squeeze_field(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"expected 2D field or trailing singleton field, got {array.shape}")
    return array


def field_coords(shape: tuple[int, int], dx: float, dz: float, centering: tuple[float, float]):
    i = np.arange(shape[0], dtype=np.float64)
    j = np.arange(shape[1], dtype=np.float64)
    rr, zz = np.meshgrid((i + centering[0]) * dx, (j + centering[1]) * dz, indexing="ij")
    return rr, zz


def interp2d(reference: np.ndarray, xi: np.ndarray, yi: np.ndarray):
    nx, ny = reference.shape
    valid = (xi >= 0.0) & (yi >= 0.0) & (xi <= nx - 1) & (yi <= ny - 1)
    x0 = np.floor(np.clip(xi, 0.0, nx - 1)).astype(np.int64)
    y0 = np.floor(np.clip(yi, 0.0, ny - 1)).astype(np.int64)
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    wx = np.clip(xi - x0, 0.0, 1.0)
    wy = np.clip(yi - y0, 0.0, 1.0)
    value = (
        (1.0 - wx) * (1.0 - wy) * reference[x0, y0]
        + wx * (1.0 - wy) * reference[x1, y0]
        + (1.0 - wx) * wy * reference[x0, y1]
        + wx * wy * reference[x1, y1]
    )
    return value, valid


def computational_region_mask(summary: dict, rr: np.ndarray, zz: np.ndarray) -> np.ndarray:
    geom = summary["geometry"]
    lob = float(geom["lob"])
    hib = float(geom["hib"])
    out = float(geom["out"])
    eps = 1.0e-14
    return (zz >= out - eps) | ((rr >= lob - eps) & (rr <= hib + eps))


def error_stats(error: np.ndarray, reference: np.ndarray) -> dict:
    if error.size == 0:
        return {"count": 0, "l2_abs": 0.0, "l2_rel": 0.0, "linf_abs": 0.0, "linf_rel": 0.0}
    l2_abs = float(np.sqrt(np.mean(error * error)))
    linf_abs = float(np.max(np.abs(error)))
    l2_ref = float(np.sqrt(np.mean(reference * reference)))
    linf_ref = float(np.max(np.abs(reference)))
    return {
        "count": int(error.size),
        "l2_abs": l2_abs,
        "l2_rel": l2_abs / max(l2_ref, 1.0e-300),
        "linf_abs": linf_abs,
        "linf_rel": linf_abs / max(linf_ref, 1.0e-300),
    }


def snap_up(value: float, dx: float, strict_gt: float, upper: float) -> float | None:
    idx = int(math.ceil(value / dx - 1.0e-12))
    snapped = idx * dx
    while snapped <= strict_gt + 1.0e-12:
        idx += 1
        snapped = idx * dx
    if snapped >= upper - 1.0e-12:
        return None
    return snapped


def two_level_patch(spec: GridSpec, requested_ratio: float) -> PatchSpec:
    dx0 = spec.rmax / (spec.nr_full // 2)
    dz0 = spec.zmax / (spec.nz_full // 2)
    min_r1 = snap_up(core.HIB, dx0, core.HIB, spec.rmax)
    min_z1 = snap_up(core.OUT, dz0, core.OUT, spec.zmax)
    if min_r1 is None or min_z1 is None:
        return PatchSpec("infeasible", "channel does not fit in two-level patch", requested_ratio, 0.0)
    min_area_fraction = (min_r1 / spec.rmax) * (min_z1 / spec.zmax)
    max_ratio = 1.0 / (0.25 + min_area_fraction)
    if requested_ratio > max_ratio:
        return PatchSpec(
            "ready_infeasible_target",
            "two-level target ratio exceeds maximum; using minimal channel-covering patch",
            requested_ratio,
            max_ratio,
            r1=min_r1,
            z1=min_z1,
        )
    target_area_fraction = max(1.0 / requested_ratio - 0.25, min_area_fraction)
    scale = math.sqrt(target_area_fraction / min_area_fraction)
    r1 = snap_up(min_r1 * scale, dx0, core.HIB, spec.rmax)
    z1 = snap_up(min_z1 * scale, dz0, core.OUT, spec.zmax)
    if r1 is None or z1 is None:
        return PatchSpec("infeasible", "two-level requested patch exceeds domain", requested_ratio, max_ratio)
    model_ratio = 1.0 / (0.25 + (r1 / spec.rmax) * (z1 / spec.zmax))
    return PatchSpec("ready", "", requested_ratio, model_ratio, r1=r1, z1=z1)


def three_level_patch(spec: GridSpec, requested_ratio: float, level1_area_factor: float) -> PatchSpec:
    dx0 = spec.rmax / (spec.nr_full // 4)
    dz0 = spec.zmax / (spec.nz_full // 4)
    dx1 = dx0 / 2.0
    dz1 = dz0 / 2.0
    min_r2 = snap_up(core.HIB, dx1, core.HIB, spec.rmax)
    min_z2 = snap_up(core.OUT, dz1, core.OUT, spec.zmax)
    if min_r2 is None or min_z2 is None:
        return PatchSpec("infeasible", "channel does not fit in level-2 patch", requested_ratio, 0.0)
    min_r1 = snap_up(min_r2, dx0, min_r2, spec.rmax)
    min_z1 = snap_up(min_z2, dz0, min_z2, spec.zmax)
    if min_r1 is None or min_z1 is None:
        return PatchSpec("infeasible", "level-1 patch does not fit", requested_ratio, 0.0)

    min_a2 = (min_r2 / spec.rmax) * (min_z2 / spec.zmax)
    min_a1 = (min_r1 / spec.rmax) * (min_z1 / spec.zmax)
    max_ratio = 1.0 / (1.0 / 16.0 + min_a1 / 4.0 + min_a2)
    if requested_ratio > max_ratio * (1.0 + 1.0e-12):
        return PatchSpec(
            "infeasible",
            "three-level target ratio exceeds maximum channel-covering ratio",
            requested_ratio,
            max_ratio,
        )

    target_den = 1.0 / requested_ratio
    desired_a2 = (target_den - 1.0 / 16.0) / (1.0 + level1_area_factor / 4.0)
    desired_a2 = max(desired_a2, min_a2)
    target_area2 = desired_a2 * spec.rmax * spec.zmax
    min_area2 = min_r2 * min_z2
    t2 = math.sqrt(max(target_area2 / min_area2, 1.0))
    r2 = snap_up(min_r2 * t2, dx1, core.HIB, spec.rmax)
    z2 = snap_up(min_z2 * t2, dz1, core.OUT, spec.zmax)
    if r2 is None or z2 is None:
        return PatchSpec("infeasible", "level-2 target patch exceeds domain", requested_ratio, max_ratio)
    t1 = math.sqrt(level1_area_factor)
    r1 = snap_up(r2 * t1, dx0, r2, spec.rmax)
    z1 = snap_up(z2 * t1, dz0, z2, spec.zmax)
    if r1 is None or z1 is None:
        return PatchSpec("infeasible", "level-1 target patch exceeds domain", requested_ratio, max_ratio)
    a2 = (r2 / spec.rmax) * (z2 / spec.zmax)
    a1 = (r1 / spec.rmax) * (z1 / spec.zmax)
    model_ratio = 1.0 / (1.0 / 16.0 + a1 / 4.0 + a2)
    return PatchSpec("ready", "", requested_ratio, model_ratio, r1=r1, z1=z1, r2=r2, z2=z2)


def case_paths(output_dir: Path, prefix: str, case: str) -> dict[str, Path]:
    stem = f"{prefix}_{case}"
    return {
        "summary": output_dir / f"{stem}.json",
        "snapshot": output_dir / f"{stem}.npz",
        "log": output_dir / f"{stem}.log",
    }


def common_command(args, spec: GridSpec, case: str, paths: dict[str, Path], nr: int, nz: int) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK),
        "--case",
        case,
        "--rmax",
        f"{spec.rmax:.17g}",
        "--zmax",
        f"{spec.zmax:.17g}",
        "--nr",
        str(nr),
        "--nz",
        str(nz),
        "--max-steps",
        str(args.steps),
        "--chunk-steps",
        str(args.chunk_steps),
        "--summary-json",
        str(paths["summary"]),
        "--snapshot-npz",
        str(paths["snapshot"]),
        "--warpx-max-grid-size",
        str(args.max_grid_size),
        "--hall-rz-amr-phi-init-weight",
        str(args.amr_phi_init_weight),
        "--eb-bc-mode",
        "neumann",
        "--scalar-boundaries",
        "--particle-mode",
        "fixed",
        "--static-rho-test",
        "--static-rho-mode",
        str(args.static_rho_mode),
        "--static-rho-amp",
        str(args.static_rho_amp),
        "--static-rho-lambda-r",
        str(args.static_rho_lambda_r),
        "--static-rho-lambda-z",
        str(args.static_rho_lambda_z),
    ]
    return command


def full_command(args, spec: GridSpec, case: str, paths: dict[str, Path]) -> list[str]:
    command = common_command(args, spec, case, paths, spec.nr_full, spec.nz_full)
    command += ["--hall-rz-max-coarsening-level", str(args.full_mcl)]
    return command


def two_command(args, spec: GridSpec, patch: PatchSpec, case: str, paths: dict[str, Path]) -> list[str]:
    command = common_command(args, spec, case, paths, spec.nr_full // 2, spec.nz_full // 2)
    command += [
        "--hall-rz-amr-enable",
        "--hall-rz-amr-max-level",
        "1",
        "--hall-rz-amr-r1",
        f"{patch.r1:.17g}",
        "--hall-rz-amr-z1",
        f"{patch.z1:.17g}",
        "--hall-rz-max-coarsening-levels",
        str(args.two_mcls[0]),
        str(args.two_mcls[1]),
    ]
    return command


def three_command(args, spec: GridSpec, patch: PatchSpec, case: str, paths: dict[str, Path]) -> list[str]:
    command = common_command(args, spec, case, paths, spec.nr_full // 4, spec.nz_full // 4)
    command += [
        "--hall-rz-amr-enable",
        "--hall-rz-amr-max-level",
        "2",
        "--hall-rz-amr-r1",
        f"{patch.r1:.17g}",
        "--hall-rz-amr-z1",
        f"{patch.z1:.17g}",
        "--hall-rz-amr-r2",
        f"{patch.r2:.17g}",
        "--hall-rz-amr-z2",
        f"{patch.z2:.17g}",
        "--hall-rz-max-coarsening-levels",
        str(args.three_mcls[0]),
        str(args.three_mcls[1]),
        str(args.three_mcls[2]),
    ]
    return command


def run_case(command: list[str], log_path: Path, summary_path: Path, reuse_existing: bool) -> str:
    if reuse_existing and summary_path.exists():
        return "reused"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return "pass" if result.returncode == 0 else f"failed_returncode_{result.returncode}"


def field_cells(summary: dict, level: int | None = None) -> int:
    if level is None:
        shape = summary["fields"]["rho_fp"]["shape"]
    else:
        shape = summary["field_levels"][level]["fields"]["rho_fp"]["shape"]
    return int(shape[0]) * int(shape[1])


def level_extent(snapshot, summary: dict, level: int) -> tuple[float, float]:
    phi = squeeze_field(snapshot[f"lev{level}_phi_fp"])
    dx, dz = [float(v) for v in summary["amr_grid"]["levels"][level]["cell_size"]]
    return (phi.shape[0] - 1) * dx, (phi.shape[1] - 1) * dz


def compare_to_full(full_path: Path, amr_path: Path) -> dict:
    with np.load(full_path) as full_npz, np.load(amr_path) as amr_npz:
        full_summary = json.loads(full_npz["summary_json"].item())
        amr_summary = json.loads(amr_npz["summary_json"].item())
        finest_level = int(amr_summary["amr_grid"]["finest_level"])
        full_geom = full_summary["geometry"]
        full_dx = (float(full_geom["rmax"]) - float(full_geom["rmin"])) / int(full_geom["nr"])
        full_dz = float(full_geom["zmax"]) / int(full_geom["nz"])
        level_extents = {
            lev: level_extent(amr_npz, amr_summary, lev) for lev in range(finest_level + 1)
        }
        metrics = {}
        for field in FIELD_CENTERING:
            all_err = []
            all_ref = []
            max_record = {
                "abs": -1.0,
                "level": -1,
                "r": 0.0,
                "z": 0.0,
                "target": 0.0,
                "reference": 0.0,
            }
            for lev in range(finest_level + 1):
                centering = FIELD_CENTERING[field]
                target = squeeze_field(amr_npz[f"lev{lev}_{field}"])
                reference = squeeze_field(full_npz[f"lev0_{field}"])
                dx, dz = [float(v) for v in amr_summary["amr_grid"]["levels"][lev]["cell_size"]]
                rr, zz = field_coords(target.shape, dx, dz, centering)
                owned = np.ones(target.shape, dtype=bool)
                finer_extent = level_extents.get(lev + 1)
                if finer_extent is not None:
                    eps = 1.0e-14
                    owned &= ~((rr <= finer_extent[0] + eps) & (zz <= finer_extent[1] + eps))
                xi = rr / full_dx - centering[0]
                yi = zz / full_dz - centering[1]
                ref_value, valid = interp2d(reference, xi, yi)
                mask = owned & valid & computational_region_mask(amr_summary, rr, zz)
                err = target[mask] - ref_value[mask]
                ref = ref_value[mask]
                if err.size > 0:
                    local = int(np.argmax(np.abs(err)))
                    abs_err = float(abs(err[local]))
                    if abs_err > max_record["abs"]:
                        flat_indices = np.flatnonzero(mask)
                        ii, jj = np.unravel_index(flat_indices[local], target.shape)
                        max_record = {
                            "abs": abs_err,
                            "level": lev,
                            "r": float(rr[ii, jj]),
                            "z": float(zz[ii, jj]),
                            "target": float(target[ii, jj]),
                            "reference": float(ref_value[ii, jj]),
                        }
                all_err.append(err)
                all_ref.append(ref)
            err = np.concatenate(all_err) if all_err else np.empty(0)
            ref = np.concatenate(all_ref) if all_ref else np.empty(0)
            stats = error_stats(err, ref)
            metrics[f"{field}_count"] = stats["count"]
            metrics[f"{field}_l2_rel"] = stats["l2_rel"]
            metrics[f"{field}_linf_rel"] = stats["linf_rel"]
            metrics[f"{field}_linf_abs"] = stats["linf_abs"]
            metrics[f"{field}_max_level"] = max_record["level"]
            metrics[f"{field}_max_r"] = max_record["r"]
            metrics[f"{field}_max_z"] = max_record["z"]
            metrics[f"{field}_max_target"] = max_record["target"]
            metrics[f"{field}_max_reference"] = max_record["reference"]
        metrics["E_l2_rel_max"] = max(metrics["Er_fp_l2_rel"], metrics["Ez_fp_l2_rel"])
        if metrics["Er_fp_linf_abs"] >= metrics["Ez_fp_linf_abs"]:
            metrics["E_linf_field"] = "Er_fp"
            prefix = "Er_fp"
        else:
            metrics["E_linf_field"] = "Ez_fp"
            prefix = "Ez_fp"
        metrics["E_linf_abs"] = metrics[f"{prefix}_linf_abs"]
        metrics["E_linf_rel"] = metrics[f"{prefix}_linf_rel"]
        metrics["E_linf_level"] = metrics[f"{prefix}_max_level"]
        metrics["E_linf_r"] = metrics[f"{prefix}_max_r"]
        metrics["E_linf_z"] = metrics[f"{prefix}_max_z"]
        return metrics


def row_from_case(
    spec: GridSpec,
    layout: str,
    requested_ratio: float,
    patch: PatchSpec | None,
    full_summary: dict,
    full_paths: dict[str, Path],
    summary: dict,
    paths: dict[str, Path],
    status: str,
) -> dict:
    full_steady = steady_step_time(full_summary)
    steady = steady_step_time(summary)
    full_cells = field_cells(full_summary)
    solve_cells = sum(field_cells(summary, lev) for lev in range(len(summary["field_levels"])))
    row = {
        "grid": spec.label,
        "layout": layout,
        "requested_ratio": requested_ratio,
        "model_ratio": 1.0 if patch is None else patch.model_ratio,
        "actual_cell_ratio": full_cells / max(solve_cells, 1),
        "full_steady_s_per_step": full_steady,
        "steady_s_per_step": steady,
        "speedup_vs_full": full_steady / max(steady, 1.0e-300),
        "full_cells": full_cells,
        "solve_cells": solve_cells,
        "r1": "" if patch is None or patch.r1 is None else patch.r1,
        "z1": "" if patch is None or patch.z1 is None else patch.z1,
        "r2": "" if patch is None or patch.r2 is None else patch.r2,
        "z2": "" if patch is None or patch.z2 is None else patch.z2,
        "checks_pass": summary["checks"]["pass"],
        "status": status,
        "reason": "" if patch is None else patch.reason,
        "summary": str(paths["summary"]),
        "log": str(paths["log"]),
    }
    if layout == "one":
        row.update({
            "phi_fp_l2_rel": 0.0,
            "E_l2_rel_max": 0.0,
            "phi_fp_linf_abs": 0.0,
            "phi_fp_linf_rel": 0.0,
            "phi_fp_max_level": 0,
            "phi_fp_max_r": 0.0,
            "phi_fp_max_z": 0.0,
            "E_linf_field": "",
            "E_linf_abs": 0.0,
            "E_linf_rel": 0.0,
            "E_linf_level": 0,
            "E_linf_r": 0.0,
            "E_linf_z": 0.0,
        })
    else:
        metrics = compare_to_full(full_paths["snapshot"], paths["snapshot"])
        row.update(metrics)
    return row


def write_reports(rows: list[dict], csv_path: Path, md_path: Path) -> None:
    fieldnames = [
        "grid", "layout", "requested_ratio", "model_ratio", "actual_cell_ratio",
        "full_steady_s_per_step", "steady_s_per_step", "speedup_vs_full",
        "full_cells", "solve_cells", "r1", "z1", "r2", "z2",
        "phi_fp_l2_rel", "E_l2_rel_max", "phi_fp_linf_abs", "phi_fp_linf_rel",
        "phi_fp_max_level", "phi_fp_max_r", "phi_fp_max_z",
        "E_linf_field", "E_linf_abs", "E_linf_rel", "E_linf_level", "E_linf_r", "E_linf_z",
        "checks_pass", "status", "reason", "summary", "log",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    lines = [
        "# HallRZ Stage 4.8 AMR Level-Count Scan",
        "",
        "| grid | layout | actual ratio | steady s/step | speedup | phi L2 rel | E L2 rel max | phi Linf abs @ level/r/z | E Linf abs @ field/level/r/z | status |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        if row.get("status", "").startswith("failed") or not row.get("checks_pass", False):
            status = f"{row.get('status','')}: {row.get('reason','')}"
        else:
            status = row.get("status", "")
        lines.append(
            "| {grid} | {layout} | {ratio:.6g} | {steady:.6g} | {speed:.6g} | "
            "{phi:.6g} | {e:.6g} | {pa:.6g} @ L{pl}/{pr:.6g}/{pz:.6g} | "
            "{ea:.6g} @ {ef}/L{el}/{er:.6g}/{ez:.6g} | {status} |".format(
                grid=row.get("grid", ""),
                layout=row.get("layout", ""),
                ratio=float(row.get("actual_cell_ratio", 0.0)),
                steady=float(row.get("steady_s_per_step", 0.0)),
                speed=float(row.get("speedup_vs_full", 0.0)),
                phi=float(row.get("phi_fp_l2_rel", 0.0)),
                e=float(row.get("E_l2_rel_max", 0.0)),
                pa=float(row.get("phi_fp_linf_abs", 0.0)),
                pl=row.get("phi_fp_max_level", ""),
                pr=float(row.get("phi_fp_max_r", 0.0)),
                pz=float(row.get("phi_fp_max_z", 0.0)),
                ea=float(row.get("E_linf_abs", 0.0)),
                ef=row.get("E_linf_field", ""),
                el=row.get("E_linf_level", ""),
                er=float(row.get("E_linf_r", 0.0)),
                ez=float(row.get("E_linf_z", 0.0)),
                status=status,
            )
        )

    lines.extend(["", "Fastest passing layout by grid:", ""])
    lines.append("| grid | fastest layout | steady s/step | speedup vs full | note |")
    lines.append("|---|---|---:|---:|---|")
    by_grid: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("checks_pass", False) and not str(row.get("status", "")).startswith("failed"):
            by_grid.setdefault(str(row["grid"]), []).append(row)
    for grid in sorted(by_grid):
        winner = min(by_grid[grid], key=lambda r: float(r["steady_s_per_step"]))
        note = "full fastest" if winner["layout"] == "one" else "AMR candidate"
        lines.append(
            "| {grid} | {layout} | {steady:.6g} | {speed:.6g} | {note} |".format(
                grid=grid,
                layout=winner["layout"],
                steady=float(winner["steady_s_per_step"]),
                speed=float(winner["speedup_vs_full"]),
                note=note,
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-grids", nargs="+", default=["448x512"])
    parser.add_argument("--rmax", type=float, default=0.28)
    parser.add_argument("--zmax", type=float, default=0.32)
    parser.add_argument("--two-target-ratio", type=float, default=7.0)
    parser.add_argument("--three-target-ratio", type=float, default=6.0)
    parser.add_argument("--level1-area-factor", type=float, default=1.2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--chunk-steps", type=int, default=20)
    parser.add_argument("--max-grid-size", type=int, default=512)
    parser.add_argument("--full-mcl", type=int, default=30)
    parser.add_argument("--two-mcls", type=int, nargs=2, default=[30, 30])
    parser.add_argument("--three-mcls", type=int, nargs=3, default=[30, 30, 30])
    parser.add_argument("--amr-phi-init-weight", type=float, default=1.0)
    parser.add_argument("--static-rho-mode", type=int, default=1)
    parser.add_argument("--static-rho-amp", type=float, default=1.0e-10)
    parser.add_argument("--static-rho-lambda-r", type=float, default=0.004)
    parser.add_argument("--static-rho-lambda-z", type=float, default=0.006)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR)
    parser.add_argument("--prefix", default="hallrz_stage4_8")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0 or args.chunk_steps <= 0:
        raise ValueError("--steps and --chunk-steps must be positive")
    if any(mcl < 0 for mcl in [args.full_mcl, *args.two_mcls, *args.three_mcls]):
        raise ValueError("MCL values must be non-negative")
    if args.rmax <= core.HIB or args.zmax <= core.OUT:
        raise ValueError("domain must contain the HallRZ channel")

    rows = []
    for grid in args.full_grids:
        nr_full, nz_full = parse_grid(grid)
        spec = GridSpec(grid, args.rmax, args.zmax, nr_full, nz_full)
        full_case = f"{token(grid)}_one"
        full_paths = case_paths(args.output_dir, args.prefix, full_case)
        status = run_case(
            full_command(args, spec, f"{args.prefix}_{full_case}", full_paths),
            full_paths["log"],
            full_paths["summary"],
            args.reuse_existing,
        )
        if status.startswith("failed") or not full_paths["summary"].exists():
            rows.append({
                "grid": grid,
                "layout": "one",
                "checks_pass": False,
                "status": status,
                "reason": str(full_paths["log"]),
            })
            continue
        full_summary = load_json(full_paths["summary"])
        rows.append(
            row_from_case(
                spec, "one", 1.0, None, full_summary, full_paths,
                full_summary, full_paths, status,
            )
        )
        if not full_summary["checks"]["pass"]:
            continue

        two_patch = two_level_patch(spec, args.two_target_ratio)
        if two_patch.r1 is not None and two_patch.z1 is not None:
            two_case = f"{token(grid)}_two_r{token(args.two_target_ratio)}"
            two_paths = case_paths(args.output_dir, args.prefix, two_case)
            status = run_case(
                two_command(args, spec, two_patch, f"{args.prefix}_{two_case}", two_paths),
                two_paths["log"],
                two_paths["summary"],
                args.reuse_existing,
            )
            if status.startswith("failed") or not two_paths["summary"].exists():
                rows.append({
                    "grid": grid,
                    "layout": "two",
                    "requested_ratio": two_patch.requested_ratio,
                    "model_ratio": two_patch.model_ratio,
                    "checks_pass": False,
                    "status": status,
                    "reason": str(two_paths["log"]),
                })
            else:
                rows.append(
                    row_from_case(
                        spec, "two", args.two_target_ratio, two_patch, full_summary,
                        full_paths, load_json(two_paths["summary"]), two_paths, status,
                    )
                )
        else:
            rows.append({
                "grid": grid,
                "layout": "two",
                "requested_ratio": two_patch.requested_ratio,
                "model_ratio": two_patch.model_ratio,
                "checks_pass": False,
                "status": two_patch.status,
                "reason": two_patch.reason,
            })

        three_patch = three_level_patch(spec, args.three_target_ratio, args.level1_area_factor)
        if all(v is not None for v in (three_patch.r1, three_patch.z1, three_patch.r2, three_patch.z2)):
            three_case = f"{token(grid)}_three_r{token(args.three_target_ratio)}"
            three_paths = case_paths(args.output_dir, args.prefix, three_case)
            status = run_case(
                three_command(args, spec, three_patch, f"{args.prefix}_{three_case}", three_paths),
                three_paths["log"],
                three_paths["summary"],
                args.reuse_existing,
            )
            if status.startswith("failed") or not three_paths["summary"].exists():
                rows.append({
                    "grid": grid,
                    "layout": "three",
                    "requested_ratio": three_patch.requested_ratio,
                    "model_ratio": three_patch.model_ratio,
                    "checks_pass": False,
                    "status": status,
                    "reason": str(three_paths["log"]),
                })
            else:
                rows.append(
                    row_from_case(
                        spec, "three", args.three_target_ratio, three_patch, full_summary,
                        full_paths, load_json(three_paths["summary"]), three_paths, status,
                    )
                )
        else:
            rows.append({
                "grid": grid,
                "layout": "three",
                "requested_ratio": three_patch.requested_ratio,
                "model_ratio": three_patch.model_ratio,
                "checks_pass": False,
                "status": three_patch.status,
                "reason": three_patch.reason,
            })

    csv_path = args.output_dir / f"{args.prefix}_amr_level_count_scan.csv"
    md_path = args.output_dir / f"{args.prefix}_amr_level_count_scan.md"
    write_reports(rows, csv_path, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
