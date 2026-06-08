# HallRZ Python EB Neumann Data

This note documents the first Python boundary-data interface for the HallRZ
Poisson path.

## Scope

Python can provide cell-centered EB Neumann data:

```python
from pywarpx import hallrz

hallrz.set_eb_neumann(g_eb_2d)
hallrz.clear_eb_neumann()
hallrz.has_eb_neumann()
```

Python can also provide node-centered physical boundary data:

```python
hallrz.set_robin_zhi(a, b, f)
hallrz.clear_robin_zhi()
hallrz.has_robin_zhi()

hallrz.set_robin_rhi(a, b, f)
hallrz.clear_robin_rhi()
hallrz.has_robin_rhi()

hallrz.set_robin_rlo(a, b, f)
hallrz.clear_robin_rlo()
hallrz.has_robin_rlo()

hallrz.set_dirichlet_rlo(phi)
hallrz.clear_dirichlet_rlo()
hallrz.has_dirichlet_rlo()

hallrz.set_inlet_dirichlet(phi)
hallrz.clear_inlet_dirichlet()
hallrz.has_inlet_dirichlet()
```

The array stores physical data only:

```text
g_EB = dphi/dn, n = fluid-to-covered EB normal
```

It is not EB geometry.  WarpX C++ still owns EB2 geometry and HallRZ nodal FVM
geometry.

## Array Contract

`g_eb_2d` must be:

```text
shape = (Nr, Nz)
dtype = np.float64
C-contiguous
```

The data are cell-centered:

```text
i = 0..Nr-1 in r
j = 0..Nz-1 in z
```

C++ consumes values only on AMReX EB2 `singleValued` cut cells.  Values on
regular or covered cells are ignored.

## PICMI Call Order

Use this order:

```python
sim.initialize_inputs()
hallrz.set_eb_neumann(g_eb_2d)
hallrz.set_robin_zhi(a_zhi, b_zhi, f_zhi)
hallrz.set_robin_rhi(a_rhi, b_rhi, f_rhi)
hallrz.set_robin_rlo(a_rlo, b_rlo, f_rlo)      # optional, only for rmin > 0
hallrz.set_dirichlet_rlo(phi_rlo)              # optional, only for rmin > 0
hallrz.set_inlet_dirichlet(phi_inlet)
sim.initialize_warpx()
sim.step(...)
```

`initialize_inputs()` is needed before loading the WarpX pybind module because
the geometry dimension selects the shared library.  Setting the array before
`initialize_warpx()` ensures the initial field solve also uses the Python data.

For per-step updates:

```python
sim.initialize_inputs()
hallrz.set_eb_neumann(g0)
sim.initialize_warpx()

for n in range(nsteps):
    hallrz.set_eb_neumann(g_n)
    sim.step(1)
```

To return to the default/scalar path:

```python
hallrz.clear_eb_neumann()
```

If scalar wall flux is disabled, clearing the Python data gives `g_EB = 0`.

## Physical Boundary Arrays

Physical boundary arrays are node-centered 1D arrays.

`z-hi` Robin:

```text
a_zhi.shape = b_zhi.shape = f_zhi.shape = (Nr+1,)
a*phi + b*dphi/dz = f at z=zmax
```

`r-hi` Robin:

```text
a_rhi.shape = b_rhi.shape = f_rhi.shape = (Nz+1,)
a*phi + b*dphi/dr = f at r=rmax
```

C++ only uses r-hi Python data on active Hall plume boundary nodes.  Inactive
covered-side r-hi nodes keep the default harmless Robin data so AMReX face
validation remains well-defined.

`r-lo` Robin, valid only for `rmin > 0`:

```text
a_rlo.shape = b_rlo.shape = f_rlo.shape = (Nz+1,)
a*phi + b*dphi/dn = f at r=rmin, n = -r
```

`r-lo` Dirichlet, valid only for `rmin > 0`:

```text
phi_rlo.shape = (Nz+1,)
phi = phi_rlo at r=rmin
```

At `rmin=0`, the lower radial face is the RZ axis and only homogeneous
Neumann/axis symmetry is supported.

`z-lo` inlet Dirichlet:

```text
phi_inlet.shape = (Nr+1,)
phi = phi_inlet at z=zmin
```

C++ only applies inlet values on active channel inlet nodes.

All physical boundary arrays must be:

```text
dtype = np.float64
C-contiguous
```

For active Robin nodes, `abs(a) + abs(b) > 0` is required.  C++ aborts if an
active Robin node has `a=b=0`.

Each face is managed independently:

```text
set_*   -> persistent until overwritten or cleared
clear_* -> return that face to the scalar/default path
```

## PICMI Solver Parameters

HallRZ scalar setup can be attached directly to the PICMI electrostatic solver:

```python
solver = picmi.ElectrostaticSolver(
    grid=grid,
    method="Multigrid",
    required_precision=1.0e-6,
    warpx_hall_rz_enable=True,
    warpx_hall_rz_lob=lob,
    warpx_hall_rz_hib=hib,
    warpx_hall_rz_out=out,
    warpx_hall_rz_bc_lo_r="dirichlet",  # auto, axis, dirichlet, neumann, robin
    warpx_hall_rz_potential_lo_r=0.0,
)
```

For `rmin > 0`, supported lo-r Poisson BCs are:

```text
dirichlet: scalar value from warpx_hall_rz_potential_lo_r or grid warpx_potential_lo_r
neumann:   homogeneous dphi/dn = 0
robin:     scalar a,b,f from warpx_hall_rz_robin_lo_r_a/b/f
```

Use `hallrz.set_robin_rlo(...)` or `hallrz.set_dirichlet_rlo(...)` after
`sim.initialize_inputs()` when the lo-r data must vary along z.

## Mutual Exclusion

Python EB Neumann arrays and scalar wall flux are mutually exclusive.

If Python data are active, keep:

```text
warpx.hall_rz_eb_wall_flux = 0
warpx.hall_rz_eb_g0 = 0
```

If both Python data and scalar wall flux are active, C++ aborts to avoid double
injection.

## Smoke Commands

Scalar reference:

```bash
python3 Test/HallRZ/picmi_hall_rz_core_smoke.py --max-steps 20
```

Constant Python EB data:

```bash
python3 Test/HallRZ/picmi_hall_rz_core_smoke.py --max-steps 20 --python-eb-g0 1.0
```

Segment masks:

```bash
for s in 0 1 2 3; do
  python3 Test/HallRZ/picmi_hall_rz_core_smoke.py \
    --max-steps 1 --python-eb-g0 1.0 --python-eb-segment ${s}
done
```

Time-varying data:

```bash
python3 Test/HallRZ/picmi_hall_rz_core_smoke.py \
  --max-steps 100 --python-eb-g0 1.0 --python-eb-time-varying
```

Clear behavior:

```bash
python3 Test/HallRZ/picmi_hall_rz_core_smoke.py \
  --max-steps 20 --python-eb-g0 1.0 --python-eb-clear-after 10
```

Aligned `rmin > 0` lower-r boundary checks:

```bash
python3 Test/HallRZ/picmi_hall_rz_core_smoke.py \
  --max-steps 1 --rmin 0.02 --rlo-bc dirichlet

python3 Test/HallRZ/picmi_hall_rz_core_smoke.py \
  --max-steps 1 --rmin 0.02 --rlo-bc neumann

python3 Test/HallRZ/picmi_hall_rz_core_smoke.py \
  --max-steps 1 --rmin 0.02 --rlo-bc robin
```

## Current Validation

The Python array path has been checked against scalar `hall_rz_eb_g0 = 1`.

Observed behavior:

```text
q_EB_total_RZ == q_EB_added_RZ
q_EB_diff_RZ = 0 or roundoff
epsilon_q = 0 or roundoff
max Final Iter <= 7 in the tested smoke cases
field and particle NaN/Inf counts are zero
```

Physical boundary arrays have been checked with:

```text
z-hi Robin: a=1, b=1, f=0
r-hi Robin: a=1, b=1, f=0
z-hi + r-hi together
z-lo inlet Dirichlet with nonzero active-channel values
inlet Dirichlet per-step update
```

These checks preserve EB flux closure for EB-only data and keep field/particle
NaN/Inf counts at zero.

Note: AMReX's current HallRZ diagnostic helper applies all inhomogeneous
Neumann/Robin RHS terms together.  If physical Robin `f` is nonzero,
`q_EB_added_RZ` is no longer a pure EB-only diagnostic.  Use `f=0` when checking
EB-only flux balance until a separate domain-boundary diagnostic is added.
