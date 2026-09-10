# pymuffintin

`pymuffintin` is the application layer and Python interface for
[`libmuffintin`](https://github.com/weiyiguo9/libmuffintin). It builds
Python-level electronic-structure workflows and algorithms on top of the
native library, including SCF orchestration, MTO/NMTO methods, auxiliary-basis
construction, Coulomb algorithms, and exchange calculations.

The intended architectural analogy is to a libcint-style separation between
a reusable computational library and its applications: `libmuffintin` is
the underlying library, while `pymuffintin` provides the Python-facing
application layer.

## Boundary

The native computational kernels and bindings belong to `libmuffintin`;
Python application logic, workflow orchestration, and algorithm experiments
belong to `pymuffintin`. The dependency direction is strictly `pymuffintin`
to `libmuffintin`, never the reverse.

The auxiliary-basis and exchange paths use provider protocols (`Orbitals`,
`LocalProduct`, `Coulomb` in `pymuffintin.providers`) and backend-neutral
array DTOs (`pymuffintin.contracts`). Algorithm code in `auxiliary/` and
`mbpt/` depends only on those contracts, never on a specific backend.

`libmuffintin` supplies the native kernels (SPEX mixed-product auxiliary
basis, k-point ISDF/THC, the Weinert/SPEX Coulomb operator) behind the
provider protocols. `pymuffintin` itself can still be imported without the
native extension built; workflows that use native kernels require it.
Foreign-dump adapters can implement the same protocols for comparison and
experiments, but these backend-neutral contracts do not change the package's
role as the application layer for `libmuffintin`.

The `MuffintinAdapter` in `backends/muffintin.py` consumes the
`libmuffintin.pyexport` schema (version 1), not Rust types directly, so it
stays pinned on the schema version rather than on `libmuffintin`'s internal
object model. See `libmuffintin`'s
[doc 21](https://github.com/weiyiguo9/libmuffintin/blob/main/doc/21_python_binding_and_export_schema.md)
for the schema and the pair-column convention
$k\,N_{\mathrm{orb}}^2 + i\,N_{\mathrm{orb}} + j$ used throughout.

## Modules

| Module | Content |
|---|---|
| `contracts` | Backend-neutral array DTOs: orbital windows, pair samples, auxiliary representations, Coulomb blocks, fixed occupations. |
| `providers` | `Orbitals` / `LocalProduct` / `Coulomb` protocols that any backend implements. |
| `backends.muffintin` | Adapts `libmuffintin` pyexport v1 arrays to the contracts above. |
| `auxiliary.lri` | Muffin-tin local-RI auxiliary fitting via a per-site overlap eigendecomposition. |
| `auxiliary.thc` | Weighted deterministic QRCP interpolative separable density fitting (ISDF) for the interstitial region. |
| `auxiliary.hybrid` | Concatenates a muffin-tin local-RI block with an interstitial THC block into one hybrid auxiliary representation, including the muffin-tin/interstitial cross block. |
| `coulomb` | Free-space and cubic-periodic 3D continuous-source DMK-lite over tensor-product Legendre leaf densities. |
| `mto` | Real-harmonic unitary spherical waves, screened slope matrices, and value-and-derivative interpolation. |
| `regional` | Point sampling of exported interstitial-Fourier plus muffin-tin radial scalar fields. |
| `symmetry` | Unified `SymmetryDataset` mirroring the Rust `muffintin_symmetry` IR, detected via spglib, with spgrep little-group scalar and spinor irrep helpers. |
| `spex_log` | SPEX stdout parser: operation table (with time reversal), atom basis, and IBZ table, exported as a `libmuffintin.spexsym` v1 file for the Rust reader. Space-group classification lines are ignored. |
| `mbpt.hf` | Fixed-orbital Fock exchange and reference-versus-trial ablation. |
| `tensor` | Backend-neutral contraction IR and host-side linear-algebra primitives; see "Tensor backend" below. |

## Continuous Coulomb DMK-lite

`pymuffintin.coulomb.ContinuousDmk` applies the free-space `1/r` kernel to
piecewise tensor-product Legendre densities on a complete, validated 2:1
dyadic leaf partition. It uses the telescoping Coulomb split

```text
coarse Gaussian band
  + dyadic Gaussian correction bands
  + localized erfc(r) / r remainder
```

Each smooth Gaussian convolution is evaluated with three one-dimensional NumPy
matrix products followed by a fixed contraction through `tensor.contract`; a
boundary-projected Duffy cubature handles singular and near-singular local
remainder interactions. This is a correctness-first DMK-lite
reference. It does not yet implement DMK's short plane-wave translations or
upward/downward passes, so no linear-complexity claim is made.

`pymuffintin.coulomb.PeriodicDmk` reuses the same leaf/tree contract for a
three-dimensional cubic cell. Its top level is the neutral, zero-mean Ewald
decomposition: a real-space `erfc(alpha*r)/r` image sum plus the reciprocal
`k != 0` Coulomb multiplier. Reciprocal leaf moments and target evaluation are
dense tensor transforms in this reference implementation.

`project_density` supplies the common cubic density boundary used by the
periodic reference and fast implementations: it projects a scalar callable
onto uniform dyadic tensor-Legendre leaves and can remove the volume-weighted
zero mode required by the periodic Green function.

Both classes are low-level density-to-potential reference kernels. They do not
implement the q-resolved `providers.Coulomb` auxiliary-matrix protocol and are
not yet adapters for `build_hybrid_coulomb` or `mbpt.hf`.

FINUFFT is a recommended optional dependency for a later independent Fourier
translation/comparison path:

```sh
pip install -e ".[nufft]"
```

The continuous-source design follows the separation used by Flatiron's
[`dmk`](https://github.com/flatironinstitute/dmk): tensor-product box
densities and Gaussian transforms remain distinct from point-source NUFFT
machinery. The cubic-periodic reference follows the Ewald normalization and
independent-reference strategy in
[`PeriodicDMK`](https://github.com/xuanzhaogao/PeriodicDMK); general lattice
cells and a FINUFFT reciprocal evaluator remain future work.

## MTO interpolation laboratory

`pymuffintin.mto` implements the pure-Python
[Nohara--Andersen](https://arxiv.org/abs/1604.08097) value-and-derivative
construction. `usw` builds bare
real-harmonic structure matrices, screens them by real-space cluster
inversion, evaluates unitary spherical waves, and provides the periodic Bloch
sum. `vd` forms four-energy divided differences, the four super-unitary
functions through third radial derivative, and the minimum-norm fifth-energy
constraint weights used for open structures.

`omt` is the overlapping-muffin-tin (Andersen's OMTA, not "optimized")
potential construction: `fit_omt` fits a periodic constant plus continuous
radial hats and reports the potential-sphere overlap/error curve, and
`fit_omt_shells` is the pinned-interior variant used by `run_nmto_scf`.
`shell` extends the radial mesh into overlapping potential shells and
back-extrapolates free waves to the hard spheres. `kink` consumes screened
slopes and the exact radial boundary energy jets exported by `libmuffintin`,
while `nmto` forms ordinary and confluent matrix divided differences,
active-channel Schur downfolding, and strict Löwdin-orthogonalized
Hamiltonians. All required square systems use `tensor.solve`/`tensor.inv`;
the V&D and NMTO constructions never replace invertibility with a
pseudoinverse.

The independent `run_nmto_scf` path uses periodic spherical-boundary Green
operators, not inverse-first finite-cluster Bloch folding. A negative-reference
lattice sum and a reciprocal resolvent correction define the same periodic
envelope for matrices and density at either sign of the interstitial energy.
Install `pymuffintin[nmto]` for FINUFFT evaluation. The explicit numerical
controls replace `minimum-cells` in `[task.scf.nmto]`:

```toml
energy-mesh = [-0.1, 0.2]
reciprocal-cutoff = 12.6
lattice-sum-radius = 16.0
reference-energy = -1.0
matrix-angular-order = 8
```

These are example settings, not converged diamond parameters. The reference
energy is a numerical splitting parameter, not a shift of the physical energy
mesh. Periodic orbitals retain inactive angular continuation inside spheres,
scalar-relativistic mass-weighted flux, and radial/tangential small-component
norms. There is no sampled-overlap remapping or nonpositive-mode clipping.
The Hamiltonian receives the nonconstant interstitial and nonspherical MT
potential matrix before solving bands and occupations, including the spherical
potential acting on inactive free partial waves. `matrix-angular-order`
controls sphere quadrature. Select `[task.scf.mixing] kind = "linear"` or
`"pulay"`; Pulay also requires `history`. Both retain the specified `beta`.

### Production recipe

`run_nmto_scf` executes exactly one construction, and every checkpoint it
writes records it as `nmto.recipe.*` annotations (see `recipe_annotations`),
so two "FP-NMTO" results can be told apart without reading the call chain.
The laboratory modules `vd` and `coulomb.*` are **not** on this path: the
density and potential live in the `libmuffintin` regional field (muffin-tin
harmonics plus interstitial Fourier components) and the Coulomb problem is
solved by the native regional solver.

| Dimension | `reference-potential = "spherical-mt"` (default) | `reference-potential = "omt"` |
|---|---|---|
| Reference potential | spherical average inside each hard sphere, constant `V_I(G=0)` outside | overlapping wells: the same spherical average inside the hard sphere `a`, a least-squares piecewise-linear tail on `a < r <= s`, and a least-squares constant `V0` |
| Sphere radii | one radius `a` per site plays every role: USW hard sphere, kink sphere, augmentation partition, regional-field boundary | the same `a`, plus potential radii `s >= a` that may overlap |
| Radial functions | native solve to `a`; kink jets at `a` | native solve of the well to `s`; free continuation `u0` back to `a` matching value and mass-weighted flux at `s`; kink jets from `u0` at `a` |
| Augmentation | own active partial wave inside the own hard sphere, envelope elsewhere | additionally `(u-u0)/u0(a)` on every shell, additive across overlapping shells and into other hard spheres |
| Full-potential matrix | direct quadrature of `V - V_ref` per orbital piece | the same, with `V - V0 - v_R` for every well piece wherever it lives; the `-v_R` shell terms use a per-site shell quadrature (Gauss--Legendre between the tail knots times the sphere angular rule) |
| Envelope, kink matrix, NMTO order, Löwdin, density, Coulomb | periodic hard-sphere USW; unchanged | unchanged |

With `s = a` the OMT recipe reproduces the default construction except for
the reference constant. Requested potential radii are rounded up to the
next native exponential-mesh point so the tail knots are mesh points:

```toml
[task.scf.nmto]
reference-potential = "omt"
potential-radius-scale = 1.2      # s = 1.2 a for every site, or per species:
[task.scf.nmto.potential-radii]
C = 1.75                           # bohr, keyed by the site-id prefix
```

The reference Hamiltonian and overlap still come from the kink formalism,
which treats each shell function as an exact solution of its own well. In
an overlap region that is Andersen's usual OMTA approximation, of second
order in the potential overlap; the explicit `V - V_ref` matrix corrects the
potential, not that kinetic-energy error. Hard spheres must not overlap in
either recipe. Results carry `nmto.scf.reference_constant_hartree`, the fit
RMS `nmto.scf.reference_fit_rms_hartree`, the nearest-neighbour distance,
and the maximum hard- and potential-sphere overlap fractions;
`NmtoScfResult` keeps `reference_constant_history` and
`reference_rms_history` per iteration.

The energy convention is Hartree throughout, with the wave equation written as
`(-nabla^2/2 - E) psi = 0`. The finite-cluster constant-density regression uses
all 25 real harmonics through `l_max=4` and reproduces the published Table I
interstitial-volume errors for bcc (`N_R=51`, `a=0.8t`) and diamond
(`N_R=159`, `a=0.8t`). This is a fixed-parameter Python oracle, not a Rust
production density representation.

The frozen-hydrogen OMT regression embeds the checkpoint's spherical radial
potential on a fixed grid and checks the overlap fractions and decreasing
weighted-RMS trend for potential-sphere radii 3.2, 4.0, and 4.8 Bohr. The
frozen-hydrogen NMTO/LAPW regression compares a second-order, s-channel NMTO at
Gamma with the lowest Gamma LAPW eigenvalue exported from the same checkpoint
using `g-cutoff = 5.0 Bohr^-1`; the two differ by less than 2 mHa. These are
same-checkpoint representation-pipeline regressions, not material or cross-code
accuracy claims.

### MPI NMTO-SCF

Install the optional dependencies with `pip install -e ".[nmto,mpi]"` and pass
the caller's communicator explicitly:

```python
from mpi4py import MPI
from pymuffintin.mto import run_nmto_scf

result = run_nmto_scf("input.toml", comm=MPI.COMM_WORLD)
if MPI.COMM_WORLD.rank == 0:
    print(result.total_energy)
```

Run the script with, for example,
`mpiexec -n 4 python -m mpi4py run_nmto.py`. All ranks must call the solver
with the same input. Without `comm`, the ordinary serial route is used.
In MPI mode, only communicator rank 0 returns `NmtoScfResult`; other ranks
return `None`. Consume results and create restart checkpoints on rank 0.
The application owns MPI initialization/finalization and output; the solver
does not initialize a second native MPI scheduler or write checkpoints.

Periodic USW construction and k-point NMTO solves are rank-distributed.
Full-potential integration and density sampling use independent work blocks.
Numerical buffers use MPI-3 shared windows (one copy per shared-memory domain), with
inter-node transfers performed by node leaders. Shared windows live for one
iteration; returned results do not borrow their storage.

Occupations use the complete weighted spectrum and a common chemical
potential. Interstitial density coefficients use the serial uniform-grid FFT,
and normalization follows the assembled global density. Scalar XC uses fixed
muffin-tin radial-shell and interstitial point blocks, independent of rank
count. Workers borrow node-shared numerical inputs and write disjoint output
blocks; rank 0 integrates the assembled arrays in their original order. Small
interstitial FFTs and Hartree construction remain local to rank 0. Core-state
searches and fixed-energy radial solves are separate rank-distributed tasks.
Only rank 0 assembles complete native potential/core objects, evaluates energy,
mixes density, and constructs the restart result. Workers do not reconstruct
complete native density or potential objects from the shared inputs.
Small spectra and result matrices are also replicated. Rank-local tensor
contractions continue through `tensor.contract` on NumPy blocks, not through
collective CTF calls inside independently scheduled tasks.

Use one native thread per MPI rank: set `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `TBLIS_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, and `RAYON_NUM_THREADS=1` before launch. This
execution model does not require a no-GIL Python build. MPI consistency
checks are not diamond material-accuracy or scaling acceptance.

For opt-in phase measurements, pass
`timing_callback=lambda iteration, phase, seconds: print(iteration, phase, seconds)`
to `run_nmto_scf`. Only rank 0 invokes it; ordinary runs do not print timings.
Subphase timings are nested (for example, FFT planning is part of potential
preparation, and radial solving is part of NMTO); do not sum all callbacks.
The native Python extension enables `fft-fftw` by default and requires system
FFTW libraries. On Homebrew macOS, set
`LIBRARY_PATH="$(brew --prefix fftw)/lib"` when building the extension.

The focused collective checks can be run with
`mpiexec -n 4 python -m mpi4py -m pytest tests/test_nmto_mpi.py tests/test_tensor_ctf.py -q`.
The CTF check requires its separately installed Python binding. A single
machine exercises shared-memory execution, not physical cross-node scaling.

## Tensor backend

All fixed-structure multilinear contractions in `auxiliary/` and `mbpt/`
route through `pymuffintin.tensor.contract`, an `opt_einsum`-backed IR that
compiles and caches one expression per `(subscript, operand shapes)` pair
and evaluates it with backend dispatch following the operands' own array
type. The optional `tensor.CtfBackend` uses CTF's own Python binding through
the existing `asarray`/`to_host` interface:

```python
from pymuffintin import tensor

backend = tensor.CtfBackend()  # CTF must already be installed against the same MPI
tensor.register_backend(backend)
tensor.set_backend("ctf")
a = backend.asarray(host_a)
b = backend.asarray(host_b)
c = tensor.contract("ab,bc->ac", a, b)
host_c = backend.to_host(c)
```

CTF conversions and contractions must be entered collectively by its world
with compatible shapes. Selecting a backend does not convert NumPy operands
automatically or change their rank-local contraction behavior. CTF is not
used as a second scheduler inside NMTO's rank-local work. The adapter does
not select a CTF subcommunicator or manage MPI's lifetime.

`tensor.eigh`, `tensor.solve`, `tensor.inv`, `tensor.lstsq`, and
`tensor.pinv` are host-side gather
points by declaration, not by omission: they always run on numpy arrays,
because they either lack a distributed CTF-native equivalent or a caller
depends on numpy's exact ordering. The deterministic weighted-QRCP column
selection in `auxiliary/thc.py` is not routed through `tensor` at all and
stays fully sequential and host-side because reproducible weighted-QRCP ISDF
point selection depends on its exact pivot order.

## Install

```sh
# libmuffintin's native extension, built once with maturin develop
# inside the shared venv (see libmuffintin/doc/21 Python-binding acceptance):
cd /path/to/libmuffintin/python && maturin develop

cd /path/to/pymuffintin
pip install -e ".[test]"
```

## Test

```sh
pytest tests/ -q
```

The native exchange test (`tests/test_muffintin_exchange_pipeline.py`) exercises
`libmuffintin` through `MuffintinAdapter` on the tracked hydrogen fixture. The
frozen-checkpoint MTO tests (`tests/test_mto_hydrogen_checkpoint.py`) also use
the native extension. Contributor-facing test scope and fixture interpretation
are documented in [AGENTS.md](AGENTS.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
