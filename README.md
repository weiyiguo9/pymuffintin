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

`omt` fits a periodic constant plus continuous radial hats and reports the
potential-sphere overlap/error curve. `kink` consumes screened slopes and the
exact radial boundary energy jets exported by `libmuffintin`, while `nmto`
forms ordinary and confluent matrix divided differences, active-channel Schur
downfolding, and strict Löwdin-orthogonalized Hamiltonians. All required
square systems use `tensor.solve`/`tensor.inv`; the V&D and NMTO constructions
never replace invertibility with a pseudoinverse.

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

## Tensor backend

All fixed-structure multilinear contractions in `auxiliary/` and `mbpt/`
route through `pymuffintin.tensor.contract`, an `opt_einsum`-backed IR that
compiles and caches one expression per `(subscript, operand shapes)` pair
and evaluates it with backend dispatch following the operands' own array
type. A future CTF backend plugs in by registering an object with a `name`
and `asarray`/`to_host` methods (`tensor.register_backend`,
`tensor.set_backend`); no call site changes.

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
