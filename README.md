# pymuffintin

`pymuffintin` is a backend-neutral research package for muffin-tin
auxiliary-basis and Coulomb algorithm experiments: muffin-tin local-RI
metrics and cutoffs, interstitial ISDF/THC point selection, muffin-tin and
interstitial stitching, and fixed-orbital exchange ablations. It holds no
production physics of its own; it is a laboratory over exported orbital,
product, and Coulomb data.

## Boundary

The package is organized around provider protocols (`Orbitals`,
`LocalProduct`, `Coulomb` in `pymuffintin.providers`) and backend-neutral
array DTOs (`pymuffintin.contracts`). Algorithm code in `auxiliary/` and
`mbpt/` depends only on those contracts, never on a specific backend.

[`libmuffintin`](https://github.com/weiyiguo9/libmuffintin) is the default
backend: it holds the stable reference kernels (SPEX mixed-product auxiliary
basis, k-point ISDF/THC, the Weinert/SPEX Coulomb operator) behind the
provider protocols. It is imported lazily, only inside
`pymuffintin.backends.muffintin`, so `pymuffintin` itself still imports on a
machine without the native extension built, and a foreign-dump adapter
(SPEX, FLEUR, CoQui, ...) can implement the same protocols as a first-class
alternative. The dependency direction is strictly `pymuffintin` to
`libmuffintin`, never the reverse.

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
| `backends.muffintin` | The only module importing `libmuffintin`; adapts its pyexport v1 arrays to the contracts above. |
| `auxiliary.lri` | Muffin-tin local-RI auxiliary fitting via a per-site overlap eigendecomposition. |
| `auxiliary.thc` | Weighted deterministic QRCP interpolative separable density fitting (ISDF) for the interstitial region. |
| `auxiliary.hybrid` | Concatenates a muffin-tin local-RI block with an interstitial THC block into one hybrid auxiliary representation, including the muffin-tin/interstitial cross block. |
| `mbpt.hf` | Fixed-orbital Fock exchange and reference-versus-trial ablation. |
| `tensor` | Backend-neutral contraction IR and host-side linear-algebra primitives; see "Tensor backend" below. |

## Tensor backend

All fixed-structure multilinear contractions in `auxiliary/` and `mbpt/`
route through `pymuffintin.tensor.contract`, an `opt_einsum`-backed IR that
compiles and caches one expression per `(subscript, operand shapes)` pair
and evaluates it with backend dispatch following the operands' own array
type. A future CTF backend plugs in by registering an object with a `name`
and `asarray`/`to_host` methods (`tensor.register_backend`,
`tensor.set_backend`); no call site changes.

`tensor.eigh`, `tensor.lstsq`, and `tensor.pinv` are host-side gather
points by declaration, not by omission: they always run on numpy arrays,
because they either lack a distributed CTF-native equivalent or a caller
depends on numpy's exact ordering. The deterministic weighted-QRCP column
selection in `auxiliary/thc.py` is not routed through `tensor` at all and
stays fully sequential and host-side, because Gate 2's same-engine THC
reproduction depends on its exact pivot order.

## Install

```sh
# libmuffintin's native extension, built once with maturin develop
# inside the shared venv (see libmuffintin/doc/21 Stage 1 acceptance):
cd /path/to/libmuffintin/python && maturin develop

cd /path/to/pymuffintin
pip install -e ".[test]"
```

## Test

```sh
pytest tests/ -q
```

Gate 3 (`tests/test_gate3.py`) exercises `libmuffintin` through
`MuffintinAdapter` on the tracked hydrogen fixture; it is skipped
automatically if the native extension is not built. Every other test,
including the multi-band exchange regression in `tests/test_hf.py`, is
backend-neutral and requires no native import. Gate 3's reported
$E_x$ and $\Sigma_x$ differences are a pipeline consistency check on that
fixture, not a material-accuracy claim.

## License

Apache-2.0. See [LICENSE](LICENSE).
