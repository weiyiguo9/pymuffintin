# Frozen-potential bcc Fe LAPW--NMTO comparison

`fe_lapw_nmto.py` compares LAPW and NMTO band energies generated from the
same `libmuffintin.CheckpointPhysics` frozen potential.  It is deliberately
limited to a one-site primitive cell, scalar/nonmagnetic radial data, and the
nine real-harmonic channels with `l = 0, 1, 2`.  It is **not** a
self-consistent NMTO DFT calculation and does not produce an NMTO density or
total energy.

## Python atomic-start producer

`materialize_atomic_starts.py` constructs neutral nonmagnetic bcc Fe and
diamond C directly through the method-neutral `libmuffintin.Structure`,
`RegionalFieldLayout`, and `materialize_atomic_start` API.  The atomic regional
field layout is explicit: it uses `g_cutoff = 4.0 bohr^-1` and muffin-tin
`l_max = 4`, independently of any compiled LAPW basis.  This field cutoff is
the density/potential layout for the atomic start; the authored V3 LAPW inputs
retain their separate orbital-envelope `g-cutoff = 2.0 bohr^-1`.

For the diamond accuracy setting with LAPW orbital `g-cutoff = 4.0 bohr^-1`,
use the requested three-times potential/density plane-wave cutoff and
`l_max = 6` density layout explicitly:

```sh
python experiment/materialize_atomic_starts.py diamond-c \
  --field-g-cutoff 12.0 --field-muffin-tin-l-max 6
```

The two cutoffs are independent: `4.0 bohr^-1` belongs to the LAPW/APW orbital
envelope, while `12.0 bohr^-1` belongs to the regional potential and density.

The default output is a new directory under
`../local_experiment/generated/python_atomic_starts/`; the producer refuses to replace an existing
checkpoint.  An explicit new directory may be supplied instead:

```sh
python experiment/materialize_atomic_starts.py fe-bcc
python experiment/materialize_atomic_starts.py diamond-c \
  --output-dir /path/to/new/diamond-start
```

Each run prints the checkpoint path, finite-layout charge-closure diagnostics,
and the checkpoint SHA-256 digest.  Record those values together with the
`libmuffintin` producer repository/ref.  The producer writes only the atomic
checkpoint; use a matching authored V3 SCF input with an explicit `g-cutoff`
for later LAPW work.

## Fe provenance

`../local_experiment/generated/fe_bcc` contains the historical atomic-superposition checkpoint and
the frozen-potential results derived from it.  It predates the Python producer
and is not overwritten or retroactively relabelled.  Do not use the hydrogen
fixtures as Fe data.  New runs should record:

- a frozen-potential checkpoint, including its producer repository/ref,
  producer input/run record, and file digest;
- the matching scalar `libmuffintin` input whose k mesh and LAPW band window
  define the comparison points.

The radial energy mesh is expressed in the checkpoint's absolute Hartree
reference.  The driver reads the interstitial potential's `G=0` component and
passes `energy_mesh - V_I(G=0)` to the USW free-space branch; that difference
uses the decaying branch below zero and the real standing-wave/Neumann branch
above zero.
The checkpoint must expose scalar frozen radial data through
`sample_frozen_scalar_radials`; magnetic/spinor Fe is outside this driver's
scope.

## Run

With `libmuffintin` and this package installed in the active environment:

```sh
python experiment/fe_lapw_nmto.py \
  --checkpoint /path/to/provenance-linked/bcc-fe-checkpoint.toml \
  --input /path/to/provenance-linked/bcc-fe-input.toml \
  --energy-mesh=-0.40,-0.20,-0.05 \
  --cluster-shell-size 59 \
  --output fe_lapw_nmto.npz
```

`--cluster-shell-size` is a minimum target.  The driver includes the entire
real-space translation shell containing that target; for a bcc primitive
lattice the default target is the complete 59-site cluster.

## NPZ output

The archive contains:

- provenance/scope metadata: `comparison`, `checkpoint`, `input`,
  `q_fractional`, `site_id`, `atomic_number`, `muffin_tin_radius`;
- geometry and sampling: `direct_lattice`, `reciprocal_lattice`,
  `k_fractional`, `k_cartesian`, `energy_mesh`,
  `interstitial_potential_zero`, `interstitial_energy_mesh`, `real_harmonic_l`,
  `real_harmonic_m`, `cluster_integer_translations`, and
  `cluster_cartesian_translations`;
- LAPW reference data: `lapw_band_window_start`, `lapw_spins`, and
  `lapw_energies` with shape `(n_spin, n_k, n_lapw_band)`;
- NMTO data: `nmto_energies` with shape `(n_k, 9)`, plus
  `nmto_coefficients` in the nonorthogonal NMTO basis and
  `nmto_orthonormal_coefficients` in the Löwdin basis.

The archive also stores the independently filled LAPW and NMTO chemical
potentials and the two spectra measured from their own chemical potentials.
This alignment is for comparing dispersions; it does not change the USW
kinetic energy `E - V_I`.

Both band sets come from the same frozen checkpoint potential.  Agreement is
a representation comparison for that supplied dataset, not cross-code or
self-consistent material validation.

## Diamond-C comparison

The diamond comparison will use one provenance-linked carbon checkpoint and
keep three quantities separate:

1. USW v&d versus the LAPW-exported interstitial field, measured on common
   points as a representation error;
2. `PeriodicDmk` versus an independent reciprocal-space solve for the same
   zero-mean charge source, measured as a periodic Hartree-potential error;
3. LAPW versus NMTO bands on the same frozen effective potential.

`../local_experiment/generated/diamond_c` contains the historical frozen-potential band experiment,
and `../local_experiment/generated/diamond_c_pseudo` contains the later exact-charge atomic start.
Both retain their original checkpoint provenance and remain distinct from new
Python-produced starts.  Replacing either checkpoint without regenerating its
derived NPZ/CSV data would break the recorded provenance.

`diamond_periodic_dmk.py` compares `PeriodicDmk` with an independent analytic
reciprocal Poisson solve for a strictly three-dimensional, zero-mean Legendre
source in the diamond conventional cubic cell.  This is a kernel validation,
not a material-density comparison: `export_frozen_potential()` is an effective
potential and is not substituted for the Hartree source.

`random_sphere_usw.py` remains a Python-only, fixed-seed method-geometry
robustness experiment.  Its spheres have centers and radii but no chemical
species, so they are deliberately not converted into atomic starts.

### Self-consistent comparison driver

`diamond_scf_comparison.py` closes the reproducible Python workflow.  It runs
the native LAPW SCF, writes the converged restart checkpoint and histories,
then evaluates two separate comparison axes:

1. dense `PeriodicDmk` versus `FastPeriodicDmk` for the same converged regional
   charge projected into a conventional cubic diamond cell;
2. LAPW versus USW/NMTO bands on the same converged effective potential and
   the same primitive-fractional G-X-L-G path.

The path driver defaults to the complete 135-translation shell, the smallest
tested shell that keeps the converged-potential NMTO overlap positive on every
path point without the memory cost of the historical 177-translation shell.
The periodic DMK pair runs in `periodic_dmk_compare.py`, a clean Python process
that avoids loading FINUFFT into the same process as the native DFT extension.

This is deliberately not labelled “FastDMK versus USW”: FastDMK is a Coulomb
backend and USW/NMTO is an orbital representation.  Keeping the axes separate
makes each numerical difference attributable.  Run from the repository root:

```sh
python experiment/diamond_scf_comparison.py \
  --input local_experiment/generated/diamond_c_pseudo/diamond_c_input.toml \
  --output-dir local_experiment/generated/diamond_c_scf_comparison
```

The output directory is required to be new.  It contains the converged restart
checkpoint, a matching restart input, DMK source/result archives, and
`diamond_scf_comparison.npz` with SCF history, raw and aligned Hartree bands,
density-projection provenance, and both periodic DMK potentials.

### Independent NMTO SCF entry point

`pymuffintin.mto.run_nmto_scf(path_or_input)` runs the Python-owned NMTO SCF
loop directly from a TOML path or a prepared `NmtoScfInput`.  It consumes the
method-neutral checkpoint structure, radial samples, regional density and
potential kernels; it does not call the LAPW orbital solver and does not need a
LAPW-converged potential.  With symmetry enabled it solves only the regular
k-mesh representatives, uses orbit weights, projects the post-core regional
density with the same active subgroup, and exposes a method-neutral restart
checkpoint on the result.

## Current numerical results

- the pseudo-density atomic-superposition extension produces an exact
  12-electron diamond start; LAPW SCF then converges in 23 iterations to
  `-71.180456493637379 Ha` at the stated 2x2x2 and orbital
  `g-cutoff = 2.0 bohr^-1` experiment settings;
- diamond USW v&d constant-field volume: relative error
  `-1.174967658936e-3` (the published Table-I value is `-1.17e-3`);
- diamond conventional-cell 3D `PeriodicDmk`: maximum potential error
  `3.998727697840e-10 Ha`, relative L2 error `2.511232991715e-7`;
- diamond frozen LAPW--NMTO, N=1 mesh `[-0.1, 0.4] Ha`, 159-cell cluster:
  after independent eight-valence-electron chemical-potential alignment, the
  first six bands have RMS `5.861646964980e-2 Ha` and maximum error
  `1.063816680989e-1 Ha`;
- Fe frozen LAPW--NMTO, N=1 mixed decaying/standing mesh `[-0.3, 0.3] Ha`,
  159-cell cluster: the same aligned first-six-band RMS is
  `1.914285922861e-1 Ha`, with maximum error `3.645472050095e-1 Ha`;
- the exact 26-electron Fe atomic-superposition checkpoint is generated from
  zero, but the crystal SCF currently stops in the four-component core solve:
  the `1s, kappa=-1` bracket scan reports two node-compatible roots.  Raising
  the scan from 512 to 2048 intervals reproduces the same ambiguity, so it is
  not reported as a converged Fe material result;
- the fixed-seed random non-overlapping sphere experiment has maximum analytic
  slope-derivative error `1.444484532431e-9` and weighted-slope symmetry error
  `5.551115123126e-17` across negative and positive interstitial energies.

The chemical-potential alignment compares dispersion.  It does not replace
the independent `E - V_I` energy reference used by the USW envelopes, and none
of these frozen-potential numbers is a self-consistent NMTO total energy.
