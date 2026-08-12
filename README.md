# StateRendering

Visualisation for k·p states, built on the `StateBundle` produced by
[`MatrixFreeKPSolver`](../MatrixFreeKPSolver).

```julia
using GLMakie, MatrixFreeKPSolver, StateRendering

grid = Grid((32, 32, 32), (24.0, 24.0, 24.0), Dirichlet())
geom = SphereDot((12.0, 12.0, 12.0), 5.0)
H    = eight_band_hamiltonian(grid, geom, InAs, GaAs; δ = 1.5 * minimum(grid.dx))

E, ψ, info = eigenstates(H, 4; method = SchurComplement(branch = :conduction))
b = state_bundle(H, E, ψ; kind = :electron, geometry = geom, info = info)

explore(b)                              # interactive browser
save("fig2.pdf", plot_slices(b, 1))     # publication figure (under CairoMakie)
```

## Two rules the package holds to

**The band basis is never inferred.** Every label, colour and grouping comes from
the bundle, which took it from the operator. A six-band valence block is
`HH↑ LH↑ LH↓ HH↓ SO↑ SO↓` for zinc-blende and `X↑ Y↑ Z↑ X↓ Y↓ Z↓` for wurtzite —
the same band *count*, a different basis. Nothing here reads `size(ψ, 4)` and
decides what the bands are.

**The backend is not assumed.** Everything except `explore` is drawn with meshes,
heatmaps and lines, so CairoMakie renders it to PDF or SVG. Isosurfaces are
extracted with marching cubes rather than GPU ray-marching precisely so that they
can go on paper. Load GLMakie when you want to spin the view, CairoMakie when you
want a file.

## Isosurfaces mean enclosed probability

`isolevel(d, 0.5)` is the level enclosing half the charge. That is comparable
between a peaked s-like state and a diffuse p-like one, between an electron and a
hole, and between grid resolutions. `0.5 * maximum(d)` is none of those things —
it describes the array, not the physics. Every function here that draws a surface
takes `fraction`, not a raw level.

```julia
lvl = isolevel(state_density(b, 1), 0.5)
enclosed_fraction(state_density(b, 1), lvl)   # ≈ 0.5, by construction
```

## Figures

| | |
|:--|:--|
| `explore(b)` | interactive: state menu, per-band toggles, overlays, live weights |
| `plot_state(b, i)` | one state: isosurface, dot outline, component weights |
| `plot_states(b)` | a panel per state, same convention throughout |
| `plot_slices(b, i)` | orthogonal cuts through the state's own centroid |
| `plot_bands(b, i)` | one cut per spinor band, labelled and weighted |
| `plot_composition(b)` | band character up the ladder, stacked |
| `plot_spectrum(b)` | level diagram, coloured by dominant component |
| `plot_field(b, f)` | strain, potential or composition on its own |

Each returns a `Figure`; pass it to `save`.

`plot_bands` is the one to reach for when a solve looks wrong. A conduction state
that is 85% S with a few percent each of HH, LH and SO is a correctly coupled
8-band state; the same state coming back 85% HH means the basis, the branch or
the operator is not what you thought.

One caveat on 3D under CairoMakie: transparent meshes are composited triangle by
triangle, so overlapping isosurfaces show faint seams. Fine for a figure you are
reading, less so for a cover image — use GLMakie for those, or prefer
`plot_slices`, which carries more information per square inch anyway.

## Primitives

Each adds one thing to an axis you own, so a figure the list above does not cover
is still a few lines:

```julia
fig = Figure()
ax  = state_axis3(fig[1, 1]; title = "e₁ and h₁")
add_density!(ax, electrons, 1; fraction = 0.5, color = :kind)
add_density!(ax, holes,     1; fraction = 0.5, color = :kind)
add_geometry!(ax, electrons)
```

`add_density!`, `add_geometry!`, `add_scalar!`, `add_vectors!`, `add_slice!`,
`add_component_bars!`, `add_levels!`. All draw in nanometres, taken from the
bundle, so overlays land in the same space and lengths can be read off the axis.

## Overlays the solver does not produce

Strain, piezoelectric potential and polarisation come from elsewhere. They get
typed wrappers so a sign convention travels with the data:

```julia
ε = StrainField(; εxx, εyy, εzz, εxy, εxz, εyz)   # keyword-only, by design
P = VectorField(Px, Py, Pz, "P")
V = ScalarField(potential, "piezo potential"; signed = true, unit = "V")

explore(b; strain = ε, vectors = P, fields = [V])
plot_field(b, ScalarField(biaxial(ε), "biaxial"; signed = true))
```

`StrainField` takes its six components by keyword because the usual failure mode
is a bare six-vector in whatever order the file used: `(XX, XY, XZ, YY, YZ, ZZ)`
and `(XX, YY, ZZ, XY, XZ, YZ)` both exist in the wild, and swapping them silently
turns `εyy` into `εxy`. `hydrostatic` and `biaxial` are provided because those,
not the raw components, are what move the band edges and split HH from LH.

Signed fields draw with a diverging map about zero and symmetric ± isosurfaces —
which is the right picture for a quadrupolar piezoelectric potential, where the
two lobes are the whole story.

## Reading files from another solver

With `HDF5` loaded, `load_h5` reads an electronic-structure file into bundles.
It requires a `BandBasis`, because a file stores an array and not what its band
axis means:

```julia
using HDF5
basis = BandBasis(["s↑","pₓ↑","p_y↑","p_z↑","s↓","pₓ↓","p_y↓","p_z↓"],
                  "s" => [1,5], "pₓ" => [2,6], "p_y" => [3,7], "p_z" => [4,8])
el, ho, ov = load_h5("Basic.h5", basis; L = (30.0, 30.0, 20.0))
explore(el; strain = ov.strain, vectors = ov.polarization)
```

`BandBasis` checks that the groups cover every band exactly once, so a mislabelled
basis fails at construction rather than producing a plausible, wrong pie chart.
Pass `L` in nm — files rarely record the box, and without it every length in every
figure is off by a constant factor.

## Testing

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite runs under CairoMakie, which is deliberate: a test that only passed
under GLMakie would not catch a GPU-only primitive sneaking into a figure meant
for print. It also checks the things that are easy to get silently wrong —
that mesh vertices are in nanometres and not grid indices (by comparing an
extracted sphere's volume against `4/3 πR³`), that `isolevel` really encloses the
fraction it claims, and that zinc-blende and wurtzite bundles produce different
panel titles.
