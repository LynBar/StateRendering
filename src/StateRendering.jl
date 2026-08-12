"""
    StateRendering

Visualisation for k·p states, built on [`StateBundle`](@ref) from
`MatrixFreeKPSolver`.

# Getting started

    using GLMakie, MatrixFreeKPSolver, StateRendering

    E, ψ, info = eigenstates(H, 4; method = SchurComplement(branch = :conduction))
    b = state_bundle(H, E, ψ; geometry = geom, info = info)

    explore(b)                  # interactive browser
    save("e1.pdf", plot_slices(b, 1))   # publication figure (CairoMakie)

# Two things this package will not do

  * **Guess a band basis.** Every label, colour and component grouping comes from
    the bundle, which took it from the operator. Six valence bands are
    |HH,LH,SO⟩ for zinc-blende and |X,Y,Z⟩ for wurtzite; nothing but the labels
    distinguishes them, so nothing here infers them from the band count.
  * **Assume a backend.** Everything except [`explore`](@ref) is drawn as meshes,
    heatmaps and lines, so CairoMakie can render it to PDF or SVG. Load GLMakie
    for interaction, CairoMakie for figures.

# What is here

| | |
|:--|:--|
| [`explore`](@ref) | interactive: state menu, per-band toggles, overlays |
| [`plot_state`](@ref), [`plot_states`](@ref) | isosurfaces, one or many |
| [`plot_slices`](@ref), [`plot_bands`](@ref) | quantitative cuts, whole state or band by band |
| [`plot_composition`](@ref), [`plot_spectrum`](@ref) | band character up the ladder; level diagram |
| [`plot_field`](@ref) | strain, potential or composition on its own |
| [`ScalarField`](@ref), [`VectorField`](@ref), [`StrainField`](@ref) | overlays the solver does not produce |
| [`isolevel`](@ref) | isosurfaces chosen by enclosed probability |

Primitives (`add_density!`, `add_geometry!`, `add_scalar!`, `add_vectors!`,
`add_slice!`, `add_component_bars!`, `add_levels!`) each draw one thing into an
axis you own, for figures that are not in the list above.

With `HDF5` loaded, `load_h5` reads a state file written by another solver into a
bundle; see its docstring for the band-basis argument it requires.
"""
module StateRendering

using Makie
using Printf
using GeometryBasics: GLTriangleFace, normal_mesh
using MatrixFreeKPSolver: StateBundle, BandBasis, state_density, band_weights,
                          component_weights, dominant_component, nbands

import Meshing

include("palette.jl")
include("fields.jl")
include("levels.jl")
include("primitives.jl")
include("figures.jl")
include("explore.jl")

# ── overlays the solver does not produce ──────────────────────────────────
export ScalarField, VectorField, StrainField
export hydrostatic, biaxial, scalar_fields, magnitude

# ── display heuristics ────────────────────────────────────────────────────
export isolevel, enclosed_fraction, isosurface_mesh, centroid, smooth

# ── colours, keyed by band label ──────────────────────────────────────────
export component_color, component_palette, band_palette, kind_color

# ── primitives: draw one thing into an axis you own ───────────────────────
export state_axis3, add_density!, add_geometry!, add_scalar!, add_vectors!
export add_slice!, add_component_bars!, add_levels!

# ── figures ───────────────────────────────────────────────────────────────
export plot_state, plot_states, plot_slices, plot_bands
export plot_composition, plot_spectrum, plot_field
export explore

# defined by the HDF5 extension
function load_h5 end
export load_h5

end # module StateRendering
