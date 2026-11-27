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

With `CICalc` loaded, a results archive (`CICalc.CICalcResults`, from
`load_results` or a run spec) is drawn by [`save_figures`](@ref) (everything it
supports), [`plot_sector`](@ref), [`plot_cascade`](@ref), and per state by
[`plot_densities`](@ref), [`plot_configurations`](@ref) and [`plot_orbital`](@ref);
with the amplitude tier, [`plot_natural_orbitals`](@ref) and [`plot_constituents`](@ref).
Point-group labels reach the single-particle figures through their `labels` keyword.

The labelling figures draw a `CICalc.LevelBundle` (`level_bundle(ks, analysis)`):
[`plot_levels`](@ref), [`plot_irrep_weights`](@ref), [`plot_operation_map`](@ref),
[`plot_symmetry_quality`](@ref), [`plot_charges`](@ref), [`plot_parent_mixing`](@ref),
[`plot_field_irreps`](@ref) and [`plot_sweep`](@ref) (a `SymmetrySweep`). Each
irrep has one colour in every figure ([`irrep_color`](@ref)), and "?" is grey.
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
export irrep_color, irrep_palette, UNLABELLED_COLOR, DISAGREE_COLOR

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

# ── one state of a CICalc results archive, defined by the CICalc extension ──
# A many-body state is a function of several positions, so its figure works off
# the REDUCED one-body densities the archive stores, not off anything that could
# be called |X(r)|². Load CICalc to get them.
function plot_densities end
function plot_configurations end
function plot_orbital end
export plot_densities, plot_configurations, plot_orbital

# ── labelling figures, defined by the CICalc extension ──────
# All draw a `CICalc.LevelBundle` (`level_bundle(ks, analysis; …)`), except
# `plot_sweep`, which draws a `SymmetrySweep`, and `plot_field_irreps`, which also
# takes `field_content` results. Each keeps its layout on an unlabelled bundle.
function plot_levels end
function plot_irrep_weights end
function plot_operation_map end
function plot_symmetry_quality end
function plot_charges end
function plot_parent_mixing end
function plot_field_irreps end
function plot_sweep end
export plot_levels, plot_irrep_weights, plot_operation_map, plot_symmetry_quality,
       plot_charges, plot_parent_mixing, plot_field_irreps, plot_sweep

# the radiative cascade (`CICalc.cascade_bundle`), defined by the CICalc extension
function plot_cascade end
export plot_cascade

# figures from a CICalc results archive (`CICalc.CICalcResults`): one sector's levels,
# and every figure the archive supports; defined by the CICalc extension
function plot_sector end
function save_figures end
export plot_sector, save_figures

# each archive figure as a panel in a grid position (`draw_*!(fig[1, 1], …)`), returning
# `(; layout, axis, size)`: a `plot_*` is a Figure around one of these, and the CICalc
# explorer is built from them
function draw_sector! end
function draw_densities! end
function draw_configurations! end
function draw_orbital! end
function draw_cascade! end
export draw_sector!, draw_densities!, draw_configurations!, draw_orbital!, draw_cascade!

# the amplitude tier of an archive (`orbitals: {amplitudes: true}`): a stored state's or
# level's natural orbitals, and a biexciton's exciton pair, as figures and as panels
function plot_natural_orbitals end
function draw_natural_orbitals! end
function plot_constituents end
function draw_constituents! end
export plot_natural_orbitals, draw_natural_orbitals!, plot_constituents, draw_constituents!

# the explorer of a CICalc results archive: its model (the state structure as a graph,
# and the selection on it), defined by the CICalc extension
function explorer_model end
function node_name end
function select! end
function focus! end
function back! end
function forward! end
function step! end
function follow! end
function find_levels end
function decay_tree end
function feeders end
function availability end
export explorer_model
public node_name, select!, focus!, back!, forward!, step!, follow!, find_levels, decay_tree,
       feeders, availability

end # module StateRendering
