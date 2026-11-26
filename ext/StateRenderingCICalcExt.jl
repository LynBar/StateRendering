module StateRenderingCICalcExt

# ---------------------------------------------------------------------------
# Drawing CICalc results: the labelling figures (LevelBundle), the radiative
# cascade, and everything a results archive (CICalcResults) holds.
#
# Loaded only when CICalc is, so the package itself stays free of TensorKit.
#
# The one idea worth stating: a many-body state is a function of several
# positions, so there is no |X(r)|² to isosurface. What the archive stores instead
# is each particle's reduced one-body density with the others traced out — a
# genuine observable, non-negative, normalised — and that is what the state
# figures draw. Plotting |C|²-weighted single-particle orbitals instead would throw
# away the interference between configurations, which is the content of the CI.
#
# Everything here is drawn as meshes, heatmaps and lines, so CairoMakie can put it
# in a paper.
# ---------------------------------------------------------------------------

using Makie
using Makie: latexstring
using Printf
using CICalc: LevelBundle, SymmetrySweep, CascadeBundle, line_yield,
              CICalcResults, StoredSector, HeavyRef, SectorSolution,
              exciton_pair_weights, species_weights, _config_name, SymmetryFrame, subduction
using CICalc.Symmetry: labelled
using StateRendering: isolevel, isosurface_mesh, smooth, state_axis3, kind_color,
                      irrep_color, component_color, component_palette,
                      UNLABELLED_COLOR, DISAGREE_COLOR
import StateRendering: plot_densities, plot_configurations, plot_orbital,
                       plot_levels, plot_irrep_weights, plot_operation_map,
                       plot_symmetry_quality, plot_charges, plot_parent_mixing,
                       plot_field_irreps, plot_sweep, plot_cascade, plot_sector,
                       save_figures, draw_sector!, draw_densities!, draw_configurations!,
                       draw_orbital!, draw_cascade!, explorer_model, node_name, select!,
                       focus!, back!, forward!, step!, follow!, find_levels, decay_tree, feeders,
                       availability, explore

include("StateRenderingCICalcExt/states.jl")     # one state from an archive: densities, configurations, orbitals
include("StateRenderingCICalcExt/labelling.jl")
include("StateRenderingCICalcExt/cascade.jl")   # needs labelling.jl's ink colours
include("StateRenderingCICalcExt/archive.jl")   # figures from a CICalcResults archive
include("StateRenderingCICalcExt/model.jl")     # the explorer model: the state structure as a graph
include("StateRenderingCICalcExt/explore.jl")   # the explorer: the map, the lenses, the 3-D window

end # module StateRenderingCICalcExt
