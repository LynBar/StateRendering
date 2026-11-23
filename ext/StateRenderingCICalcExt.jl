module StateRenderingCICalcExt

# ---------------------------------------------------------------------------
# Drawing exciton CI results.
#
# Loaded only when CICalc is, so the package itself stays free of TensorKit.
#
# The one idea worth stating: |X> = Σ C_eh |e>|h> is a function of TWO
# positions, so there is no |X(r)|² to isosurface. What `ExcitonBundle` carries
# instead is each particle's reduced one-body density with the other traced out
# — a genuine observable, non-negative, normalised — and that is what these
# figures draw. Plotting |C|²-weighted single-particle orbitals instead would
# throw away the interference between configurations, which is the entire
# content of the CI.
#
# Everything here is drawn as meshes, heatmaps and lines, so CairoMakie can put
# it in a paper. Nothing takes a `StateBundle`: an exciton figure works off raw
# arrays and the bundle's own coordinates.
# ---------------------------------------------------------------------------

using Makie
using Makie: latexstring
using Printf
using CICalc: ExcitonBundle, LevelBundle, SymmetrySweep
using CICalc.Symmetry: labelled
using StateRendering: isolevel, isosurface_mesh, smooth, state_axis3, kind_color,
                      irrep_color, UNLABELLED_COLOR, DISAGREE_COLOR
import StateRendering: plot_exciton_spectrum, plot_exciton, plot_configurations,
                       plot_levels, plot_irrep_weights, plot_operation_map,
                       plot_symmetry_quality, plot_charges, plot_parent_mixing,
                       plot_field_irreps, plot_sweep

_extent(b::ExcitonBundle) = (b.x, b.y, b.z)

# index into the per-state density arrays, which may cover only a subset
function _density_index(b::ExcitonBundle, i::Integer)
    sel = get(b.metadata, :density_states, collect(1:length(b)))
    k = findfirst(==(i), sel)
    k === nothing && throw(ArgumentError(
        "exciton $i has no stored density; the bundle was built with " *
        "states = $(sel). Rebuild with `states = :all` or include $i."))
    return k
end

function _geometry_mesh!(ax, b::ExcitonBundle; level = 0.5, σ = 1.0,
                         color = :grey, alpha = 0.10)
    b.geometry === nothing && return nothing
    χ = σ > 0 ? smooth(b.geometry; σ = σ) : b.geometry
    m = isosurface_mesh(χ, _extent(b)..., level)
    m === nothing && return nothing
    return mesh!(ax, m; color = (color, alpha), transparency = true)
end

"""
    plot_exciton(xb, i; fraction = 0.5, geometry = true) -> Figure

The electron and hole of exciton `i` as isosurfaces of their **reduced one-body
densities**, in one axis.

Both surfaces enclose `fraction` of their particle's charge, so they are
directly comparable: where the hole surface sits inside the electron one, the
hole is the more strongly confined particle, and the offset between the two
centroids is the exciton's static dipole.
"""
function plot_exciton(xb::ExcitonBundle, i::Integer; fraction::Real = 0.5,
                      geometry::Bool = true, size = (620, 560), title = nothing)
    k = _density_index(xb, i)
    fig = Figure(size = size)
    lab = isempty(xb.labels) || isempty(xb.labels[i]) ? "" : "  " * xb.labels[i]
    ttl = title === nothing ?
        @sprintf("X%d   %.4f eV%s%s", i, xb.energies[i], lab,
                 isempty(xb.total) || xb.total[i] == 0 ? "" :
                 (xb.bright[i] ? "   bright" : "   dark")) : title
    ax = state_axis3(fig[1, 1]; title = ttl)

    geometry && _geometry_mesh!(ax, xb)
    for (d, kind) in ((xb.e_density[k], :electron), (xb.h_density[k], :hole))
        m = isosurface_mesh(d, _extent(xb)..., isolevel(d, fraction))
        m === nothing && continue
        mesh!(ax, m; color = (kind_color(kind), 0.5), transparency = true)
    end
    Legend(fig[2, 1],
           [PolyElement(color = (kind_color(:electron), 0.5)),
            PolyElement(color = (kind_color(:hole), 0.5))],
           ["electron", "hole"]; orientation = :horizontal, framevisible = false,
           tellheight = true)
    return fig
end

"""
    plot_exciton_spectrum(xb; ...) -> Figure

Level diagram of the exciton spectrum: one tick per degenerate level, labelled
with its irrep and degeneracy, with the marker weight set by oscillator
strength.

Dark levels are drawn as thin outlines rather than omitted — a symmetry-forbidden
level is a prediction, and leaving it off the figure hides the very thing the
irrep labelling is for.
"""
function plot_exciton_spectrum(xb::ExcitonBundle; size = (560, 520),
                               title = "exciton spectrum", relative::Bool = true)
    fig = Figure(size = size)
    E0 = relative ? minimum(xb.energies) : 0.0
    unit = relative ? "E − E₁ (meV)" : "E (eV)"
    scale = relative ? 1000.0 : 1.0
    ax = Axis(fig[1, 1]; title = title, ylabel = unit)

    tmax = isempty(xb.total) ? 0.0 : maximum(xb.total)
    ys, txts, cols = Float64[], String[], Any[]
    for ids in xb.levels
        i = first(ids)
        y = (xb.energies[i] - E0) * scale
        f = tmax > 0 ? xb.total[i] / tmax : 0.0
        on = tmax > 0 && xb.bright[i]
        lines!(ax, [-0.4, 0.4], fill(y, 2);
               color = on ? kind_color(:electron) : (:grey, 0.8),
               linewidth = on ? 1.5 + 4f : 1.2,
               linestyle = on ? :solid : :dash)
        lab = isempty(xb.labels) ? "" : xb.labels[i]
        # the irrep's colour, the same as in every other figure of this group
        scatter!(ax, [-0.5], [y]; markersize = 9, strokewidth = 0,
                 color = irrep_color(lab, xb.irreps))
        deg = length(ids)
        txt = @sprintf("%s%s%s", lab, deg > 1 ? " ×$deg" : "",
                       tmax > 0 ? (on ? @sprintf("  f=%.2f", f) : "  dark") : "")
        isempty(strip(txt)) && continue
        push!(ys, y); push!(txts, txt); push!(cols, on ? kind_color(:electron) : :grey)
    end
    # labels of close levels (a fine-structure manifold) are moved apart, not stacked
    span = isempty(ys) ? 1.0 : max(maximum(ys) - minimum(ys), 1e-9)
    for (y, t, c) in zip(_repel(ys, 0.032 * span), txts, cols)
        text!(ax, 0.46, y; text = t, align = (:left, :center), fontsize = 10, color = c)
    end
    xlims!(ax, -0.6, 1.6)
    hidexdecorations!(ax)
    return fig
end

"""
    plot_configurations(xb, i; ...) -> Figure

`|C_eh|²` for exciton `i` as a heatmap over the single-particle pairs, with the
axes labelled by the electron and hole irreps.

This is the figure that says *what the exciton is made of*. A level that is one
irrep should be supported only on pairs whose product contains it, so a
symmetry-allowed but unoccupied cell and an occupied but forbidden one are both
visible at a glance — the second means the labelling or the adaptation is wrong.
"""
function plot_configurations(xb::ExcitonBundle, i::Integer; size = (560, 440),
                             title = nothing, colormap = :viridis)
    C = xb.C[i]
    ne, nh = Base.size(C)
    w = abs2.(C)

    etick = [isempty(xb.e_labels) || isempty(xb.e_labels[e]) ?
             @sprintf("e%d", e) : @sprintf("e%d\n%s", e, xb.e_labels[e])
             for e in 1:ne]
    htick = [isempty(xb.h_labels) || isempty(xb.h_labels[h]) ?
             @sprintf("h%d", h) : @sprintf("h%d\n%s", h, xb.h_labels[h])
             for h in 1:nh]

    fig = Figure(size = size)
    lab = isempty(xb.labels) || isempty(xb.labels[i]) ? "" : "   " * xb.labels[i]
    ax = Axis(fig[1, 1];
              title = title === nothing ?
                  @sprintf("X%d   %.4f eV%s", i, xb.energies[i], lab) : title,
              xlabel = "hole", ylabel = "electron",
              xticks = (1:nh, htick), yticks = (1:ne, etick),
              aspect = DataAspect())
    hm = heatmap!(ax, 1:nh, 1:ne, permutedims(w); colormap = colormap,
                  colorrange = (0, maximum(w)))
    for e in 1:ne, h in 1:nh
        w[e, h] < 1e-3 && continue
        text!(ax, h, e; text = @sprintf("%.2f", w[e, h]), align = (:center, :center),
              fontsize = 10, color = w[e, h] > 0.5 * maximum(w) ? :white : :black)
    end
    Colorbar(fig[1, 2], hm; label = "|C|²")
    return fig
end

include("StateRenderingCICalcExt/labelling.jl")

end # module StateRenderingCICalcExt
