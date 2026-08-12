# ---------------------------------------------------------------------------
# Composed figures.
#
# All of these are static and backend-agnostic: they build a Figure out of the
# primitives and return it, so `save("fig.pdf", plot_slices(b, 1))` works under
# CairoMakie and the same call under GLMakie opens a window. Interaction lives in
# exactly one place, `explore`.
#
# Each figure answers one question:
#   plot_state       what does this state look like
#   plot_states      how do these states differ
#   plot_slices      what is the density, quantitatively, through the middle
#   plot_bands       which band is carrying the state, in space
#   plot_composition how does band character evolve up the ladder
#   plot_spectrum    where are the levels and what are they made of
#   plot_field       what does the strain / potential / composition look like
# ---------------------------------------------------------------------------

_state_title(b::StateBundle, i::Integer) = begin
    lbl, w = dominant_component(b, i)
    @sprintf("%s%d   E = %.4f eV   %s %.0f%%",
             b.kind[i] === :hole ? "h" : "e", i, b.energies[i], lbl, 100w)
end

"""
    plot_state(b, i; bands = :all, fraction = 0.5, geometry = true,
               fields = ScalarField[], bars = true, size = (760, 560)) -> Figure

One state: the isosurface enclosing `fraction` of its probability, the dot
outline, any overlay fields, and its orbital-component weights.

    save("e1.png", plot_state(bundle, 1))
"""
function plot_state(b::StateBundle, i::Integer;
                    bands = :all, fraction::Real = 0.5,
                    geometry::Bool = true, fields = ScalarField[],
                    field_mode::Symbol = :iso, field_level::Real = 0.5,
                    vectors = nothing, bars::Bool = true,
                    color = :auto, size = (760, 560), title = nothing)
    fig = Figure(size = size)
    ax = state_axis3(fig[1, 1]; title = title === nothing ? _state_title(b, i) : title)
    add_density!(ax, b, i; bands = bands, fraction = fraction, color = color)
    geometry && add_geometry!(ax, b)
    for f in fields
        add_scalar!(ax, b, f; mode = field_mode, level = field_level)
    end
    vectors === nothing || add_vectors!(ax, b, vectors)

    if bars
        bax = Axis(fig[1, 2]; title = "components", width = 150)
        add_component_bars!(bax, b, i)
        hidespines!(bax, :t, :r)
        colsize!(fig.layout, 2, Auto(false))
    end
    return fig
end

"""
    plot_states(b; states = 1:length(b), ncols = 3, fraction = 0.5, ...) -> Figure

A panel per state, same isosurface convention throughout, so the shapes are
directly comparable. This is the figure the old package could not produce at
all — it could only show one state at a time, behind a dropdown.
"""
function plot_states(b::StateBundle; states = 1:length(b), ncols::Integer = 3,
                     bands = :all, fraction::Real = 0.5, geometry::Bool = true,
                     color = :auto, panel = (300, 280), link::Bool = true)
    idx = collect(states)
    nc = min(ncols, length(idx))
    nr = cld(length(idx), nc)
    fig = Figure(size = (panel[1] * nc, panel[2] * nr))
    axs = Axis3[]
    for (n, i) in pairs(idx)
        r, c = fldmod1(n, nc)
        ax = state_axis3(fig[r, c]; title = _state_title(b, i), titlesize = 12)
        add_density!(ax, b, i; bands = bands, fraction = fraction, color = color)
        geometry && add_geometry!(ax, b)
        push!(axs, ax)
    end
    link && length(axs) > 1 && for ax in axs[2:end]
        ax.azimuth = axs[1].azimuth[]; ax.elevation = axs[1].elevation[]
    end
    return fig
end

"""
    plot_slices(b, i; planes = (:z, :y, :x), at = :centroid, bands = :all) -> Figure

Orthogonal cuts through state `i`, with the dot boundary contoured on each. The
slice positions default to the state's own weight-centre, and a shared colour
bar makes the three panels quantitatively comparable.
"""
function plot_slices(b::StateBundle, i::Integer;
                     planes = (:z, :y, :x), at = :centroid, bands = :all,
                     colormap = :inferno, geometry::Bool = true,
                     size = (900, 330), title = nothing)
    fig = Figure(size = size)
    Label(fig[0, 1:length(planes)], title === nothing ? _state_title(b, i) : title;
          fontsize = 15, padding = (0, 0, 5, 0))
    d = state_density(b, i; bands = bands)
    crange = (0.0, maximum(d))
    hm = nothing
    for (n, p) in pairs(planes)
        ax = Axis(fig[1, n]; aspect = DataAspect(),
                  title = "$(_AXIS_NAMES[_axis_of(p)]) cut")
        hm = add_slice!(ax, b, i; plane = p, at = at, bands = bands,
                        colormap = colormap, geometry = geometry, colorrange = crange)
    end
    hm === nothing || Colorbar(fig[1, length(planes) + 1], hm; label = "|ψ|²")
    return fig
end

"""
    plot_bands(b, i; plane = :z, at = :centroid, ncols = 4) -> Figure

The state split into its spinor bands: one cut per band, each labelled with the
band's name **from the bundle** and its share of the probability.

This is the figure where getting the basis right matters most. The same six
panels mean |HH↑,LH↑,LH↓,HH↓,SO↑,SO↓⟩ for zinc-blende and |X↑,Y↑,Z↑,X↓,Y↓,Z↓⟩
for wurtzite, and nothing but the labels distinguishes them.

Each panel is scaled to its own maximum by default, because the interesting
thing about a 2%-weight band is its *shape* — the nodal structure that says
which p-like envelope it carries — and a shared scale renders every minority
band as a black square. The magnitude is not lost: it is printed in the panel
title. Pass `shared_scale = true` to compare panels directly instead.
"""
function plot_bands(b::StateBundle, i::Integer;
                    plane::Symbol = :z, at = :centroid, ncols::Integer = 4,
                    colormap = :inferno, geometry::Bool = true,
                    panel = (230, 230), shared_scale::Bool = false)
    nb = nbands(b)
    nc = min(ncols, nb)
    nr = cld(nb, nc)
    fig = Figure(size = (panel[1] * nc + 90, panel[2] * nr + 60))
    Label(fig[0, 1:(nc + 1)], _state_title(b, i); fontsize = 15)

    w = band_weights(b, i)
    # a common colour scale exposes how little weight the minority bands carry;
    # per-panel scaling makes a 1% band look as strong as the dominant one
    crange = shared_scale ? (0.0, maximum(state_density(b, i))) : nothing
    hm = nothing
    for k in 1:nb
        r, c = fldmod1(k, nc)
        ax = Axis(fig[r, c]; aspect = DataAspect(), titlesize = 12,
                  title = @sprintf("%s   %.3f", b.band_labels[k], w[k]))
        kw = crange === nothing ? (;) : (; colorrange = crange)
        hm = add_slice!(ax, b, i; plane = plane, at = at, bands = [k],
                        colormap = colormap, geometry = geometry, kw...)
        (r == nr) || hidexdecorations!(ax; grid = false)
        (c == 1)  || hideydecorations!(ax; grid = false)
    end
    hm === nothing || Colorbar(fig[1:nr, nc + 1], hm; label = "|ψ_b|²")
    return fig
end

"""
    plot_composition(b; states = 1:length(b)) -> Figure

Orbital-component weights of every state as stacked bars — how band character
evolves up the ladder. A conduction ladder should stay flat and S-dominated; a
valence ladder that flips from HH to LH partway up is telling you something
about the confinement.
"""
function plot_composition(b::StateBundle; states = 1:length(b),
                          size = (740, 380), title = "band character")
    idx = collect(states)
    cols = component_palette(b.component_labels)
    nc = length(b.component_labels)

    fig = Figure(size = size)
    ax = Axis(fig[1, 1]; title = title, xlabel = "state", ylabel = "weight",
              xticks = (1:length(idx),
                        [@sprintf("%s%d", b.kind[i] === :hole ? "h" : "e", i) for i in idx]))
    xs = Int[]; ys = Float64[]; grp = Int[]
    for (n, i) in pairs(idx)
        w = component_weights(b, i)
        for c in 1:nc
            push!(xs, n); push!(ys, w[c]); push!(grp, c)
        end
    end
    barplot!(ax, xs, ys; stack = grp, color = cols[grp])
    ylims!(ax, 0, 1)
    Legend(fig[1, 2], [PolyElement(color = c) for c in cols], b.component_labels,
           "component"; framevisible = false)
    return fig
end

"""
    plot_spectrum(b; ...) -> Figure

Level diagram, each level coloured and annotated by its dominant component.
Electrons and holes get their own panel when both are present — they are ~1 eV
apart, and a shared axis would compress each ladder to a smudge.
"""
function plot_spectrum(b::StateBundle; size = (520, 480), title = "spectrum")
    kinds = unique(b.kind)
    fig = Figure(size = size)
    Label(fig[0, 1:max(length(kinds), 1)], title; fontsize = 15)
    for (n, k) in pairs(kinds)
        sel = findall(==(k), b.kind)
        ax = Axis(fig[1, n]; title = k === :hole ? "holes" : "electrons",
                  ylabel = "E (eV)")
        sub = _subset(b, sel)
        add_levels!(ax, sub)
        xlims!(ax, -0.6, 1.4)
    end
    return fig
end

# a view of the bundle restricted to some states, for per-kind panels.
# `typeof(b)` keeps the element-type parameters, which cannot be inferred from
# the field values when `states` has been dropped.
function _subset(b::StateBundle, sel)
    return typeof(b)(b.density[sel],
                     b.states === nothing ? nothing : b.states[sel],
                     b.energies[sel], b.kind[sel],
                     b.band_labels, b.component_labels, b.component_groups,
                     b.x, b.y, b.z, b.geometry, b.metadata)
end

"""
    plot_field(b, f::ScalarField; planes = (:z, :y, :x), at = :middle) -> Figure

A scalar overlay on its own — strain component, piezoelectric potential,
composition — as orthogonal cuts. Signed fields get a diverging map centred on
zero with symmetric limits, so a lobe pattern reads correctly and the zero
contour is where it should be.
"""
function plot_field(b::StateBundle, f::ScalarField;
                    planes = (:z, :y, :x), at = :middle,
                    geometry::Bool = true, size = (900, 330))
    _check_grid(f, b, "field \"$(f.name)\"")
    coords = (b.x, b.y, b.z)
    peak = maximum(abs, f.data)
    crange = f.signed ? (-peak, peak) : extrema(f.data)
    cmap = f.signed ? :RdBu : :viridis

    fig = Figure(size = size)
    unit = isempty(f.unit) ? "" : " ($(f.unit))"
    Label(fig[0, 1:length(planes)], f.name * unit; fontsize = 15)
    hm = nothing
    for (n, p) in pairs(planes)
        ax_i = _axis_of(p)
        idx = at === :middle ? (length(coords[ax_i]) + 1) ÷ 2 :
              nearest_index(coords[ax_i], at)
        a1, a2 = _inplane_axes(p)
        ax = Axis(fig[1, n]; aspect = DataAspect(),
                  title = "$(_AXIS_NAMES[ax_i]) cut",
                  xlabel = "$(_AXIS_NAMES[a1]) (nm)", ylabel = "$(_AXIS_NAMES[a2]) (nm)")
        hm = heatmap!(ax, coords[a1], coords[a2],
                      Array(selectdim(f.data, ax_i, idx));
                      colormap = cmap, colorrange = crange)
        if geometry && b.geometry !== nothing
            contour!(ax, coords[a1], coords[a2],
                     Array(selectdim(b.geometry, ax_i, idx));
                     levels = [0.5], color = :black, linewidth = 1.5)
        end
    end
    hm === nothing || Colorbar(fig[1, length(planes) + 1], hm)
    return fig
end

plot_field(b::StateBundle, s::StrainField; kwargs...) =
    plot_field(b, ScalarField(hydrostatic(s), "hydrostatic"; signed = true); kwargs...)
