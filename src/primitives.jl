# ---------------------------------------------------------------------------
# Drawing primitives: each one adds ONE thing to an axis you already own.
#
# Nothing here creates a Figure, chooses a layout, or assumes a backend. That is
# what makes the same code serve both the interactive explorer and a static
# figure destined for a paper — the old package could only do the former, because
# every plot function built its own GLMakie window.
#
# Everything is drawn in nanometres, taken from the bundle's own coordinates, so
# a density, a dot outline and a strain field overlay in the same space and a
# length can be read off the axis.
# ---------------------------------------------------------------------------

"""
    state_axis3(position; title = "", decorated = false) -> Axis3

An `Axis3` set up for nanostructure volumes: data aspect (so the box is not
distorted), nm-labelled axes, and decorations off by default since a density
isosurface rarely needs tick marks.
"""
function state_axis3(position; title = "", decorated::Bool = false, kwargs...)
    ax = Axis3(position; title = title, aspect = :data,
               xlabel = "x (nm)", ylabel = "y (nm)", zlabel = "z (nm)",
               protrusions = 0, kwargs...)
    decorated || hidedecorations!(ax)
    return ax
end

_extent(b::StateBundle) = (b.x, b.y, b.z)

"""
    add_density!(ax, b, i; bands = :all, fraction = 0.5, color = :auto,
                 alpha = 0.55, shading = true) -> plot

Isosurface of state `i` enclosing `fraction` of its probability (see
[`isolevel`](@ref)). `bands` selects a subset of the spinor bands — a vector of
indices, a `Bool` mask, or `:all`.

`color = :auto` uses the colour of the state's dominant orbital component, so an
S-like state and an HH-like state are distinguishable without a legend; pass
`:kind` for electron-blue / hole-red instead, or any colour.

Returns `nothing` if the state is too flat to have a surface at that level.
"""
function add_density!(ax, b::StateBundle, i::Integer;
                      bands = :all, fraction::Real = 0.5,
                      color = :auto, alpha::Real = 0.55,
                      transparency::Bool = true, kwargs...)
    d = state_density(b, i; bands = bands)
    m = isosurface_mesh(d, _extent(b)..., isolevel(d, fraction))
    m === nothing && return nothing
    c = _density_color(b, i, color)
    return mesh!(ax, m; color = (c, alpha), transparency = transparency, kwargs...)
end

function _density_color(b::StateBundle, i::Integer, color)
    color === :kind && return kind_color(b.kind[i])
    color === :auto || return color
    lbl, _ = dominant_component(b, i)
    j = findfirst(==(lbl), b.component_labels)
    return component_palette(b.component_labels)[j]
end

"""
    add_geometry!(ax, b; level = 0.5, color = :grey, alpha = 0.12, σ = 1.0)

Outline of the dot: the `χ(r) = level` surface of the bundle's composition
profile. `σ` is a cosmetic blur in grid units — a sharp interface is a 0/1 step
whose isosurface is a staircase of grid facets, and one spacing of blur restores
the interface the model actually describes.

Does nothing if the bundle carries no geometry.
"""
function add_geometry!(ax, b::StateBundle; level::Real = 0.5, color = :grey,
                       alpha::Real = 0.12, σ::Real = 1.0, kwargs...)
    b.geometry === nothing && return nothing
    χ = σ > 0 ? smooth(b.geometry; σ = σ) : b.geometry
    m = isosurface_mesh(χ, _extent(b)..., level)
    m === nothing && return nothing
    return mesh!(ax, m; color = (color, alpha), transparency = true, kwargs...)
end

"""
    add_scalar!(ax, b, f::ScalarField; mode = :iso, level = 0.5, alpha = 0.3)

Overlay a scalar field — a strain component, a piezoelectric potential, a
composition.

`mode = :iso` draws isosurfaces, and for a signed field draws **both** `±level`
(blue and red), which is the right picture for a quadrupolar piezoelectric
potential where the two lobes are the whole story. `level` is a fraction of the
field's peak magnitude.

`mode = :volume` ray-marches the whole field instead, which is more informative
but needs a GPU backend (GLMakie/WGLMakie); CairoMakie cannot draw it.
"""
function add_scalar!(ax, b::StateBundle, f::ScalarField;
                     mode::Symbol = :iso, level::Real = 0.5, alpha::Real = 0.3,
                     absorption::Real = 2.0, colormap = nothing, kwargs...)
    _check_grid(f, b, "field \"$(f.name)\"")
    peak = maximum(abs, f.data)
    peak > 0 || return nothing
    n = f.data ./ peak

    if mode === :volume
        cmap = colormap === nothing ?
            (f.signed ? diverging_map(alpha) : to_colormap(:viridis)) : colormap
        crange = f.signed ? (-1.0, 1.0) : (0.0, 1.0)
        # `volume!` takes interval endpoints, not coordinate vectors
        return volume!(ax, extrema(b.x), extrema(b.y), extrema(b.z), n;
                       algorithm = :absorption, absorption = Float32(absorption),
                       colormap = cmap, colorrange = crange,
                       transparency = true, kwargs...)
    elseif mode === :iso
        out = Any[]
        levels = f.signed ? (level, -level) : (level,)
        colors = f.signed ? (:red, :blue) : (:seagreen,)
        for (l, c) in zip(levels, colors)
            m = isosurface_mesh(n, _extent(b)..., l)
            m === nothing && continue
            push!(out, mesh!(ax, m; color = (c, alpha), transparency = true, kwargs...))
        end
        return out
    end
    throw(ArgumentError("mode must be :iso or :volume, got :$mode"))
end

"""
    add_vectors!(ax, b, v::VectorField; plane = :z, at = nothing, step = 4,
                 lengthscale = :auto, surface = true)

Arrows sampled from a vector field on one plane, coloured by magnitude, with an
optional translucent magnitude surface underneath.

A full 3D arrow cloud is unreadable, so a plane is the honest way to show this;
`at` is the physical coordinate of the plane (default: the middle of the box) and
`step` thins the arrows.
"""
function add_vectors!(ax, b::StateBundle, v::VectorField;
                      plane::Symbol = :z, at = nothing, step::Integer = 4,
                      lengthscale = :auto, colormap = :viridis,
                      surface::Bool = true, alpha::Real = 0.5, kwargs...)
    _check_grid(v, b, "vector field \"$(v.name)\"")
    ax_i = _axis_of(plane)
    coords = (b.x, b.y, b.z)
    idx = at === nothing ? (length(coords[ax_i]) + 1) ÷ 2 : nearest_index(coords[ax_i], at)

    mag = magnitude(v)
    peak = maximum(mag)
    peak > 0 || return nothing
    crange = (0.0, peak)

    ranges = [1:step:length(coords[d]) for d in 1:3]
    ranges[ax_i] = idx:idx

    pts = Point3f[]; dirs = Vec3f[]; cols = Float64[]
    for k in ranges[3], j in ranges[2], i in ranges[1]
        push!(pts, Point3f(b.x[i], b.y[j], b.z[k]))
        push!(dirs, Vec3f(v.x[i, j, k], v.y[i, j, k], v.z[i, j, k]) / peak)
        push!(cols, mag[i, j, k])
    end

    ls = lengthscale === :auto ?
        Float32(step * minimum(c -> length(c) > 1 ? c[2] - c[1] : 1.0, coords) * 2) :
        Float32(lengthscale)

    out = Any[]
    if surface
        a1, a2 = _inplane_axes(plane)
        s = selectdim(mag, ax_i, idx)
        c1, c2 = coords[a1], coords[a2]
        grids = Array{Float32}[fill(Float32(coords[ax_i][idx]), length(c1), length(c2)),
                               [Float32(u) for u in c1, _ in c2],
                               [Float32(w) for _ in c1, w in c2]]
        # order the three coordinate grids as (x, y, z)
        xyz = Vector{Any}(undef, 3)
        xyz[ax_i] = grids[1]; xyz[a1] = grids[2]; xyz[a2] = grids[3]
        push!(out, surface!(ax, xyz...; color = Array{Float32}(s), colormap = colormap,
                            colorrange = crange, transparency = true, alpha = alpha))
    end
    push!(out, arrows3d!(ax, pts, dirs; lengthscale = ls, color = cols,
                         colormap = colormap, colorrange = crange, kwargs...))
    return out
end

# ── 2D: slices, bars, levels ───────────────────────────────────────────────

"""
    add_slice!(ax, b, i; plane = :z, at = :centroid, bands = :all,
               colormap = :inferno, geometry = true)

A 2D cut through the density of state `i`. `at = :centroid` places the plane
through the weight-centre of the state, which for an off-centre state is where
the physics is; the middle of the box may well be empty.

Slices are the workhorse for publication: they carry quantitative information a
3D isosurface cannot, and every backend can draw them.
"""
function add_slice!(ax, b::StateBundle, i::Integer;
                    plane::Symbol = :z, at = :centroid, bands = :all,
                    colormap = :inferno, geometry::Bool = true,
                    geometry_color = :white, kwargs...)
    d = state_density(b, i; bands = bands)
    ax_i = _axis_of(plane)
    coords = (b.x, b.y, b.z)
    pos = at === :centroid ? centroid(d, b.x, b.y, b.z)[ax_i] :
          at === :middle   ? coords[ax_i][(end + 1) ÷ 2] : at
    idx = nearest_index(coords[ax_i], pos)

    a1, a2 = _inplane_axes(plane)
    s = Array(selectdim(d, ax_i, idx))
    hm = heatmap!(ax, coords[a1], coords[a2], s; colormap = colormap, kwargs...)
    if geometry && b.geometry !== nothing
        contour!(ax, coords[a1], coords[a2], Array(selectdim(b.geometry, ax_i, idx));
                 levels = [0.5], color = geometry_color, linewidth = 1.5)
    end
    ax.xlabel = "$(_AXIS_NAMES[a1]) (nm)"
    ax.ylabel = "$(_AXIS_NAMES[a2]) (nm)"
    return hm
end

"""
    add_component_bars!(ax, b, i; horizontal = true, annotate = true)

Orbital-component weights of state `i` as a bar chart, labelled and coloured from
the bundle.

Bars rather than the more decorative pie: a state that is 85% S with three 5%
admixtures has three slices too thin to see, and the interesting number is
exactly how small they are. Bars also accept any number of components, where a
fixed four-colour pie silently breaks on a 3- or 6-component basis.
"""
function add_component_bars!(ax, b::StateBundle, i::Integer;
                             horizontal::Bool = true, annotate::Bool = true,
                             kwargs...)
    w = component_weights(b, i)
    n = length(w)
    cols = component_palette(b.component_labels)
    bp = barplot!(ax, 1:n, w; color = cols,
                  direction = horizontal ? :x : :y, kwargs...)
    if horizontal
        ax.yticks = (1:n, b.component_labels)
        ax.xlabel = "weight"
        xlims!(ax, 0, 1.15)
    else
        ax.xticks = (1:n, b.component_labels)
        ax.ylabel = "weight"
        ylims!(ax, 0, 1.15)
    end
    if annotate
        txt = [@sprintf("%.3f", v) for v in w]
        pos = horizontal ? Point2f.(w .+ 0.02, 1:n) : Point2f.(1:n, w .+ 0.02)
        text!(ax, pos; text = txt, fontsize = 10,
              align = horizontal ? (:left, :center) : (:center, :bottom))
    end
    return bp
end

"""
    add_levels!(ax, b; x = 0, width = 0.8, label = true)

Energy levels of the bundle as horizontal ticks, coloured by dominant component.
Degenerate levels are drawn on top of each other, which is the point: a Kramers
pair looks like one line, and a split one does not.
"""
function add_levels!(ax, b::StateBundle; x::Real = 0, width::Real = 0.8,
                     label::Bool = true, fontsize = 10, kwargs...)
    cols = component_palette(b.component_labels)
    for i in 1:length(b)
        lbl, w = dominant_component(b, i)
        c = cols[findfirst(==(lbl), b.component_labels)]
        lines!(ax, [x - width/2, x + width/2], fill(b.energies[i], 2);
               color = c, linewidth = 2.5, kwargs...)
        label && text!(ax, x + width/2 + 0.05, b.energies[i];
                       text = @sprintf("%s %.0f%%", lbl, 100w),
                       align = (:left, :center), fontsize = fontsize, color = c)
    end
    ax.ylabel = "E (eV)"
    hidexdecorations!(ax)
    return ax
end
