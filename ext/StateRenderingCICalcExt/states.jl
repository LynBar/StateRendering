# ---------------------------------------------------------------------------
# One state from a results archive: its reduced e/h densities, its configurations,
# and the single-particle orbitals of the render tier. These replace the former
# ExcitonBundle figures and work for any sector the archive stores.
#
# Each reads what the archive holds and says which spec key would fill what it
# lacks: densities (`densities:` of the sector), vectors (`keep:`), orbitals
# (`orbitals:`).
# ---------------------------------------------------------------------------

# a stored value: an array as is, else read it (`HeavyRef` from a file, or a heavy-marked
# value of an archive built in memory)
_val(x) = x isa AbstractArray || x === nothing ? x : x[]

function _sector_state(res::CICalcResults, name, i::Integer)
    s = _pick(res.sectors, String(name), "sector", "Add it to the run spec's `compute: sectors:`.")
    1 <= i <= length(s) || throw(ArgumentError(
        "state $i is out of range: sector $(s.name) has $(length(s)) states"))
    return s
end

_state_title(s, i) = @sprintf("%s%d   %.4f eV%s", s.name, i, s.energies[i],
                              isempty(s.state_labels) || isempty(s.state_labels[i]) ?
                              "" : "   " * s.state_labels[i])

# the irrep label of orbital `o` (an index into the archive's orbitals)
function _orbital_label(res::CICalcResults, o::Integer)
    lb = res.levels
    lb === nothing && return ""
    b = findfirst(r -> o in r, lb.blocks)
    return b === nothing || isempty(lb.labels) ? "" : lb.labels[b]
end

# the geometry χ(r) of the render tier, if it is on the same coarse grid
function _render_geometry(res::CICalcResults, coords)
    ro = get(res.extra, "render/orbitals", nothing)
    (ro === nothing || ro.geometry === nothing) && return nothing
    all(length(ro.coords[d]) == length(coords[d]) for d in 1:3) || return nothing
    return Float64.(_val(ro.geometry))
end

function _geometry!(ax, χ, coords; level = 0.5, σ = 1.0)
    χ === nothing && return nothing
    m = isosurface_mesh(σ > 0 ? smooth(χ; σ) : χ, coords..., level)
    m === nothing || mesh!(ax, m; color = (:grey, 0.10), transparency = true)
    return nothing
end

"""
    plot_densities(res::CICalcResults, name, i; fraction = 0.5, geometry = true) -> Figure

The electron and hole of state `i` of a stored sector, as isosurfaces of their
**reduced one-body densities**, in one axis. Both surfaces enclose `fraction` of their
particle's charge, so they are directly comparable: the offset between them is the
state's static dipole. The densities are a heavy product: list the state in the
sector's `densities:` in the run spec. With the render tier stored on the same grid,
the dot's geometry is drawn faintly behind them.
"""
function plot_densities(res::CICalcResults, name::AbstractString, i::Integer;
                        fraction::Real = 0.5, geometry::Bool = true, size = (620, 560),
                        title = nothing)
    s = _sector_state(res, name, i)
    haskey(s.densities, i) || throw(ArgumentError(
        "no densities are stored for $(s.name) state $i" *
        (isempty(s.densities) ? "" : " (stored: $(join(sort!(collect(keys(s.densities))), ", ")))") *
        ". Add it to `densities:` of sector $(s.name) in the run spec."))
    d = s.densities[i]
    xyz = d.coords
    fig = Figure(size = size)
    ax = state_axis3(fig[1, 1]; title = title === nothing ? _state_title(s, i) : title)
    geometry && _geometry!(ax, _render_geometry(res, xyz), xyz)
    shown = Any[]
    for (ρ, kind) in ((_val(d.e), :electron), (_val(d.h), :hole))
        maximum(ρ) > 0 || continue
        m = isosurface_mesh(Float64.(ρ), xyz..., isolevel(ρ, fraction))
        m === nothing && continue
        mesh!(ax, m; color = (kind_color(kind), 0.5), transparency = true)
        push!(shown, kind)
    end
    isempty(shown) || Legend(fig[2, 1], [PolyElement(color = (kind_color(k), 0.5)) for k in shown],
                             string.(shown); orientation = :horizontal, framevisible = false,
                             tellheight = true)
    return fig
end

"""
    plot_configurations(res::CICalcResults, name, i; top = 16) -> Figure

What state `i` of a stored sector is made of: `|C|²` over its determinants. For an
exciton (1,1) it is a heatmap over (electron, hole) orbital pairs, the axes labelled
by the orbitals' irreps; for any other sector, a bar chart of the `top` largest
determinants. A level that is one irrep should be supported only on configurations
whose product contains it, so an occupied but forbidden cell means the labelling or
the adaptation is wrong. Needs the state's CI vector (`keep:` of the sector).
"""
function plot_configurations(res::CICalcResults, name::AbstractString, i::Integer;
                             top::Integer = 16, size = (600, 460), title = nothing,
                             colormap = :viridis)
    s = _sector_state(res, name, i)
    i <= s.nkept || throw(ArgumentError(
        "the CI vectors of $(s.name) are stored for its lowest $(s.nkept) states; " *
        "raise `keep:` of sector $(s.name) in the run spec to see state $i"))
    v = _val(s.vectors)[:, i]
    ne, nh = length(s.estates), length(s.hstates)
    occ(b) = ([e for e in 1:ne if (b >> (e - 1)) & 1 == 1],
              [h for h in 1:nh if (b >> (ne + h - 1)) & 1 == 0])
    olab(st, k, pre) = (l = _orbital_label(res, st[k].orb); isempty(l) ? "$pre$k" : "$pre$k\n$l")
    w = abs2.(v)
    fig = Figure(size = size)
    ttl = title === nothing ? _state_title(s, i) : title
    if s.sector == (1, 1)
        W = zeros(ne, nh)
        for (k, b) in enumerate(s.basis)
            es, hs = occ(b)
            W[only(es), only(hs)] += w[k]
        end
        ax = Axis(fig[1, 1]; title = ttl, xlabel = "hole", ylabel = "electron",
                  xticks = (1:nh, [olab(s.hstates, h, "h") for h in 1:nh]),
                  yticks = (1:ne, [olab(s.estates, e, "e") for e in 1:ne]),
                  aspect = DataAspect())
        hm = heatmap!(ax, 1:nh, 1:ne, permutedims(W); colormap, colorrange = (0, maximum(W)))
        for e in 1:ne, h in 1:nh
            W[e, h] < 1e-3 && continue
            text!(ax, h, e; text = @sprintf("%.2f", W[e, h]), align = (:center, :center),
                  fontsize = 10, color = W[e, h] > 0.5 * maximum(W) ? :white : :black)
        end
        Colorbar(fig[1, 2], hm; label = "|C|²")
    else
        ord = sortperm(w; rev = true)[1:min(top, length(w))]
        labs = map(ord) do k
            es, hs = occ(s.basis[k])
            join(("e$e" for e in es), " ") * " | " * join(("h$h" for h in hs), " ")
        end
        ax = Axis(fig[1, 1]; title = ttl, ylabel = "|C|²",
                  xticks = (eachindex(ord), labs), xticklabelrotation = π / 4)
        barplot!(ax, eachindex(ord), w[ord]; color = kind_color(:electron))
        Label(fig[2, 1], @sprintf("the %d largest of %d determinants (Σ = %.3f)", length(ord),
                                  length(w), sum(w[ord])); fontsize = 10, color = _MUTED,
              tellwidth = false)
    end
    return fig
end

"""
    plot_orbital(res::CICalcResults, k; fraction = 0.5, bands = true) -> Figure

Single-particle orbital `k` from the render tier: the isosurface enclosing `fraction`
of |ψ|², one surface per band group (S, HH, LH, SO or S, X, Y, Z) holding at least 2 %
of the weight, coloured by group, or the total alone with `bands = false`. The render
tier is a heavy product: add `orbitals:` to the run spec.
"""
function plot_orbital(res::CICalcResults, k::Integer; fraction::Real = 0.5,
                      bands::Bool = true, geometry::Bool = true, size = (620, 560),
                      title = nothing)
    ro = get(res.extra, "render/orbitals", nothing)
    ro === nothing && _missing_product("render tier (orbitals on a coarse grid)",
                                       "Add `orbitals: {stride: 2}` to the run spec's `compute`.")
    n = length(ro.energies)
    1 <= k <= n || throw(ArgumentError("orbital $k is out of range 1:$n"))
    lab = _orbital_label(res, k)
    fig = Figure(size = size)
    ttl = title === nothing ?
        @sprintf("orbital %d (%s)   %.4f eV%s", k, ro.kind[k] === :e ? "electron" : "hole",
                 ro.energies[k], isempty(lab) ? "" : "   " * lab) : title
    ax = state_axis3(fig[1, 1]; title = ttl)
    geometry && ro.geometry !== nothing && _geometry!(ax, Float64.(_val(ro.geometry)), ro.coords)
    if bands && ro.by_group !== nothing
        G = _val(ro.by_group)[:, :, :, :, k]
        tot = sum(G)
        cols = component_palette(ro.groups)
        els, names = Any[], String[]
        for (g, name) in enumerate(ro.groups)
            ρ = Float64.(G[:, :, :, g])
            wgt = sum(ρ) / tot
            wgt < 0.02 && continue
            m = isosurface_mesh(ρ, ro.coords..., isolevel(ρ, fraction))
            m === nothing && continue
            mesh!(ax, m; color = (cols[g], 0.45), transparency = true)
            push!(els, PolyElement(color = (cols[g], 0.45)))
            push!(names, @sprintf("%s  %.0f%%", name, 100wgt))
        end
        isempty(els) || Legend(fig[2, 1], els, names; orientation = :horizontal,
                               framevisible = false, tellheight = true)
    else
        ρ = Float64.(_val(ro.total)[:, :, :, k])
        m = isosurface_mesh(ρ, ro.coords..., isolevel(ρ, fraction))
        m === nothing || mesh!(ax, m; color = (kind_color(ro.kind[k] === :e ? :electron : :hole), 0.5),
                               transparency = true)
    end
    return fig
end
