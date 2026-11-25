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

# the configuration heatmaps: one hue, near-white (empty) to deep blue (the whole state);
# the labelling figures' `_SEQ` ramp with a near-white foot
const _CONFIG_CMAP = cgrad([RGBf(0.97, 0.98, 0.995), RGBf(0xcd / 255, 0xe2 / 255, 0xfb / 255),
                            RGBf(0x86 / 255, 0xb6 / 255, 0xef / 255), RGBf(0x39 / 255, 0x87 / 255, 0xe5 / 255),
                            RGBf(0x1c / 255, 0x5c / 255, 0xab / 255), RGBf(0x0d / 255, 0x36 / 255, 0x6b / 255)])

# a 3-D figure's title: as a Label above the axis, which an Axis3 with no protrusions
# would clip
function _axis3_titled(fig, ttl)
    Label(fig[0, 1], ttl; fontsize = 15, font = :bold, tellwidth = false)
    return state_axis3(fig[1, 1])
end

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
    ax = _axis3_titled(fig, title === nothing ? _state_title(s, i) : title)
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

# the stored exciton sector a biexciton can be written in: (1,1), with vectors kept,
# over the same active orbitals
function _exciton_sector(res::CICalcResults, xx::StoredSector)
    for (n, s) in sort!(collect(res.sectors); by = first)
        s.sector == (1, 1) && s.nkept > 0 &&
            [e.orb for e in s.estates] == [e.orb for e in xx.estates] &&
            [h.orb for h in s.hstates] == [h.orb for h in xx.hstates] && return s
    end
    return nothing
end

"""
    plot_configurations(res::CICalcResults, name, i; basis = :auto, top = 16,
                        nlevels = 8) -> Figure

What state `i` of a stored sector is made of.

- **An exciton (1,1):** a heatmap of `|C|²` over (electron, hole) orbital pairs, the
  axes labelled by the orbitals' irreps.
- **A biexciton (2,2):** written in terms of excitons. The heatmap is over pairs of
  the lowest `nlevels` exciton levels (A, B), and each cell is the weight of that
  exciton pair in the state (`exciton_pair_weights`).
  - Over all levels the weights sum to 1.
  - An uncorrelated determinant is half of each of its two e–h pairings.
  - A weight does not depend on the basis within a degenerate level.

  This needs the archive's X sector with its vectors (`keep:`). `basis =
  :determinants` gives the determinant bar chart instead.
- **A trion (2,1) or (1,2):** split into its electrons ⊗ its holes
  (`species_weights`): a heatmap over configurations of orbital levels, the pair of
  identical carriers on one axis and the spectator on the other. Unlike exciton +
  carrier, this never splits a state between its e–h pairings. The level holding
  state `i` is shown, with each species' natural spectrum underneath.
- **`basis = :species`** gives that split for any sector with electrons and holes. For
  a biexciton it is the electron pair against the hole pair.
- **Any other sector:** a bar chart of the `top` largest determinants.

A level that is one irrep should be supported only on configurations whose product
contains it, so an occupied but forbidden cell means the labelling or the adaptation
is wrong. Needs the state's CI vector (`keep:` of the sector).
"""
function plot_configurations(res::CICalcResults, name::AbstractString, i::Integer;
                             basis::Symbol = :auto, top::Integer = 16, nlevels::Integer = 8,
                             size = nothing, title = nothing, colormap = _CONFIG_CMAP)
    s = _sector_state(res, name, i)
    i <= s.nkept || throw(ArgumentError(
        "the CI vectors of $(s.name) are stored for its lowest $(s.nkept) states; " *
        "raise `keep:` of sector $(s.name) in the run spec to see state $i"))
    basis in (:auto, :excitons, :species, :determinants) || throw(ArgumentError(
        "basis must be :auto, :excitons, :species or :determinants"))
    Ne, Nh = s.sector
    if basis === :species || (basis === :auto && Ne >= 1 && Nh >= 1 && Ne + Nh == 3)
        Ne >= 1 && Nh >= 1 || throw(ArgumentError(
            "basis = :species needs a sector with both electrons and holes; $(s.name) is $(s.sector)"))
        return _plot_by_species(res, s, i; size, title, colormap)
    end
    if s.sector == (2, 2) && basis in (:auto, :excitons)
        x = _exciton_sector(res, s)
        if x === nothing
            basis === :excitons && throw(ArgumentError(
                "writing $(s.name) in excitons needs the exciton sector with its vectors: " *
                "add `X: {keep: all}` to the run spec's sectors"))
        else
            return _plot_in_excitons(res, s, x, i; nlevels, size, title, colormap)
        end
    end
    size = something(size, (600, 460))
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

function _plot_by_species(res, s, i; size, title, colormap, cut = 1e-3)
    l = s.levels[findfirst(r -> i in r, s.levels)]
    last(l) <= s.nkept || throw(ArgumentError(
        "the level of $(s.name)$i is states $l, but only the lowest $(s.nkept) vectors are " *
        "stored; raise `keep:` of sector $(s.name) in the run spec"))
    S = SectorSolution(s)
    w = species_weights(S, collect(l))
    rows = [r for r in axes(w.W, 1) if maximum(w.W[r, :]) > cut]
    cols = [c for c in axes(w.W, 2) if maximum(w.W[:, c]) > cut]
    W = w.W[rows, cols]
    # a configuration: its levels, and their irreps underneath
    function name(c, lv, st, p)
        irr = [_orbital_label(res, st[first(lv[k])].orb) for k in c]
        txt = _config_name(c, p)
        all(isempty, irr) ? txt : txt * "
" * join((isempty(x) ? "?" : x for x in irr), " · ")
    end
    nr, nc = length(rows), length(cols)
    fig = Figure(size = something(size, (max(600, 110nc + 260), max(460, 60nr + 280))))
    ax = Axis(fig[1, 1]; title = title === nothing ? _state_title(s, i) : title,
              xlabel = s.sector[2] >= 2 ? "hole pair" : "hole",
              ylabel = s.sector[1] >= 2 ? "electron pair" : "electron",
              xticks = (1:nc, [name(w.hconfigs[c], w.hlevels, s.hstates, "H") for c in cols]),
              yticks = (1:nr, [name(w.econfigs[r], w.elevels, s.estates, "E") for r in rows]),
              xticklabelrotation = nc > 4 ? π / 4 : 0.0)
    hm = heatmap!(ax, 1:nc, 1:nr, permutedims(W); colormap, colorrange = (0, maximum(W)))
    top = maximum(W)
    for r in 1:nr, c in 1:nc
        W[r, c] < 5e-3 && continue
        text!(ax, c, r; text = @sprintf("%.3f", W[r, c]), align = (:center, :center), fontsize = 10,
              color = W[r, c] > 0.5top ? :white : :black)
    end
    Colorbar(fig[1, 2], hm; label = "weight of (electrons, holes)")
    λ(v) = join((@sprintf("%.3f", x) for x in v[1:min(3, end)]), " ")
    Label(fig[2, 1:2], @sprintf("level %s%s (states %d:%d); the cells shown hold %.3f
natural spectrum: electrons %s | holes %s; %d×%d natural states hold %.3f",
                                s.name, string(findfirst(r -> i in r, s.levels)), first(l), last(l), sum(W),
                                λ(w.λe), λ(w.λh), w.product.ranks..., w.product.weight);
          fontsize = 10, color = _MUTED, tellwidth = false)
    return fig
end

function _plot_in_excitons(res, s, x, i; nlevels, size, title, colormap)
    w = exciton_pair_weights(SectorSolution(s), SectorSolution(x); states = [i],
                             xlevels = x.levels, nlevels)
    L = length(w.levels)
    W = w.W[:, :, 1]
    Z = [a <= b ? W[a, b] : NaN for a in 1:L, b in 1:L]
    lab(k) = isempty(x.labels) || isempty(x.labels[k]) ? "" : x.labels[k]
    names = [string(x.name, k, "  ", lab(k)) for k in 1:L]
    top = maximum(filter(isfinite, Z); init = 0.0)
    fig = Figure(size = something(size, (max(560, 70L + 300), max(520, 60L + 300))))
    ax = Axis(fig[1, 1]; title = title === nothing ? _state_title(s, i) : title,
              xlabel = "exciton level", ylabel = "exciton level",
              xticks = (1:L, names), yticks = (1:L, names), xticklabelrotation = π / 4,
              aspect = DataAspect())
    hm = heatmap!(ax, 1:L, 1:L, Z; colormap, colorrange = (0, max(top, 1e-12)), nan_color = :transparent)
    for a in 1:L, b in a:L
        W[a, b] < 5e-3 && continue
        text!(ax, a, b; text = @sprintf("%.2f", W[a, b]), align = (:center, :center), fontsize = 10,
              color = W[a, b] > 0.5top ? :white : :black)
    end
    Colorbar(fig[1, 2], hm; label = "weight of the exciton pair (A, B)")
    Label(fig[2, 1:2], @sprintf("¼ Σ |⟨XX|Xₐ† X_b†|0⟩|² over the two levels; the lowest %d levels hold %.3f of the state",
                                L, w.captured[1]); fontsize = 10, color = _MUTED, tellwidth = false)
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
    ax = _axis3_titled(fig, ttl)
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
