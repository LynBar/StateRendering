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

# a 3-D panel: its title as a Label above the axis (an Axis3 with no protrusions would
# clip its own), the axis below it, row 3 left for a legend
function _axis3_titled(gl, ttl)
    Label(gl[1, 1], ttl; fontsize = 15, font = :bold, tellwidth = false)
    return state_axis3(gl[2, 1])
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

# The orbitals' Kramers pairs, E1, E2, … and H1, H2, … in energy order (the orbital levels
# of `species_weights` when every level is a pair): `(; name, members, label, kind)`, with
# `members` archive orbital indices. A spinless or unpaired orbital is a pair of one.
function _orbital_pairs(res::CICalcResults)
    o = res.orbitals !== nothing ? res.orbitals : get(res.extra, "render/orbitals", nothing)
    o === nothing && return NamedTuple[]
    out = NamedTuple[]
    for (kind, pre) in ((:e, "E"), (:h, "H"))
        idx = findall(==(kind), o.kind)
        for (k, p) in enumerate(kramers_pairs(o.energies[idx]))
            mem = idx[p]
            push!(out, (name = pre * string(k), members = mem, label = _orbital_label(res, first(mem)),
                        kind))
        end
    end
    return out
end

# the pair holding archive orbital `k`
_pair_of_orbital(res, k) = something(findfirst(p -> k in p.members, _orbital_pairs(res)), 0)

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
state's static dipole. The densities are the sector's stored ones (`densities:` in the
run spec, at the full grid), or, for any state whose CI vector is kept, contracted from
that vector and the orbital amplitudes (`orbitals: {amplitudes: true}`, on the render
grid). With the render tier stored on the same grid, the dot's geometry is drawn
faintly behind them.
"""
plot_densities(res::CICalcResults, name::AbstractString, i::Integer; size = nothing, kwargs...) =
    _single(pos -> draw_densities!(pos, res, name, i; kwargs...), size)

"""
    draw_densities!(pos, res, name, i; fraction = 0.5, geometry = true, title,
                    amplitudes = nothing) -> (; layout, axis, size, source)

[`plot_densities`](@ref) drawn into a grid position. `source` is `:stored` or
`:amplitudes`; `amplitudes` passes the amplitude array already read.
"""
function draw_densities!(pos, res::CICalcResults, name::AbstractString, i::Integer;
                         fraction::Real = 0.5, geometry::Bool = true, title = nothing,
                         amplitudes = nothing)
    s = _sector_state(res, name, i)
    d, source = _state_densities(res, s, i; amplitudes)
    xyz = d.coords
    gl = GridLayout(pos)
    ax = _axis3_titled(gl, title === nothing ? _state_title(s, i) : title)
    geometry && _geometry!(ax, _render_geometry(res, xyz), xyz)
    _density_legend!(gl[3, 1], _density_pair!(ax, d.e, d.h, xyz, fraction))
    source === :amplitudes &&
        Label(gl[4, 1], "from the CI vector and the orbital amplitudes (render grid, stride " *
                        string(res.extra["render/orbitals"].stride) * ")";
              fontsize = 10, color = _MUTED, tellwidth = false)
    return (; layout = gl, axis = ax, size = (620, 560), source)
end

# a state's (or a level's) reduced densities: stored, else from its vectors and the
# amplitude tier; the hint names both spec keys when neither is there
function _state_densities(res::CICalcResults, s::StoredSector, states; amplitudes = nothing)
    S = collect(Int, states isa Integer ? (states:states) : states)
    if length(S) == 1 && haskey(s.densities, only(S))
        d = s.densities[only(S)]
        return (e = _val(d.e), h = _val(d.h), coords = d.coords), :stored
    end
    if _has_amplitudes(res) && last(S) <= s.nkept
        return reduced_densities(res, s.name, S; amplitudes), :amplitudes
    end
    throw(ArgumentError(
        "no densities for $(s.name) state(s) $(join(S, ", "))" *
        (isempty(s.densities) ? "" : " (stored: $(join(sort!(collect(keys(s.densities))), ", ")))") *
        ". Add `orbitals: {amplitudes: true}` to the run spec's `compute` (every state whose " *
        "vector is kept" * (last(S) <= s.nkept ? "" : "; raise `keep:` of sector $(s.name) to $(last(S))") *
        "), or list the state in `densities:` of sector $(s.name)."))
end

# a state's electron and hole densities as isosurfaces, each enclosing `fraction` of
# its particle; the kinds drawn
function _density_pair!(ax, ρe, ρh, xyz, fraction)
    shown = Symbol[]
    for (ρ, kind) in ((ρe, :electron), (ρh, :hole))
        (ρ === nothing || !(maximum(ρ) > 0)) && continue
        m = isosurface_mesh(Float64.(ρ), xyz..., isolevel(ρ, fraction))
        m === nothing && continue
        mesh!(ax, m; color = (kind_color(kind), 0.5), transparency = true)
        push!(shown, kind)
    end
    return shown
end

_density_legend!(pos, shown) =
    isempty(shown) || Legend(pos, [PolyElement(color = (kind_color(k), 0.5)) for k in shown],
                             string.(shown); orientation = :horizontal, framevisible = false,
                             tellheight = true)

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

- **An exciton (1,1):** a heatmap of `|C|²` over the Kramers pairs of orbitals (E1, H1,
  …; `kramers_pairs`), the axes labelled by their irreps. A cell is the same for every
  state of its manifold Eₖ ⊗ Hₗ; its inset, how the cell splits over the 2 × 2 partners,
  is what tells dark from bright. `pairs = false` draws single orbitals instead.
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
plot_configurations(res::CICalcResults, name::AbstractString, i::Integer; size = nothing,
                    kwargs...) =
    _single(pos -> draw_configurations!(pos, res, name, i; kwargs...), size)

"""
    draw_configurations!(pos, res, name, i; basis = :auto, top = 16, nlevels = 8, title,
                         colormap, pairs = true) -> (; layout, axis, size, kind)

[`plot_configurations`](@ref) drawn into a grid position.
"""
function draw_configurations!(pos, res::CICalcResults, name::AbstractString, i::Integer;
                              basis::Symbol = :auto, top::Integer = 16, nlevels::Integer = 8,
                              title = nothing, colormap = _CONFIG_CMAP, pairs::Bool = true)
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
        return _draw_by_species!(pos, res, s, i; title, colormap)
    end
    if s.sector == (2, 2) && basis in (:auto, :excitons)
        x = _exciton_sector(res, s)
        if x === nothing
            basis === :excitons && throw(ArgumentError(
                "writing $(s.name) in excitons needs the exciton sector with its vectors: " *
                "add `X: {keep: all}` to the run spec's sectors"))
        else
            return _draw_in_excitons!(pos, res, s, x, i; nlevels, title, colormap)
        end
    end
    v = _val(s.vectors)[:, i]
    ne, nh = length(s.estates), length(s.hstates)
    occ(b) = ([e for e in 1:ne if (b >> (e - 1)) & 1 == 1],
              [h for h in 1:nh if (b >> (ne + h - 1)) & 1 == 0])
    olab(st, k, pre) = (l = _orbital_label(res, st[k].orb); isempty(l) ? "$pre$k" : "$pre$k\n$l")
    w = abs2.(v)
    gl = GridLayout(pos)
    ttl = title === nothing ? _state_title(s, i) : title
    if s.sector == (1, 1) && pairs
        W = zeros(ne, nh)
        for (k, b) in enumerate(s.basis)
            es, hs = occ(b)
            W[only(es), only(hs)] += w[k]
        end
        return _draw_exciton_pairs!(gl, res, s, W, ttl, colormap)
    elseif s.sector == (1, 1)
        W = zeros(ne, nh)
        for (k, b) in enumerate(s.basis)
            es, hs = occ(b)
            W[only(es), only(hs)] += w[k]
        end
        ax = Axis(gl[1, 1]; title = ttl, xlabel = "hole", ylabel = "electron",
                  xticks = (1:nh, [olab(s.hstates, h, "h") for h in 1:nh]),
                  yticks = (1:ne, [olab(s.estates, e, "e") for e in 1:ne]),
                  aspect = DataAspect())
        hm = heatmap!(ax, 1:nh, 1:ne, permutedims(W); colormap, colorrange = (0, maximum(W)))
        for e in 1:ne, h in 1:nh
            W[e, h] < 1e-3 && continue
            text!(ax, h, e; text = @sprintf("%.2f", W[e, h]), align = (:center, :center),
                  fontsize = 10, color = W[e, h] > 0.5 * maximum(W) ? :white : :black)
        end
        Colorbar(gl[1, 2], hm; label = "|C|²")
    else
        ord = sortperm(w; rev = true)[1:min(top, length(w))]
        labs = map(ord) do k
            es, hs = occ(s.basis[k])
            join(("e$e" for e in es), " ") * " | " * join(("h$h" for h in hs), " ")
        end
        ax = Axis(gl[1, 1]; title = ttl, ylabel = "|C|²",
                  xticks = (eachindex(ord), labs), xticklabelrotation = π / 4)
        barplot!(ax, eachindex(ord), w[ord]; color = kind_color(:electron))
        Label(gl[2, 1], @sprintf("the %d largest of %d determinants (Σ = %.3f)", length(ord),
                                 length(w), sum(w[ord])); fontsize = 10, color = _MUTED,
              tellwidth = false)
    end
    return (; layout = gl, axis = ax, size = (600, 460),
            kind = s.sector == (1, 1) ? :orbitals : :determinants)
end

# An exciton over Kramers pairs of orbitals: the cell (Eₖ, Hₗ) holds the weight of the
# electron in pair Eₖ and the hole in Hₗ, and an inset shows how that weight splits over
# the 2 × 2 partners. The cell is the same for every state of the manifold Eₖ ⊗ Hₗ (dark
# and bright alike); the inset is what tells them apart.
function _draw_exciton_pairs!(gl, res, s, W, ttl, colormap)
    Pe = kramers_pairs([st.energy for st in s.estates])
    Ph = kramers_pairs([st.energy for st in s.hstates])
    Wp = [sum(W[a, b]) for a in Pe, b in Ph]
    ne, nh = size(Wp)
    lab(st, p, pre, k) = (l = _orbital_label(res, st[first(p)].orb);
                          string(pre, k, length(p) == 2 ? "" : "*") * (isempty(l) ? "" : "\n" * l))
    ax = Axis(gl[1, 1]; title = ttl, xlabel = "hole (Kramers pair)", ylabel = "electron (Kramers pair)",
              xticks = (1:nh, [lab(s.hstates, Ph[b], "H", b) for b in 1:nh]),
              yticks = (1:ne, [lab(s.estates, Pe[a], "E", a) for a in 1:ne]),
              aspect = DataAspect())
    top = max(maximum(Wp), 1e-12)
    hm = heatmap!(ax, 1:nh, 1:ne, permutedims(Wp); colormap, colorrange = (0, top))
    cs = Makie.to_colormap(colormap)
    pick(v) = cs[clamp(round(Int, 1 + v * (length(cs) - 1)), 1, length(cs))]
    for a in 1:ne, b in 1:nh
        Wp[a, b] < 1e-3 && continue
        dark = Wp[a, b] > 0.5top
        text!(ax, b, a + 0.2; text = @sprintf("%.2f", Wp[a, b]), align = (:center, :center),
              fontsize = 10, color = dark ? :white : :black)
        (length(Pe[a]) == 2 && length(Ph[b]) == 2 && Wp[a, b] > 0.02) || continue
        sub = W[Pe[a], Ph[b]] ./ Wp[a, b]
        for r in 1:2, c in 1:2                         # rows: electron partners, top first
            poly!(ax, Rect2f(b - 0.2 + (c - 1) * 0.2, a - 0.02 - r * 0.2, 0.2, 0.2);
                  color = pick(sub[r, c]), strokecolor = dark ? :white : _MUTED, strokewidth = 0.8)
        end
    end
    Colorbar(gl[1, 2], hm; label = "|C|² over the pair (inset: split over its partners)")
    Label(gl[2, 1:2], "cells: Kramers pairs of orbitals (* unpaired); the inset splits a cell over " *
                      "the electron's (rows) and the hole's (columns) partners, on the same colours " *
                      "as a share of the cell";
          fontsize = 10, color = _MUTED, tellwidth = false, word_wrap = true)
    return (; layout = gl, axis = ax, size = (max(560, 110nh + 260), max(460, 100ne + 260)),
            kind = :pairs)
end

function _draw_by_species!(pos, res, s, i; title, colormap, cut = 1e-3)
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
        all(isempty, irr) ? txt : txt * "\n" * join((isempty(x) ? "?" : x for x in irr), " · ")
    end
    nr, nc = length(rows), length(cols)
    gl = GridLayout(pos)
    ax = Axis(gl[1, 1]; title = title === nothing ? _state_title(s, i) : title,
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
    Colorbar(gl[1, 2], hm; label = "weight of (electrons, holes)")
    λ(v) = join((@sprintf("%.3f", x) for x in v[1:min(3, end)]), " ")
    Label(gl[2, 1:2], @sprintf("level %s%s (states %d:%d); the cells shown hold %.3f\nnatural spectrum: electrons %s | holes %s; %d×%d natural states hold %.3f",
                                s.name, string(findfirst(r -> i in r, s.levels)), first(l), last(l), sum(W),
                                λ(w.λe), λ(w.λh), w.product.ranks..., w.product.weight);
          fontsize = 10, color = _MUTED, tellwidth = false)
    return (; layout = gl, axis = ax, size = (max(600, 110nc + 260), max(460, 60nr + 280)),
            kind = :species)
end

function _draw_in_excitons!(pos, res, s, x, i; nlevels, title, colormap)
    w = exciton_pair_weights(SectorSolution(s), SectorSolution(x); states = [i],
                             xlevels = x.levels, nlevels)
    L = length(w.levels)
    W = w.W[:, :, 1]
    Z = [a <= b ? W[a, b] : NaN for a in 1:L, b in 1:L]
    lab(k) = isempty(x.labels) || isempty(x.labels[k]) ? "" : x.labels[k]
    names = [string(x.name, k, "  ", lab(k)) for k in 1:L]
    top = maximum(filter(isfinite, Z); init = 0.0)
    gl = GridLayout(pos)
    ax = Axis(gl[1, 1]; title = title === nothing ? _state_title(s, i) : title,
              xlabel = "exciton level", ylabel = "exciton level",
              xticks = (1:L, names), yticks = (1:L, names), xticklabelrotation = π / 4,
              aspect = DataAspect())
    hm = heatmap!(ax, 1:L, 1:L, Z; colormap, colorrange = (0, max(top, 1e-12)), nan_color = :transparent)
    for a in 1:L, b in a:L
        W[a, b] < 5e-3 && continue
        text!(ax, a, b; text = @sprintf("%.2f", W[a, b]), align = (:center, :center), fontsize = 10,
              color = W[a, b] > 0.5top ? :white : :black)
    end
    Colorbar(gl[1, 2], hm; label = "weight of the exciton pair (A, B)")
    Label(gl[2, 1:2], @sprintf("¼ Σ |⟨XX|Xₐ† X_b†|0⟩|² over the two levels; the lowest %d levels hold %.3f of the state",
                                L, w.captured[1]); fontsize = 10, color = _MUTED, tellwidth = false)
    return (; layout = gl, axis = ax, size = (max(560, 70L + 300), max(520, 60L + 300)),
            kind = :excitons, levels = w.levels, W, exciton_sector = x.name)
end

# ── natural orbitals ─────────────────────────────────────────────────────────

# A group of natural orbitals, by how many carriers it holds: Σ n over the group. A
# Kramers pair holds up to 2. Counting per pair, not per spin-orbital, is what reads
# right for a level: averaged over a Kramers-degenerate level, one hole shows as 0.5 in
# each partner, yet the pair holds the whole hole, 1.00. Only a level's average is
# time-reversal symmetric, so only there is a group of two a Kramers pair (`paired`); for
# one state a group of two is just two orbitals of one occupation.
function _no_name(kind, g, grp; paired::Bool = true)
    m = length(grp.n)
    what = m == 1 ? "1 orbital" : !(paired && iseven(m)) ? "$m orbitals" :
           m == 2 ? "a Kramers pair" : "$(m ÷ 2) Kramers pairs"
    return @sprintf("%s NO %d   %.2f %s in %s", kind === :e ? "e" : "h", g, sum(grp.n),
                    kind === :e ? "electrons" : "holes", what)
end

# every Axis3 of a figure turns with the others: azimuth, elevation and zoom
function _link_cameras!(axes)
    axs = [a for a in axes if a !== nothing]
    length(axs) < 2 && return nothing
    for attr in (:azimuth, :elevation, :zoommult), a in axs
        attr in propertynames(a) || continue
        on(getproperty(a, attr)) do v
            for b in axs
                b === a && continue
                o = getproperty(b, attr)
                o[] == v || (o[] = v)
            end
        end
    end
    return nothing
end

"""
    plot_natural_orbitals(res::CICalcResults, name, states; groups = 3, fraction = 0.5,
                          geometry = true) -> Figure

The natural orbitals of a stored state, or of a level (pass its states: the density
matrix is averaged over them), from its CI vectors and the orbital amplitudes
(`orbitals: {amplitudes: true}`). The `groups` most occupied of each species, electrons
on top and holes below.

**What is drawn.** The density matrix of each species is diagonalised: its eigenvectors
are the natural orbitals (combinations of the stored orbitals), its eigenvalues their
occupations, and the species' density is Σ occupation × |natural orbital|². A group is
the natural orbitals with one occupation (within 1e-3): a Kramers pair, or pairs the
breaking mixes. Each panel draws a group's summed density, the isosurface enclosing
`fraction` of it, titled with how many carriers the group holds (Σ of its occupations;
a pair holds up to 2).

**How to read it.** One configuration (one Slater determinant per species) puts whole
carriers in a few groups: 2.00 electrons in a pair is a closed shell. Correlation
spreads carriers over more groups, and the leftovers (0.01, 0.02, …) are how much.
For a level, the matrix is averaged over its states: a Kramers-degenerate trion level
(X⁻: states 1, 2) has its one hole half in each partner of a pair in each state's
average, which is why it reads "1.00 holes in a Kramers pair", not a full occupation
of one orbital. Within one Kramers state the hole is in one partner; the pair's
density is the same either way.
"""
plot_natural_orbitals(res::CICalcResults, name::AbstractString, states; size = nothing, kwargs...) =
    _single(pos -> draw_natural_orbitals!(pos, res, name, states; kwargs...), size)

"""
    draw_natural_orbitals!(pos, res, name, states; groups = 3, pick = nothing,
                           fraction = 0.5, geometry = true, amplitudes = nothing, no = nothing)
        -> (; layout, axis, size, no)

[`plot_natural_orbitals`](@ref) drawn into a grid position. `pick = (:e, g)` or `(:h, g)`
draws that one group alone; `no` passes `natural_orbitals(res, name, states)` already
computed.
"""
function draw_natural_orbitals!(pos, res::CICalcResults, name::AbstractString, states;
                                groups::Integer = 3, pick = nothing, fraction::Real = 0.5,
                                geometry::Bool = true, amplitudes = nothing, no = nothing)
    S = collect(Int, states isa Integer ? (states:states) : states)
    s = _sector_state(res, name, first(S))
    _has_amplitudes(res) || _missing_product("orbital amplitudes (the amplitude tier)",
        "Add `orbitals: {amplitudes: true}` to the run spec's `compute`.")
    no = no === nothing ? natural_orbitals(res, name, S; amplitudes) : no
    χ = geometry ? _render_geometry(res, no.coords) : nothing
    which = length(S) == 1 ? @sprintf("%s%d", s.name, only(S)) :
            @sprintf("%s states %d:%d (the level's average)", s.name, first(S), last(S))
    gl = GridLayout(pos)
    function panel(p, kind, g)
        grp = getproperty(no, kind)[g]
        sub = GridLayout(p)
        ax = _axis3_titled(sub, _no_name(kind, g, grp; paired = length(S) > 1))
        χ === nothing || _geometry!(ax, χ, no.coords)
        ρ = grp.density
        m = isosurface_mesh(ρ, no.coords..., isolevel(ρ, fraction))
        m === nothing || mesh!(ax, m; color = (kind_color(kind === :e ? :electron : :hole), 0.5),
                               transparency = true)
        return ax
    end
    occ(v) = join((@sprintf("%.2f", sum(g.n)) for g in v[1:min(4, end)]), " + ")
    foot = "carriers per group (a Kramers pair holds up to 2; groups within 1e-3 merge, mixed " *
           "by the breaking): electrons " * occ(no.e) * " | holes " * occ(no.h)
    if pick !== nothing
        kind, g = pick
        1 <= g <= length(getproperty(no, kind)) || throw(ArgumentError(
            "no natural-orbital group $g for the $(kind === :e ? "electrons" : "holes")"))
        Label(gl[1, 1], which; fontsize = 13, color = _INK, tellwidth = false)
        ax = panel(gl[2, 1], kind, g)
        Label(gl[3, 1], foot; fontsize = 10, color = _MUTED, tellwidth = false, word_wrap = true)
        return (; layout = gl, axis = ax, size = (620, 600), no)
    end
    ne, nh = min(groups, length(no.e)), min(groups, length(no.h))
    nc = max(ne, nh, 1)
    Label(gl[1, 1:nc], which * ": natural orbitals"; fontsize = 15, font = :bold, tellwidth = false)
    axs = [panel(gl[row, g], kind, g) for (row, kind, n) in ((2, :e, ne), (3, :h, nh)) for g in 1:n]
    _link_cameras!(axs)
    first_ax = isempty(axs) ? nothing : first(axs)
    Label(gl[4, 1:nc], foot; fontsize = 10, color = _MUTED, tellwidth = false, word_wrap = true)
    return (; layout = gl, axis = first_ax, size = (320nc + 40, 700), no)
end

# ── constituents: the two excitons of a biexciton's exciton pair ─────────────

"""
    plot_constituents(res::CICalcResults, name, i; cell = nothing, nlevels = 8,
                      fraction = 0.5, geometry = true) -> Figure

A biexciton state written in excitons (`exciton_pair_weights`), shown by its pair: the
densities of the two exciton levels A and B of `cell = (A, B)`, by default the pair
holding the most weight, each averaged over its level. Needs the X sector's vectors
and the orbital amplitudes. The two excitons are the state's description, not its
density: the state's own densities are `plot_densities`.
"""
plot_constituents(res::CICalcResults, name::AbstractString, i::Integer; size = nothing, kwargs...) =
    _single(pos -> draw_constituents!(pos, res, name, i; kwargs...), size)

"""
    draw_constituents!(pos, res, name, i; cell = nothing, nlevels = 8, fraction = 0.5,
                       geometry = true, amplitudes = nothing) -> (; layout, axis, size, cell, cells)

[`plot_constituents`](@ref) drawn into a grid position. `cells` lists every pair
`(A, B, weight)`, heaviest first.
"""
function draw_constituents!(pos, res::CICalcResults, name::AbstractString, i::Integer;
                            cell = nothing, nlevels::Integer = 8, fraction::Real = 0.5,
                            geometry::Bool = true, amplitudes = nothing)
    s = _sector_state(res, name, i)
    s.sector == (2, 2) || throw(ArgumentError(
        "constituents are the exciton pairs of a biexciton (2,2); $(s.name) is $(s.sector)"))
    i <= s.nkept || throw(ArgumentError(
        "the CI vectors of $(s.name) are stored for its lowest $(s.nkept) states; raise `keep:`"))
    x = _exciton_sector(res, s)
    x === nothing && throw(ArgumentError(
        "writing $(s.name) in excitons needs the exciton sector with its vectors: add " *
        "`X: {keep: all}` to the run spec's sectors"))
    cells = _pair_cells(res, s, x, i; nlevels)
    A, B = cell === nothing ? (cells[1][1], cells[1][2]) : minmax(cell...)
    k = findfirst(c -> (c[1], c[2]) == (A, B), cells)
    k === nothing && throw(ArgumentError("($A, $B) is not a pair of the lowest $nlevels exciton levels"))
    wt = cells[k][3]
    lv = x.levels
    gl = GridLayout(pos)
    Label(gl[1, 1:2], @sprintf("%s  ⊃  %s%d · %s%d   (weight %.3f)", _state_title(s, i), x.name, A,
                               x.name, B, wt); fontsize = 15, font = :bold, tellwidth = false)
    axs = Any[]
    for (c, L) in enumerate(A == B ? (A,) : (A, B))
        d, _ = _state_densities(res, x, collect(lv[L]); amplitudes)
        sub = GridLayout(gl[2, c])
        lab = isempty(x.labels) || isempty(x.labels[L]) ? "" : "   " * x.labels[L]
        ax = _axis3_titled(sub, @sprintf("%s%d%s%s", x.name, L, lab, A == B ? "  (twice)" : ""))
        geometry && _geometry!(ax, _render_geometry(res, d.coords), d.coords)
        _density_legend!(sub[3, 1], _density_pair!(ax, d.e, d.h, d.coords, fraction))
        push!(axs, ax)
    end
    _link_cameras!(axs)
    first_ax = first(axs)
    Label(gl[3, 1:2], "each exciton level's densities, averaged over its states; the weight is " *
                      "¼ Σ |⟨XX|Xₐ† X_b†|0⟩|² over the two levels";
          fontsize = 10, color = _MUTED, tellwidth = false)
    return (; layout = gl, axis = first_ax, size = (A == B ? 620 : 1100, 600), cell = (A, B), cells)
end

# (A, B, weight) of an XX state over pairs of the lowest exciton levels, heaviest first
function _pair_cells(res, s::StoredSector, x::StoredSector, i; nlevels = 8)
    w = exciton_pair_weights(SectorSolution(s), SectorSolution(x); states = [i],
                             xlevels = x.levels, nlevels)
    L = length(w.levels)
    cells = [(a, b, w.W[a, b, 1]) for a in 1:L for b in a:L]
    return sort!(cells; by = c -> -c[3])
end

"""
    plot_orbital(res::CICalcResults, k; fraction = 0.5, bands = true) -> Figure

Single-particle orbital `k` from the render tier: the isosurface enclosing `fraction`
of |ψ|², one surface per band group (S, HH, LH, SO or S, X, Y, Z) holding at least 2 %
of the weight, coloured by group, or the total alone with `bands = false`. The render
tier is a heavy product: add `orbitals:` to the run spec.

An orbital is drawn as its Kramers pair (`pair = true`, named E1, H2, … in energy
order): time reversal leaves |ψ|² and its band groups unchanged, so the two partners
are one picture, and the pair's mean is drawn.
"""
plot_orbital(res::CICalcResults, k::Integer; size = nothing, kwargs...) =
    _single(pos -> draw_orbital!(pos, res, k; kwargs...), size)

"""
    draw_orbital!(pos, res, k; fraction = 0.5, bands = true, geometry = true, pair = true,
                  title) -> (; layout, axis, size)

[`plot_orbital`](@ref) drawn into a grid position.
"""
function draw_orbital!(pos, res::CICalcResults, k::Integer; fraction::Real = 0.5,
                       bands::Bool = true, geometry::Bool = true, pair::Bool = true,
                       title = nothing)
    ro = get(res.extra, "render/orbitals", nothing)
    ro === nothing && _missing_product("render tier (orbitals on a coarse grid)",
                                       "Add `orbitals: {stride: 2}` to the run spec's `compute`.")
    n = length(ro.energies)
    1 <= k <= n || throw(ArgumentError("orbital $k is out of range 1:$n"))
    lab = _orbital_label(res, k)
    j = pair ? _pair_of_orbital(res, k) : 0
    pr = j == 0 ? nothing : _orbital_pairs(res)[j]
    mem = pr === nothing ? [k] : pr.members
    gl = GridLayout(pos)
    ttl = title !== nothing ? title : pr !== nothing && length(mem) == 2 ?
        @sprintf("%s (orbitals %d, %d: a Kramers pair)   %.4f eV%s", pr.name, mem...,
                 ro.energies[k], isempty(lab) ? "" : "   " * lab) :
        @sprintf("orbital %d (%s)   %.4f eV%s", k, ro.kind[k] === :e ? "electron" : "hole",
                 ro.energies[k], isempty(lab) ? "" : "   " * lab)
    ax = _axis3_titled(gl, ttl)
    geometry && ro.geometry !== nothing && _geometry!(ax, Float64.(_val(ro.geometry)), ro.coords)
    if bands && ro.by_group !== nothing
        G = sum(_val(ro.by_group)[:, :, :, :, q] for q in mem) ./ length(mem)
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
        isempty(els) || Legend(gl[3, 1], els, names; orientation = :horizontal,
                               framevisible = false, tellheight = true)
    else
        ρ = Float64.(sum(_val(ro.total)[:, :, :, q] for q in mem) ./ length(mem))
        m = isosurface_mesh(ρ, ro.coords..., isolevel(ρ, fraction))
        m === nothing || mesh!(ax, m; color = (kind_color(ro.kind[k] === :e ? :electron : :hole), 0.5),
                               transparency = true)
    end
    return (; layout = gl, axis = ax, size = (620, 560))
end
