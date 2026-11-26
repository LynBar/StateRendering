# ---------------------------------------------------------------------------
# The explorer of a results archive: a Grotrian map of the levels, seen through a
# lens, with detail panels and a 3-D window that follow the selection.
#
# Everything is drawn from the `ExplorerModel` (model.jl) and the `draw_*!` panels
# the static figures use. Interaction is plain Makie: clicks and keys are read from
# `events`, positions from `mouseposition`, so it builds and responds the same under
# any backend, and the tests drive it without a screen. GLMakie makes it a window.
# ---------------------------------------------------------------------------

const _LENSES = ["recombination", "symmetry", "composition"]
const _SUBGROUP_CANDIDATES = ["C6v", "C6", "C4v", "C4", "D2d", "S4", "C3v", "C3", "C2v", "C2", "Cs"]

struct Explorer
    fig::Figure
    fig3d::Figure
    model::ExplorerModel
    lens::Observable{String}
    group::Observable{String}
    map_axis::Axis
    fine_axis::Axis
    segments::Ref{Vector{NamedTuple}}   # what the map draws: (id, x, y, label) per line
    state::Dict{Symbol,Any}             # menus, the 3-D view's camera, …
end

Base.show(io::IO, e::Explorer) =
    print(io, "Explorer(", e.lens[], " lens, ", e.group[], "; ", e.model, ")")

# ── the map's geometry: sectors side by side, an irrep column each ──────────

# the archive's frame and its irrep labels, or nothing
function _stored_frame(res)
    (res.orbitals === nothing || !haskey(res.orbitals, :frame)) && return nothing
    f = res.orbitals.frame
    return (; group = String(f.group), axis = f.axis, inplane = f.inplane, centre = f.centre)
end

_frame_of(fr, g) = SymmetryFrame(g; centre = fr.centre, axis = fr.axis, inplane = fr.inplane)

# the groups the map can be drawn in: the frame's, and each subgroup in the same frame
function _map_groups(res)
    fr = _stored_frame(res)
    fr === nothing && return ["—"]
    high = _frame_of(fr, fr.group)
    subs = filter(_SUBGROUP_CANDIDATES) do g
        g == fr.group && return false
        try
            subduction(high, _frame_of(fr, g)); true
        catch
            false
        end
    end
    return [fr.group; subs]
end

# a label such as "Γ5", "2Γ4 ⊕ Γ5" as multiplicities over `labels`; nothing for "?"
function _label_content(label, labels)
    c = zeros(Int, length(labels))
    for part in split(label, '⊕')
        p = strip(part)
        m = match(r"^(\d*)(.*)$", p)
        k = findfirst(==(m.captures[2]), labels)
        k === nothing && return nothing
        c[k] += isempty(m.captures[1]) ? 1 : parse(Int, m.captures[1])
    end
    return c
end

# Each node's place on the map, in group `g`: one segment per irrep of `g` it holds
# (in a subgroup a level can sit in several columns: that is the split the lowering
# allows). Returns the segments, the x ticks and the sector headers.
function _map_geometry(e_or_model, g::AbstractString)
    m = e_or_model
    res = m.res
    fr = _stored_frame(res)
    high_labels = fr === nothing ? String[] : collect(String, res.orbitals.table.labels)
    sub = nothing
    low_labels = high_labels
    if fr !== nothing && g != fr.group && g != "—"
        S = subduction(_frame_of(fr, fr.group), _frame_of(fr, g))
        sub = S.table
        low_labels = collect(String, S.low.table.labels)
    end
    parts(x) = begin
        if sub === nothing
            [x.label]
        else
            c = _label_content(x.label, high_labels)
            c === nothing ? ["?"] : [low_labels[b] for b in findall(>(0), vec(c' * sub))]
        end
    end
    segs = NamedTuple[]
    ticks, ticklabels, headers = Float64[], String[], Tuple{Float64,String}[]
    x0 = 0.0
    dividers = Float64[]
    charge(n) = (c = res.sectors[n].sector; c[2] - c[1])
    for (j, n) in enumerate(m.sectors)
        if j > 1 && charge(n) != charge(m.sectors[j-1])
            push!(dividers, x0 - 0.4)
            x0 += 0.8
        end
        xs = [x for x in m.nodes if x.sector == n]
        cols = if sub === nothing
            m.columns[n]
        else
            labs = unique(reduce(vcat, parts.(xs); init = String[]))
            rank(l) = (k = findfirst(==(l), low_labels); k === nothing ? (2, l) : (1, lpad(k, 3)))
            sort(labs; by = rank)
        end
        for x in xs, l in parts(x)
            push!(segs, (id = x.id, x = x0 + findfirst(==(l), cols) - 1, y = x.dE, label = l))
        end
        append!(ticks, x0 .+ (0:length(cols)-1))
        append!(ticklabels, cols)
        push!(headers, (x0 + (length(cols) - 1) / 2, n))
        x0 += length(cols) + 0.8
    end
    return (; segs, ticks, ticklabels, headers, dividers, labels = low_labels)
end

# ── colours and widths per lens ─────────────────────────────────────────────

function _node_axial(m::ExplorerModel, id)
    o = m.out[id]
    isempty(o) && return NaN
    t = sum(m.lines[k].rate for k in o)
    return sum(m.lines[k].rate * m.lines[k].axial for k in o) / t
end

# how much of a node is one configuration of orbital levels, electrons ⊗ holes (the
# largest cell of `species_weights`): basis-free, and blind to the spin entanglement
# that makes a 1-D level's natural spectrum ½, ½. NaN without its vectors.
function _product_weight(e::Explorer, id)
    m = e.model
    get!(m.cache, (:configuration, id)) do
        x = m.nodes[id]
        s = m.res.sectors[x.sector]
        (x.charge[1] >= 1 && x.charge[2] >= 1 && last(x.states) <= s.nkept) || return NaN
        S = get!(() -> SectorSolution(s), m.cache, (:solution, x.sector))
        maximum(species_weights(S, x.states).W)
    end
end

function _segment_style(e::Explorer, labels)
    m = e.model
    lens = e.lens[]
    segs = e.segments[]
    if lens == "recombination"
        top = Dict(n => maximum(filter(isfinite, [m.total[x.id] for x in m.nodes if x.sector == n]);
                                init = 0.0) for n in m.sectors)
        cols = map(segs) do s
            a = _node_axial(m, s.id)
            isfinite(a) ? _POL_GRAD[clamp(a, 0, 1)] : UNLABELLED_COLOR
        end
        widths = map(segs) do s
            t, T = m.total[s.id], top[m.nodes[s.id].sector]
            (isfinite(t) && T > 0) ? 1 + 4clamp(1 + log10(max(t / T, 1e-6)) / 6, 0, 1) : 1.0
        end
    elseif lens == "composition"
        cols = map(segs) do s
            w = _product_weight(e, s.id)
            isfinite(w) ? _CONFIG_CMAP[clamp(w, 0, 1)] : UNLABELLED_COLOR
        end
        widths = fill(3.0, length(segs))
    else
        cols = [irrep_color(s.label, labels) for s in segs]
        widths = fill(3.0, length(segs))
    end
    return cols, widths
end

# ── drawing ─────────────────────────────────────────────────────────────────

function _draw_map!(e::Explorer; keep_limits::Bool = true)
    ax, m = e.map_axis, e.model
    lims = ax.targetlimits[]
    empty!(ax)
    geo = _map_geometry(m, e.group[])
    e.segments[] = geo.segs
    segs = geo.segs
    sel = m.selection[]
    cols, widths = _segment_style(e, geo.labels)
    ax.xticks = (geo.ticks, geo.ticklabels)
    ymax = isempty(segs) ? 1.0 : maximum(s.y for s in segs)
    isempty(geo.dividers) || vlines!(ax, geo.dividers; color = _GRID, linewidth = 1.5,
                                     inspectable = false)
    for (xc, n) in geo.headers
        text!(ax, xc, ymax * 1.06 + 1; text = n, align = (:center, :bottom), fontsize = 14,
              font = :bold, color = _INK)
    end
    center(id) = (ss = [s for s in segs if s.id == id]; isempty(ss) ? nothing : first(ss))
    # the cascade through the selection: where it decays (grey) and what feeds it (orange)
    if e.lens[] == "recombination"
        for (flow, col) in ((decay_tree(m, sel), (_INK, 0.18)),
                            (feeders(m, sel), (RGBf(0.85, 0.45, 0.1), 0.25)))
            pts = Point2f[]; ms = Float64[]
            for (id, f) in flow, s in segs
                s.id == id || continue
                push!(pts, Point2f(s.x, s.y)); push!(ms, 8 + 26sqrt(f))
            end
            isempty(pts) || scatter!(ax, pts; markersize = ms, color = col, strokewidth = 0,
                                     inspectable = false)
        end
        # the lines out of and into the selection
        for (ks, ls) in ((m.out[sel], :solid), (m.into[sel], :dash))
            pts = Point2f[]; c = RGBf[]; w = Float64[]
            T = maximum((m.lines[k].rate for k in ks); init = 0.0)
            for k in ks[1:min(12, end)]
                l = m.lines[k]
                a, b = center(l.from), center(l.to)
                (a === nothing || b === nothing) && continue
                append!(pts, (Point2f(a.x, a.y), Point2f(b.x, b.y)))
                cc = _POL_GRAD[clamp(l.axial, 0, 1)]
                append!(c, (cc, cc))
                ww = 0.5 + 3clamp(1 + log10(max(l.rate / T, 1e-6)) / 6, 0, 1)
                append!(w, (ww, ww))
            end
            isempty(pts) || linesegments!(ax, pts; color = c, linewidth = w, linestyle = ls,
                                          inspectable = false)
        end
    end
    # the selection, behind its line
    for s in segs
        s.id == sel || continue
        linesegments!(ax, [Point2f(s.x - 0.45, s.y), Point2f(s.x + 0.45, s.y)];
                      color = _INK, linewidth = maximum(widths) + 5, inspectable = false)
    end
    pts = reduce(vcat, [[Point2f(s.x - 0.38, s.y), Point2f(s.x + 0.38, s.y)] for s in segs];
                 init = Point2f[])
    linesegments!(ax, pts; color = repeat(cols; inner = 2), linewidth = repeat(widths; inner = 2),
                  inspector_label = (self, i, p) -> _hover_text(e, segs[cld(i, 2)].id))
    if keep_limits && all(isfinite, lims.origin) && all(>(0), lims.widths)
        ax.targetlimits[] = lims
    else
        reset_limits!(ax)
    end
    return nothing
end

function _hover_text(e::Explorer, id)
    m = e.model
    t = m.total[id]
    τ = isfinite(t) && t > 0 ? "\n" * _fmt_tau(1 / t) : ""
    return node_name(m, id) * "\nstates " * join(m.nodes[id].states, ", ") * τ
end

# the selected level's fine structure, in µeV, with the symmetric fit where an
# effective-Hamiltonian product covers exactly this level
function _draw_fine!(e::Explorer)
    ax, m = e.fine_axis, e.model
    empty!(ax)
    x = m.nodes[m.selection[]]
    s = m.res.sectors[x.sector]
    parts = [y for y in m.nodes if y.sector == x.sector && y.level == x.level]
    st = sort!(reduce(vcat, [y.states for y in parts]))
    E0 = minimum(s.energies[st])
    irreps = _map_geometry(m, e.group[]).labels
    spread = 1e6 * (maximum(s.energies[st]) - E0)
    # states within a column that are degenerate to the drawing sit side by side
    fine = NamedTuple[]
    for (c, y) in enumerate(parts)
        n = length(y.states)
        for (j, i) in enumerate(y.states)
            v = 1e6 * (s.energies[i] - E0)
            dx = n > 1 ? 0.7 / n : 0.7
            x0 = c - 0.35 + (j - 1) * dx
            push!(fine, (state = i, x = x0 + dx / 2, y = v, w = dx))
            i == m.focus[] && lines!(ax, [x0 + 0.02, x0 + dx - 0.02], [v, v]; color = _INK,
                                     linewidth = 8, inspectable = false)
            lines!(ax, [x0 + 0.04, x0 + dx - 0.04], [v, v]; color = irrep_color(y.label, irreps),
                   linewidth = 3, inspector_label = (_, _, _) -> _state_text(e, i))
            text!(ax, x0 + dx / 2, v; text = string(i), align = (:center, :bottom), fontsize = 9,
                  color = _MUTED, offset = (0, 3))
        end
    end
    e.state[:fine] = fine
    ax.xticks = (1:length(parts), [y.label for y in parts])
    ax.title = @sprintf("%s%d fine structure: %d states over %.3g µeV; focused: state %d (click or Tab)",
                        x.sector, x.level, length(st), spread, m.focus[])
    for (k, v) in m.res.extra
        startswith(k, "effective/") || continue
        (v isa NamedTuple && hasproperty(v, :of) && v.of == x.sector && collect(v.states) == st) || continue
        f = v.zero_field
        f === nothing && continue
        for (lab, c) in zip(f.labels, f.c)
            mm = match(r"^1 on (\S+)$", lab)
            mm === nothing && continue
            col = findfirst(y -> y.label == mm.captures[1], parts)
            col === nothing && continue
            hlines!(ax, [1e6 * c]; xmin = 0, xmax = 1, color = (_MUTED, 0.6), linestyle = :dash)
        end
        ax.title = ax.title[] * @sprintf("; dashed: the symmetric fit (residual %.3g µeV)", 1e6 * f.residual)
    end
    xlims!(ax, 0.4, length(parts) + 0.6)
    spread > 0 ? ylims!(ax, -0.15spread, 1.2spread) : ylims!(ax, -1, 1.5)
    return nothing
end

function _state_text(e::Explorer, i)
    m = e.model
    x = m.nodes[m.selection[]]
    s = m.res.sectors[x.sector]
    lab = isempty(s.state_labels) || isempty(s.state_labels[i]) ? "?" : s.state_labels[i]
    return @sprintf("%s state %d   %s\n%.6f eV", x.sector, i, lab, s.energies[i])
end

# a layout cell cleared of everything drawn in it
function _clear!(gl::GridLayout)
    for c in copy(contents(gl))
        if c isa GridLayout
            _clear!(c)
            gc = Makie.GridLayoutBase.gridcontent(c)
            gc === nothing || Makie.GridLayoutBase.remove_from_gridlayout!(gc)
        else
            delete!(c)
        end
    end
    return gl
end

_note(pos, msg) = Label(pos, msg; color = _MUTED, fontsize = 12, tellwidth = false,
                        tellheight = false, word_wrap = true, valign = :top)

function _card_text(e::Explorer, id)
    m = e.model
    x = m.nodes[id]
    s = m.res.sectors[x.sector]
    av = availability(m, id)
    lines = [node_name(m, id),
             @sprintf("%s %s, level %d (%s), states %s", x.sector, string(x.charge), x.level,
                      isempty(s.labels) ? "?" : s.labels[x.level], join(x.states, ", ")),
             @sprintf("E = %.6f eV", x.E),
             "focused: " * replace(_state_text(e, m.focus[]), "\n" => ", ")]
    isempty(s.closure) || push!(lines, @sprintf("closure %.2g%s", s.closure[x.level],
                                               !isempty(s.kramers) && s.kramers[x.level] ? ", Kramers" : ""))
    t = m.total[id]
    isfinite(t) && t > 0 && push!(lines, "radiative " * _fmt_tau(1 / t))
    for (k, v) in pairs(av)
        v === true || push!(lines, "no $(k): $(v)")
    end
    return join(lines, "\n")
end

function _draw_detail!(e::Explorer)
    host = e.state[:detail]
    _clear!(host)
    gl = GridLayout(host[1, 1])
    m = e.model
    id = m.selection[]
    x = m.nodes[id]
    Label(gl[1, 1], _card_text(e, id); justification = :left, halign = :left, fontsize = 12,
          tellwidth = false, color = _INK)
    lens = e.lens[]
    if lens == "recombination"
        for (row, ks, which, other) in ((2, m.out[id], "lines out", l -> l.to),
                                        (3, m.into[id], "lines in", l -> l.from))
            if isempty(ks)
                _note(gl[row, 1], "$which: none stored")
                continue
            end
            ax = Axis(gl[row, 1]; title = which, xlabel = "photon energy (eV)",
                      ylabel = "rate (ns⁻¹)", yscale = log10)
            ls = [m.lines[k] for k in ks]
            top = maximum(l.rate for l in ls)
            floor_ = max(top * 1e-7, minimum(l.rate for l in ls) / 2)
            pts = reduce(vcat, [[Point2f(l.ħω, floor_), Point2f(l.ħω, l.rate)] for l in ls])
            c = [_POL_GRAD[clamp(l.axial, 0, 1)] for l in ls]
            linesegments!(ax, pts; color = repeat(c; inner = 2), linewidth = 2.5)
            scatter!(ax, [Point2f(l.ħω, l.rate) for l in ls]; color = c, markersize = 7)
            for l in ls[1:min(3, end)]
                text!(ax, l.ħω, l.rate; text = " " * node_name(m, other(l)), fontsize = 9,
                      align = (:left, :bottom), color = _INK)
            end
            ylims!(ax, floor_, top * 5)
            # one line, or lines within µeV, still get an axis: at least 20 meV wide
            lo, hi = extrema(l.ħω for l in ls)
            pad = max(0.08 * (hi - lo), 0.01)
            xlims!(ax, lo - pad, hi + 3pad)
            e.state[Symbol(which)] = (axis = ax, lines = ls, other = other)
        end
    elseif lens == "symmetry"
        Label(gl[2, 1], _symmetry_text(e, id); color = _INK, fontsize = 12, tellwidth = false,
              tellheight = false, justification = :left, halign = :left, valign = :top)
    else
        try
            draw_configurations!(gl[2, 1], m.res, x.sector, m.focus[])
        catch err
            err isa ArgumentError || rethrow()
            _note(gl[2, 1], sprint(showerror, err))
        end
    end
    rowsize!(gl, 1, Auto())
    return nothing
end

function _symmetry_text(e::Explorer, id)
    m = e.model
    x = m.nodes[id]
    s = m.res.sectors[x.sector]
    fr = _stored_frame(m.res)
    fr === nothing && return "The archive holds no symmetry frame: run with `symmetry:` in the spec."
    out = ["$(x.label) in $(fr.group)" * (isempty(s.labels) ? "" : ", in the level $(s.labels[x.level])")]
    high = collect(String, m.res.orbitals.table.labels)
    c = _label_content(x.label, high)
    if c !== nothing
        hf = _frame_of(fr, fr.group)
        for g in _map_groups(m.res)[2:end]
            S = subduction(hf, _frame_of(fr, g))
            low = vec(c' * S.table)
            parts = join(((v > 1 ? "$v" : "") * S.low.table.labels[b] for (b, v) in pairs(low) if v > 0), " ⊕ ")
            push!(out, "  in $g: $parts" * (count(>(0), low) > 1 || any(>(1), low) ? "   (splits)" : ""))
        end
    end
    isempty(s.closure) || push!(out, @sprintf("closure %.2g", s.closure[x.level]))
    for (k, v) in m.res.extra
        startswith(k, "effective/") || continue
        (v isa NamedTuple && hasproperty(v, :of) && v.of == x.sector && x.states ⊆ collect(v.states)) || continue
        push!(out, "effective Hamiltonian $(k[11:end]) (states $(v.states)):")
        for t in v.terms
            push!(out, @sprintf("  %-10s %d invariant(s)", t.field, t.count))
        end
        f = v.zero_field
        f === nothing || push!(out, @sprintf("  zero-field residual %.3g µeV (floor %.2g): what only the breaking explains",
                                             1e6 * f.residual, 1e6 * f.floor))
    end
    return join(out, "\n")
end

# The occupation of each active orbital in a node, from its CI vectors and averaged over
# its states: `e[k]` for electron mode k, `h[k]` the probability that hole mode k is
# empty (a hole). Nothing without the vectors.
_occupations(m::ExplorerModel, id::Integer) =
    _occupations(m, m.nodes[id].sector, m.nodes[id].states)

function _occupations(m::ExplorerModel, sector::AbstractString, states::AbstractVector{<:Integer})
    get!(m.cache, (:occupations, sector, collect(states))) do
        s = m.res.sectors[sector]
        last(states) <= s.nkept || return nothing
        V = get!(() -> _val(s.vectors), m.cache, (:vectors, sector))
        ne, nh = length(s.estates), length(s.hstates)
        ne_, nh_ = zeros(ne), zeros(nh)
        for i in states, (r, b) in enumerate(s.basis)
            w = abs2(V[r, i]) / length(states)
            w > 0 || continue
            for k in 1:ne
                (b >> (k - 1)) & 1 == 1 && (ne_[k] += w)
            end
            for k in 1:nh
                (b >> (ne + k - 1)) & 1 == 0 && (nh_[k] += w)
            end
        end
        (; e = ne_, h = nh_)
    end
end

# Σₖ nₖ |ψₖ|² per species, from the occupations and the render tier's |ψ|²: the diagonal
# of the reduced density matrix in the orbitals. Exact for one carrier; for a many-body
# state it lacks the interference between configurations. Nothing without the data.
_occupation_densities(m::ExplorerModel, id::Integer) =
    _occupation_densities(m, m.nodes[id].sector, m.nodes[id].states)

function _occupation_densities(m::ExplorerModel, sector::AbstractString, states)
    ro = get(m.res.extra, "render/orbitals", nothing)
    n = _occupations(m, sector, states)
    (ro === nothing || n === nothing) && return nothing
    s = m.res.sectors[sector]
    T = get!(() -> _val(ro.total), m.cache, :render_total)
    dens(st, occ) = (any(>(1e-6), occ) ?
        sum(occ[k] .* Float64.(T[:, :, :, st[k].orb]) for k in eachindex(occ) if occ[k] > 1e-6) : nothing)
    return (; e = dens(s.estates, n.e), h = dens(s.hstates, n.h), coords = ro.coords,
            ro = ro)
end

# the orbital menu, labelled by each orbital's occupation in the selection, and set to
# the most occupied orbital
function _follow_orbitals!(e::Explorer)
    menu = e.state[:orbital_menu]
    ro = get(e.model.res.extra, "render/orbitals", nothing)
    ro === nothing && return
    m = e.model
    x = m.nodes[m.selection[]]
    s = m.res.sectors[x.sector]
    n = _occupations(m, x.sector, [m.focus[]])
    occ = zeros(length(ro.energies))
    if n !== nothing
        for (k, st) in enumerate(s.estates)
            occ[st.orb] = n.e[k]
        end
        for (k, st) in enumerate(s.hstates)
            occ[st.orb] = n.h[k]
        end
    end
    opts = [@sprintf("%d %s %.4f eV%s", k, ro.kind[k] === :e ? "e" : "h", ro.energies[k],
                     occ[k] > 5e-3 ? @sprintf("   n %.2f", occ[k]) : "")
            for k in eachindex(ro.energies)]
    best = any(>(5e-3), occ) ? argmax(occ) : e.state[:orbital][]
    e.state[:following] = true
    menu.options[] = opts
    menu.i_selected[] = best
    e.state[:following] = false
    e.state[:orbital][] = best
    return nothing
end

# the 3-D window: the selection's stored densities (else its occupation densities), or
# an orbital (by default the selection's most occupied)
function _draw_3d!(e::Explorer)
    host = e.state[:host3d]
    old = get(e.state, :axis3d, nothing)
    if old !== nothing
        e.state[:camera] = (old.azimuth[], old.elevation[])
    end
    _clear!(host)
    e.state[:axis3d] = nothing
    m = e.model
    x = m.nodes[m.selection[]]
    s = m.res.sectors[x.sector]
    frac = e.state[:fraction][]
    geom = e.state[:geometry][]
    p = if e.state[:mode3d][] == "densities"
        i = m.focus[]
        stored = haskey(s.densities, i)
        od = stored ? nothing : _occupation_densities(m, x.sector, [i])
        if stored
            draw_densities!(host[1, 1], m.res, x.sector, i; fraction = frac, geometry = geom)
        elseif od !== nothing && (od.e !== nothing || od.h !== nothing)
            gl = GridLayout(host[1, 1])
            ax = _axis3_titled(gl, _state_title(s, i) * "   (from occupations)")
            geom && od.ro.geometry !== nothing && _geometry!(ax, Float64.(_val(od.ro.geometry)), od.coords)
            _density_legend!(gl[3, 1], _density_pair!(ax, od.e, od.h, od.coords, frac))
            one_carrier = sum(x.charge) == 1
            Label(gl[4, 1], one_carrier ? "one carrier: Σₖ nₖ|ψₖ|² is its density" :
                  "Σₖ nₖ|ψₖ|² over the orbitals, nₖ from the CI vector: it lacks the interference " *
                  "between configurations. Exact: " * string(availability(m, x.id).densities);
                  fontsize = 10, color = _MUTED, tellwidth = false, word_wrap = true)
            (; axis = ax)
        else
            _note(host[1, 1], "nothing to draw for $(node_name(m, x.id)): " *
                              string(availability(m, x.id).densities) *
                              (availability(m, x.id).vectors === true ? "" :
                               "; or " * string(availability(m, x.id).vectors)))
            nothing
        end
    else
        k = e.state[:orbital][]
        av = availability(m, x.id).orbitals
        av === true ? draw_orbital!(host[1, 1], m.res, k; fraction = frac, geometry = geom) :
                      (_note(host[1, 1], "no render tier: " * string(av)); nothing)
    end
    if p !== nothing
        e.state[:axis3d] = p.axis
        cam = get(e.state, :camera, nothing)
        if cam !== nothing
            p.axis.azimuth[] = cam[1]
            p.axis.elevation[] = cam[2]
        end
    end
    return nothing
end

# ── interaction ─────────────────────────────────────────────────────────────

# the node nearest a point of the map, in data coordinates; nothing when none is close
function _node_at(e::Explorer, xy)
    segs = e.segments[]
    lim = e.map_axis.finallimits[]
    tol = 0.03 * lim.widths[2]
    best, d = nothing, Inf
    for s in segs
        abs(xy[1] - s.x) <= 0.45 || continue
        dy = abs(xy[2] - s.y)
        dy < d && ((best, d) = (s.id, dy))
    end
    return d <= tol ? best : nothing
end

"Focus the state at position `xy` of the fine-structure strip; what a click there does."
function _click_fine!(e::Explorer, xy)
    best, d = nothing, Inf
    lim = e.fine_axis.finallimits[]
    for f in get(e.state, :fine, NamedTuple[])
        abs(xy[1] - f.x) <= f.w / 2 || continue
        dy = abs(xy[2] - f.y)
        dy < d && ((best, d) = (f.state, dy))
    end
    (best === nothing || d > 0.08 * lim.widths[2]) && return nothing
    focus!(e.model, best)
    return best
end

"Select the node at map position `xy` (data coordinates); what a click does."
function _click_map!(e::Explorer, xy)
    id = _node_at(e, xy)
    id === nothing || select!(e.model, id)
    return id
end

# a click on a line in the detail panel selects the level at its other end
function _click_lines!(e::Explorer, which::Symbol, xy)
    d = get(e.state, which, nothing)
    d === nothing && return nothing
    isempty(d.lines) && return nothing
    k = argmin([abs(l.ħω - xy[1]) for l in d.lines])
    abs(d.lines[k].ħω - xy[1]) <= 0.02 * d.axis.finallimits[].widths[1] || return nothing
    id = d.other(d.lines[k])
    select!(e.model, id)
    return id
end

const _KEYS = Dict(Keyboard.up => :up, Keyboard.down => :down, Keyboard.left => :left,
                   Keyboard.right => :right, Keyboard.page_up => :next,
                   Keyboard.page_down => :previous)

"What a key does: arrows and Page Up/Down move, Enter follows the strongest line,
Backspace goes back, 1–3 switch the lens."
function _key!(e::Explorer, key)
    m = e.model
    if haskey(_KEYS, key)
        step!(m, _KEYS[key])
    elseif key == Keyboard.enter
        follow!(m)
    elseif key == Keyboard.tab
        st = level_states(m)
        k = findfirst(==(m.focus[]), st)
        focus!(m, st[mod1(something(k, 0) + 1, length(st))])
    elseif key == Keyboard.backspace
        back!(m)
    elseif key in (Keyboard._1, Keyboard._2, Keyboard._3)
        e.lens[] = _LENSES[Int(key) - Int(Keyboard._1) + 1]
    else
        return false
    end
    return true
end

"""
    explore(res::CICalcResults; lens = "recombination", nlevels = 40, display = true)
        -> Explorer

The explorer of a results archive: two windows that follow one selected level.

**The map window.**
- The **map** is a Grotrian diagram: sectors side by side in charge order, one column
  per irrep, each level at its energy above the sector's ground (meV). Click a level
  to select it, or hover for its name and lifetime.
- The **lens** sets what the map shows:
  - `recombination`: lines coloured by the axial share of their emission and as wide
    as their total rate; the selection's lines out (solid) and in (dashed); grey halos
    where its decay goes, orange where its feeders' decay comes through it;
  - `symmetry`: lines coloured by irrep. The **group** menu redraws the map in a
    subgroup, so a level that the lowering splits sits in several columns;
  - `composition`: lines coloured by how much of the level is one configuration of
    orbital levels, electrons ⊗ holes (white: mixed, deep blue: one configuration;
    grey without its vectors).
- The **fine-structure** strip shows every state of the selected level (the sector's
  degenerate block) in µeV, one column per irrep, with the symmetric fit when the
  archive's effective-Hamiltonian product covers it. Click a state, or press Tab, to
  focus it: the card, the make-up and the 3-D window then show that state.
- The **detail** panel: the level's card (energy, label, closure, lifetime, and what
  the archive lacks with the spec key that adds it), then per lens its emission and
  absorption lines (click one to go to its other end), its labels in every subgroup
  and its effective Hamiltonian, or its make-up.
- **Keys:** arrows move along and across columns, Page Up/Down across sectors, Enter
  follows the strongest line down, Tab steps through the level's states, Backspace
  goes back, 1–3 switch the lens. The
  **find** menu is searchable ("XX 3", "Γ6"); ◀ ▶ walk the history.

**The 3-D window** follows the selection too. `densities` draws its stored densities, or,
where none are stored, Σₖ nₖ|ψₖ|² from its CI vector's orbital occupations and the render
tier (exact for one carrier; for more it lacks the interference between configurations,
and says so). `orbital` lists every orbital with its occupation in the selection and
shows the most occupied, or the one picked. The enclosed fraction and the dot's
outline are adjustable.

Under GLMakie both windows open (`display = false` builds them without showing). Under
CairoMakie they build, respond to `select!(e.model, id)` and save, but take no input.
"""
function explore(res::CICalcResults; lens::AbstractString = "recombination",
                 nlevels::Integer = 40, display::Bool = true, size = (1650, 980),
                 size3d = (760, 720))
    lens in _LENSES || throw(ArgumentError("lens must be one of $(join(_LENSES, ", "))"))
    m = explorer_model(res; nlevels)
    groups = _map_groups(res)
    fig = Figure(size = size)
    bar = GridLayout(fig[1, 1:2]; tellwidth = false)
    Label(bar[1, 1], "lens"; halign = :right)
    lens_menu = Menu(bar[1, 2]; options = _LENSES, default = lens, width = 150)
    Label(bar[1, 3], "group"; halign = :right)
    group_menu = Menu(bar[1, 4]; options = groups, default = first(groups), width = 90)
    Label(bar[1, 5], "find"; halign = :right)
    names = [node_name(m, x.id) for x in m.nodes]
    find_menu = Menu(bar[1, 6]; options = names, default = nothing, width = 260,
                     prompt = "find a level…")
    back_b = Button(bar[1, 7]; label = "◀")
    fwd_b = Button(bar[1, 8]; label = "▶")
    sel_label = Label(bar[1, 9], node_name(m, m.selection[]); font = :bold, halign = :left,
                      tellwidth = false)
    colgap!(bar, 8)

    map_ax = Axis(fig[2, 1]; ylabel = "E above the sector's ground (meV)", xgridvisible = false,
                  ygridcolor = _GRID, xticklabelrotation = π / 2, xticklabelsize = 10)
    hidespines!(map_ax, :t, :r)
    fine_ax = Axis(fig[3, 1]; ylabel = "µeV", xgridvisible = false, ygridcolor = _GRID,
                   titlesize = 12, titlefont = :regular)
    detail = GridLayout(fig[2:3, 2])
    rowsize!(fig.layout, 3, Relative(0.24))
    colsize!(fig.layout, 1, Relative(0.64))

    fig3d = Figure(size = size3d)
    bar3 = GridLayout(fig3d[1, 1]; tellwidth = false)
    Label(bar3[1, 1], "show"; halign = :right)
    mode_menu = Menu(bar3[1, 2]; options = ["densities", "orbital"], default = "densities", width = 110)
    ro = get(res.extra, "render/orbitals", nothing)
    orbs = ro === nothing ? ["—"] :
           [@sprintf("%d %s %.4f eV", k, ro.kind[k] === :e ? "e" : "h", ro.energies[k])
            for k in eachindex(ro.energies)]
    orb_menu = Menu(bar3[1, 3]; options = orbs, default = first(orbs), width = 170)
    Label(bar3[1, 4], "fraction"; halign = :right)
    frac = Slider(bar3[1, 5]; range = 0.1:0.05:0.9, startvalue = 0.5, width = 120)
    geo_toggle = Toggle(bar3[1, 6]; active = true)
    Label(bar3[1, 7], "dot"; halign = :left)
    host3d = GridLayout(fig3d[2, 1])

    e = Explorer(fig, fig3d, m, Observable(lens), Observable(first(groups)), map_ax, fine_ax,
                 Ref(NamedTuple[]),
                 Dict{Symbol,Any}(:detail => detail, :host3d => host3d, :fraction => frac.value,
                                  :geometry => geo_toggle.active, :mode3d => Observable("densities"),
                                  :orbital => Observable(1), :orbital_menu => orb_menu,
                                  :following => false, :menus => (lens_menu, group_menu, find_menu)))
    redraw() = (_draw_map!(e); _draw_fine!(e); _draw_detail!(e); _follow_orbitals!(e); _draw_3d!(e);
                sel_label.text = node_name(m, m.selection[]))
    _draw_map!(e; keep_limits = false)
    _draw_fine!(e)
    _draw_detail!(e)
    _follow_orbitals!(e)
    _draw_3d!(e)

    on(_ -> redraw(), m.selection)
    on(m.focus) do _                    # another state of the same line: no new map
        _draw_fine!(e); _draw_detail!(e); _follow_orbitals!(e); _draw_3d!(e)
    end
    on(e.lens) do l
        lens_menu.selection[] == l || (lens_menu.i_selected[] = findfirst(==(l), _LENSES))
        _draw_map!(e); _draw_detail!(e)
    end
    on(e.group) do g
        group_menu.selection[] == g || (group_menu.i_selected[] = findfirst(==(g), groups))
        _draw_map!(e; keep_limits = false); _draw_fine!(e)
    end
    on(s -> s === nothing || s == e.lens[] || (e.lens[] = s), lens_menu.selection)
    on(s -> s === nothing || s == e.group[] || (e.group[] = s), group_menu.selection)
    on(find_menu.selection) do s
        s === nothing && return
        k = findfirst(==(s), names)
        k === nothing || select!(m, k)
    end
    on(_ -> back!(m), back_b.clicks)
    on(_ -> forward!(m), fwd_b.clicks)
    on(s -> s === nothing || (e.state[:mode3d][] = s; _draw_3d!(e)), mode_menu.selection)
    on(orb_menu.selection) do s
        (s === nothing || s == "—" || e.state[:following]) && return
        e.state[:orbital][] = parse(Int, first(split(s)))
        e.state[:mode3d][] == "orbital" && _draw_3d!(e)
    end
    on(_ -> _draw_3d!(e), Makie.Observables.throttle(0.2, frac.value))
    on(_ -> _draw_3d!(e), geo_toggle.active)

    # clicks on the map and on the lines of the detail panel; keys anywhere but a menu
    on(events(fig).mousebutton; priority = 2) do ev
        (ev.button == Mouse.left && ev.action == Mouse.press) || return Consume(false)
        if is_mouseinside(map_ax)
            _click_map!(e, mouseposition(map_ax.scene))
        elseif is_mouseinside(fine_ax)
            _click_fine!(e, mouseposition(fine_ax.scene))
        else
            for w in (Symbol("lines out"), Symbol("lines in"))
                d = get(e.state, w, nothing)
                (d !== nothing && is_mouseinside(d.axis)) || continue
                _click_lines!(e, w, mouseposition(d.axis.scene))
            end
        end
        return Consume(false)
    end
    on(events(fig).keyboardbutton) do ev
        ev.action in (Keyboard.press, Keyboard.repeat) || return Consume(false)
        any(mn -> mn.is_open[], (lens_menu, group_menu, find_menu)) && return Consume(false)
        return Consume(_key!(e, ev.key))
    end
    try
        DataInspector(fig)
    catch
    end
    if display
        be = Makie.current_backend()
        if be !== missing && nameof(be) == :GLMakie
            Base.display(be.Screen(; title = "CICalc explorer"), fig)
            Base.display(be.Screen(; title = "CICalc explorer: 3-D"), fig3d)
        else
            @info "explore: the windows need GLMakie (`using GLMakie`); built them without showing"
        end
    end
    return e
end
