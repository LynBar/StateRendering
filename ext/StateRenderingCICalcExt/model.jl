# ---------------------------------------------------------------------------
# The explorer's model: the state structure of an archive as a graph, and the
# selection on it. No drawing here, so it is tested without a backend.
#
# The nodes are the lines of the by-irrep spectra (`_sector_lines`): a level, or the
# part of a merged level in one irrep, of one sector. Every node has a physical place,
# its sector, its irrep column and its energy, so the map is a Grotrian diagram and
# needs no graph layout. The edges are the stored radiative lines between sectors.
# ---------------------------------------------------------------------------

struct LevelNode
    id::Int
    sector::String                 # the archive's name for the sector
    charge::Tuple{Int,Int}         # (Nₑ, Nₕ)
    level::Int                     # index into the sector's levels
    label::String                  # this line's irrep, "?" without one
    states::Vector{Int}
    E::Float64                     # mean energy of the states, eV
    dE::Float64                    # above the sector's ground state, meV
    column::Int                    # irrep column within the sector
end

struct LineEdge
    from::Int                      # node ids, initial and final
    to::Int
    rate::Float64                  # ns⁻¹: summed over the final states, averaged over the initial
    ħω::Float64                    # eV
    axial::Float64                 # the axial share of the emission
end

mutable struct ExplorerModel
    res::CICalcResults
    nodes::Vector{LevelNode}
    sectors::Vector{String}                # in charge order
    columns::Dict{String,Vector{String}}   # each sector's irrep columns
    lines::Vector{LineEdge}
    out::Vector{Vector{Int}}               # per node: its lines out, strongest first
    into::Vector{Vector{Int}}              # per node: its lines in, strongest first
    total::Vector{Float64}                 # per node: its total rate out, ns⁻¹ (NaN: not stored)
    selection::Observable{Int}
    focus::Observable{Int}                 # one state of the selection's level, a sector index
    history::Vector{Int}
    cursor::Int
    cache::Dict{Any,Any}                   # heavy reads and derived data, by key
end

Base.show(io::IO, m::ExplorerModel) =
    print(io, "ExplorerModel(", length(m.nodes), " levels in ", join(m.sectors, " "), "; ",
          length(m.lines), " lines; selected ", node_name(m, m.selection[]), ")")

"""
    explorer_model(res::CICalcResults; nlevels = 40, min_branch = 1e-6) -> ExplorerModel

The state structure of an archive as a graph, for the explorer. Sectors are ordered by net
charge (0 X XX, then h X⁺, then e X⁻, …), so every radiative line joins neighbours. The
nodes are the lines
of each sector's by-irrep spectrum (its lowest `nlevels` levels): a level, or the part
of a merged level in one irrep. The edges are the radiative lines the archive stores,
from a level to each level of the next sector down that takes at least `min_branch` of
its decay. The selection (a node id) and its history start at the exciton's ground
level (else the first sector with more than one level).
"""
function explorer_model(res::CICalcResults; nlevels::Integer = 40, min_branch::Real = 1e-6)
    # grouped by net charge q = Nₕ − Nₑ (0 X XX | h X⁺ | e X⁻ | …), each group by size:
    # a radiative line removes one e–h pair, so it always joins neighbours in a group
    q(n) = res.sectors[n].sector[2] - res.sectors[n].sector[1]
    names = sort!(collect(keys(res.sectors)); by = n -> (abs(q(n)), q(n) < 0,
                                                       sum(res.sectors[n].sector), n))
    nodes = LevelNode[]
    columns = Dict{String,Vector{String}}()
    for n in names
        s = res.sectors[n]
        d = _sector_lines(res, s; nlevels, by_irrep = true)
        columns[n] = d.cols
        for (k, (lab, st)) in enumerate(d.lines_)
            j = findfirst(l -> first(st) in l, s.levels)
            push!(nodes, LevelNode(length(nodes) + 1, n, s.sector, j, lab, st,
                                   sum(s.energies[st]) / length(st), d.y[k],
                                   findfirst(==(lab), d.cols)))
        end
    end
    lines, total = _radiative_edges(res, nodes, min_branch)
    out = [Int[] for _ in nodes]
    into = [Int[] for _ in nodes]
    for (k, e) in enumerate(lines)
        push!(out[e.from], k)
        push!(into[e.to], k)
    end
    foreach(v -> sort!(v; by = k -> -lines[k].rate), out)
    foreach(v -> sort!(v; by = k -> -lines[k].rate), into)
    # start at the exciton's ground level, else the first sector with more than one
    start = something(findfirst(n -> res.sectors[n].sector == (1, 1), names),
                      findfirst(n -> length(res.sectors[n].levels) > 1, names), 1)
    first_node = something(findfirst(x -> x.sector == names[start], nodes), 1)
    return ExplorerModel(res, nodes, names, columns, lines, out, into, total,
                         Observable(first_node), Observable(first(nodes[first_node].states)),
                         [first_node], 1, Dict{Any,Any}())
end

# the stored lines between nodes: for each rates product, every initial node to every
# final node, the rate averaged over the initial node's states
function _radiative_edges(res, nodes, min_branch)
    lines = LineEdge[]
    total = fill(NaN, length(nodes))
    by_sector = Dict{Tuple{Int,Int},Vector{Int}}()
    for x in nodes
        push!(get!(by_sector, x.charge, Int[]), x.id)
    end
    for k in sort!(collect(keys(res.rates)))
        r = res.rates[k]
        (haskey(by_sector, r.initial) && haskey(by_sector, r.final)) || continue
        nf, ni = size(r.Γ)
        for a in by_sector[r.initial]
            A = nodes[a].states
            all(<=(ni), A) || continue
            tot = sum(sum(r.Γ[:, i]) for i in A) / length(A)
            total[a] = tot
            tot > 0 || continue
            for b in by_sector[r.final]
                B = nodes[b].states
                all(<=(nf), B) || continue
                Γ = sum(r.Γ[f, i] for f in B, i in A) / length(A)
                Γ >= min_branch * tot || continue
                ax = sum(r.Γpol[3][f, i] for f in B, i in A) / length(A) / Γ
                push!(lines, LineEdge(a, b, Γ, r.ħω[first(B), first(A)], ax))
            end
        end
    end
    return lines, total
end

"""
    node_name(m, id) -> String

A node as text: sector, level, irrep and energy above the sector's ground, e.g.
`"X1 Γ6 (+0.0 meV)"`. A sector named by a number (the vacuum, `"0"`) is set apart
from its level: `"0 #1 Γ1"`.
"""
function node_name(m::ExplorerModel, id::Integer)
    x = m.nodes[id]
    sep = isdigit(last(x.sector)) ? " #" : ""
    return @sprintf("%s%s%d %s (%+.1f meV)", x.sector, sep, x.level, x.label, x.dE)
end

# ── selection and history ───────────────────────────────────────────────────

"""
    select!(m, id; state = first state of the node) -> m

Select node `id`, and with it the focused state (`m.focus`, a state index of the
sector), which must be one of the node's. The history keeps the path, like a browser:
selecting after going back drops what was ahead.
"""
function select!(m::ExplorerModel, id::Integer; state = nothing)
    1 <= id <= length(m.nodes) || throw(ArgumentError("no node $id: the model has $(length(m.nodes))"))
    st = something(state, first(m.nodes[id].states))
    st in m.nodes[id].states || throw(ArgumentError(
        "state $st is not one of $(node_name(m, id))'s ($(join(m.nodes[id].states, ", ")))"))
    # an odd sector focuses a Kramers pair, by its first state when that is in the node
    lead = first(pair_of(m, m.nodes[id].sector, st))
    lead in m.nodes[id].states && (st = lead)
    if id == m.selection[]
        st == m.focus[] || (m.focus[] = st)
        return m
    end
    resize!(m.history, m.cursor)
    push!(m.history, id)
    m.cursor = length(m.history)
    m.focus.val = st                    # set quietly: the selection's listeners draw both
    m.selection[] = id
    return m
end

"""
    focus!(m, i) -> m

Focus state `i` (a state index of the selected node's sector) of the selected level:
one of its own states, or one of another irrep's part of the same level, which then
becomes the selection.
"""
function focus!(m::ExplorerModel, i::Integer)
    x = m.nodes[m.selection[]]
    i = first(pair_of(m, x.sector, i))         # a Kramers pair is focused by its first state
    k = findfirst(y -> y.sector == x.sector && y.level == x.level && i in y.states, m.nodes)
    k === nothing && throw(ArgumentError(
        "state $i is not in the level of $(node_name(m, x.id))"))
    return select!(m, k; state = i)
end

# the states of the selection's level, in energy order; in an odd sector one per Kramers
# pair (its first), the focus steps through pairs
level_states(m::ExplorerModel) =
    (x = m.nodes[m.selection[]];
     unique!(sort!([first(pair_of(m, x.sector, i))
                    for y in m.nodes if y.sector == x.sector && y.level == x.level
                    for i in y.states])))

# ── Kramers pairs ───────────────────────────────────────────────────────────
# A sector with an odd number of fermions has Kramers-degenerate states, T ψ and ψ, which
# agree in everything spin-blind (densities, make-up by orbital pair, natural
# occupations): the explorer treats each pair as one state. An even sector (X, XX) has no
# such pairs, and its states within a level are distinct physics (dark, bright).

_odd(s) = isodd(sum(s.sector))

# state ⇒ its Kramers pair, within each level of an odd sector, in energy order (of
# anything with `sector`, `levels` and `energies`: a StoredSector)
function _kramers_map(s)
    d = Dict{Int,Vector{Int}}()
    _odd(s) || return d
    for l in s.levels
        for p in kramers_pairs(s.energies[l]; tol = Inf)   # the level is one block
            pr = collect(l)[p]
            foreach(i -> d[i] = pr, pr)
        end
    end
    return d
end

_state_pairs(m::ExplorerModel, sector) =
    get!(() -> _kramers_map(m.res.sectors[sector]), m.cache, (:kramers, sector))

"""
    pair_of(m, sector, i) -> Vector{Int}

State `i`'s Kramers pair in an odd sector (both states, in index order), or `[i]`.
"""
pair_of(m::ExplorerModel, sector, i::Integer) = get(_state_pairs(m, sector), i, [i])

"The focused state's Kramers pair (odd sectors), or the focused state alone."
focus_states(m::ExplorerModel) = pair_of(m, m.nodes[m.selection[]].sector, m.focus[])

"`back!(m)`, `forward!(m)`: move along the selection history."
function back!(m::ExplorerModel)
    m.cursor > 1 || return m
    m.cursor -= 1
    m.focus.val = first(m.nodes[m.history[m.cursor]].states)
    m.selection[] = m.history[m.cursor]
    return m
end
function forward!(m::ExplorerModel)
    m.cursor < length(m.history) || return m
    m.cursor += 1
    m.focus.val = first(m.nodes[m.history[m.cursor]].states)
    m.selection[] = m.history[m.cursor]
    return m
end

"""
    step!(m, direction) -> m

Move the selection on the map. `:up`/`:down`: the next line up or down in the same
irrep column. `:left`/`:right`: the nearest line in energy in the neighbouring column
of the same sector. `:next`/`:previous`: the ground level of the neighbouring sector.
"""
function step!(m::ExplorerModel, dir::Symbol)
    x = m.nodes[m.selection[]]
    same = [y for y in m.nodes if y.sector == x.sector]
    target = if dir in (:up, :down)
        col = sort([y for y in same if y.column == x.column]; by = y -> (y.dE, y.id))
        k = findfirst(y -> y.id == x.id, col) + (dir === :up ? 1 : -1)
        1 <= k <= length(col) ? col[k].id : nothing
    elseif dir in (:left, :right)
        c = x.column + (dir === :right ? 1 : -1)
        col = [y for y in same if y.column == c]
        isempty(col) ? nothing : col[argmin([abs(y.dE - x.dE) for y in col])].id
    elseif dir in (:next, :previous)
        k = findfirst(==(x.sector), m.sectors) + (dir === :next ? 1 : -1)
        1 <= k <= length(m.sectors) ?
            findfirst(y -> y.sector == m.sectors[k], m.nodes) : nothing
    else
        throw(ArgumentError("direction must be :up, :down, :left, :right, :next or :previous"))
    end
    target === nothing || select!(m, target)
    return m
end

"""
    follow!(m) -> m

Select the final level of the strongest line out of the selection (the brightest step
down the cascade). Does nothing when the archive stores no line out of it.
"""
function follow!(m::ExplorerModel)
    o = m.out[m.selection[]]
    isempty(o) || select!(m, m.lines[first(o)].to)
    return m
end

"""
    find_levels(m, query) -> Vector{Int}

The nodes a query names. Words are matched in any order: a sector name (`"XX"`,
`"X-"`, the vacuum `"0"`), a level number, an irrep (`"Γ6"`). `"XX 3"` is every line
of XX's third level; `"Γ6"` every Γ6 line; `"X Γ6"` the Γ6 lines of X. Sector names
are matched exactly, so `"X"` is not `"XX"`.
"""
function find_levels(m::ExplorerModel, query::AbstractString)
    words = split(strip(query))
    # a word that names a sector is that sector (the vacuum is "0"), not a level number
    secs = [w for w in words if w in m.sectors]
    nums = [parse(Int, w) for w in words if !(w in secs) && all(isdigit, w)]
    labs = [w for w in words if !(w in secs) && !all(isdigit, w)]
    return [x.id for x in m.nodes if (isempty(secs) || x.sector in secs) &&
                                       (isempty(nums) || x.level in nums) &&
                                       (isempty(labs) || x.label in labs)]
end

# ── the cascade through a node ──────────────────────────────────────────────

"""
    decay_tree(m, id) -> Dict{Int,Float64}

Where node `id` decays to, radiatively: every node below it with the fraction of `id`'s
population that passes through it, the branching ratios multiplied along every path.
The fractions into each lower sector sum to 1, less what the stored rates do not reach.
No relaxation within a sector is assumed (as in `cascade_bundle`).
"""
function decay_tree(m::ExplorerModel, id::Integer)
    flow = Dict{Int,Float64}(id => 1.0)
    # sectors only go down in charge, so nodes in decreasing charge are an order
    order = sort([x.id for x in m.nodes if sum(x.charge) < sum(m.nodes[id].charge)];
                 by = k -> -sum(m.nodes[k].charge))
    for a in [id; order]
        f = get(flow, a, 0.0)
        (f > 0 && isfinite(m.total[a]) && m.total[a] > 0) || continue
        for k in m.out[a]
            e = m.lines[k]
            flow[e.to] = get(flow, e.to, 0.0) + f * e.rate / m.total[a]
        end
    end
    delete!(flow, id)
    return flow
end

"""
    feeders(m, id) -> Dict{Int,Float64}

The nodes above `id` whose decay passes through it, each with the fraction of its own
population that does.
"""
feeders(m::ExplorerModel, id::Integer) =
    Dict(a => f for a in eachindex(m.nodes)
         if sum(m.nodes[a].charge) > sum(m.nodes[id].charge)
         for f in (get(decay_tree(m, a), id, 0.0),) if f > 0)

# ── what the archive holds for a node ───────────────────────────────────────

"""
    availability(m, id) -> NamedTuple

What the archive holds for node `id`, and for what it lacks, the spec key that would
fill it: `vectors` (its CI vectors, `keep:`), `amplitudes` (the amplitude tier),
`densities` (stored, or from its vectors and the amplitudes), `rates` (lines out),
`orbitals` (the render tier). Each is `true` or the text to show instead.
"""
function availability(m::ExplorerModel, id::Integer)
    x = m.nodes[id]
    s = m.res.sectors[x.sector]
    vec = last(x.states) <= s.nkept ||
          "raise `keep:` of sector $(x.sector) to $(last(x.states)) or more"
    amp = _has_amplitudes(m.res) || "add `orbitals: {amplitudes: true}` to the run spec's `compute`"
    den = any(i -> haskey(s.densities, i), x.states) || (vec === true && amp === true) ||
          (vec === true ? amp * " (densities of every kept state)" : vec) *
          ", or add $(join(x.states, ", ")) to `densities:` of sector $(x.sector)"
    rat = isfinite(m.total[id]) || (sum(x.charge) == 0 ? "the vacuum emits nothing" :
          "add `rates:` from $(x.sector) to the run spec's `compute`")
    orb = haskey(m.res.extra, "render/orbitals") || "add `orbitals:` to the run spec's `compute`"
    return (; vectors = vec, amplitudes = amp, densities = den, rates = rat, orbitals = orb)
end

_has_amplitudes(res) = (ro = get(res.extra, "render/orbitals", nothing);
                        ro !== nothing && get(ro, :amplitudes, nothing) !== nothing)
