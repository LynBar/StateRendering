# ---------------------------------------------------------------------------
# The labelling figures (CICalc SYMMETRY_DECOMPOSITION_PLAN L6).
#
# Every figure but `plot_sweep` draws a `LevelBundle` (`level_bundle(ks, a; …)`),
# plain arrays flattened out of a `SymmetryAnalysis`. Each also draws an
# unlabelled bundle (`level_bundle(ks)`) with the same layout: grey, "?", and
# a note where a panel has nothing to show.
#
# Colour is the irrep (`irrep_color`, one colour per irrep per group in every
# figure), grey is "?", red outlines a declared label the measurement disagrees
# with. Text stays in ink; a coloured mark beside it carries the identity.
#
# Small numbers are the point of most of these figures (leakage 1e-6, closure
# 1e-15), so the stacked-to-1 panels that show composition are paired with a
# log panel that shows what the stack hides. Two measures of different scale go
# in two panels sharing an axis, never on twin y-axes.
# ---------------------------------------------------------------------------

const _INK = RGBf(0.15, 0.15, 0.15)
const _MUTED = RGBf(0.45, 0.45, 0.43)
const _GRID = RGBf(0.88, 0.88, 0.86)
const _SEQ = cgrad([RGBf(0xcd / 255, 0xe2 / 255, 0xfb / 255), RGBf(0x86 / 255, 0xb6 / 255, 0xef / 255),
                    RGBf(0x39 / 255, 0x87 / 255, 0xe5 / 255), RGBf(0x1c / 255, 0x5c / 255, 0xab / 255),
                    RGBf(0x0d / 255, 0x36 / 255, 0x6b / 255)])
const _LOGMIN = 1e-16

_icol(lb::LevelBundle, l) = irrep_color(l, lb.irreps, lb.double)
_icol(t, l) = irrep_color(l, t.labels, t.double)          # a CharacterTable
# species are :e/:h from a file, :electron/:hole from a solver
_ishole(k::Symbol) = k in (:h, :hole, :valence)
_kindname(k::Symbol) = _ishole(k) ? "holes" : "electrons"
_tag(lb::LevelBundle, i) = @sprintf("%s%d", _ishole(lb.kind[i]) ? "h" : "e", i)
# a figure title: `tellwidth = false`, or a one-column title sets the column's width
_title!(pos, s) = Label(pos, s; fontsize = 15, tellwidth = false)
_clip(x) = isfinite(x) ? max(x, _LOGMIN) : NaN
_sci(x) = !isfinite(x) ? "–" : x < 1e-14 ? "<1e-14" : @sprintf("%.0e", x)

# the block's dominant irrep (from its basis-independent weights), "?" without one
function _block_irrep(lb::LevelBundle, b)
    labelled(lb) || return "?"
    w = lb.block_weights[b]
    isempty(w) ? "?" : lb.irreps[argmax(w)]
end

# class names for a figure axis: the combining bar of `frame_classes` ("C6̄") is
# not drawn by every font, and two classes can share a name (σv, σd), so they are
# typeset in LaTeX with a bar and primed in order of appearance.
function _class_ticks(names::AbstractVector{<:AbstractString})
    seen = Dict{String,Int}()
    out = Any[]
    for n in names
        barred = endswith(n, "̄")
        base = barred ? String(chop(n)) : String(n)
        k = get(seen, n, 0)
        seen[n] = k + 1
        m = match(r"^([A-Za-zσ]+)(\d*)$", base)
        head, sub = m === nothing ? (base, "") : (m.captures[1], m.captures[2])
        head = replace(head, "σ" => "\\sigma")
        tex = barred ? "\\bar{" * head * "}" : head
        isempty(sub) || (tex *= "_{" * sub * "}")
        tex *= repeat("'", k)
        push!(out, latexstring(tex))
    end
    return out
end

# a state's own irrep (its dominant weight) where its block is labelled, "?" otherwise;
# a block such as Γ8 ⊕ Γ9 holds states of either
function _slabel(lb::LevelBundle, i)
    b = findfirst(bb -> i in bb, lb.blocks)
    (!labelled(lb) || lb.labels[b] == "?" || lb.dominant[i] == 0) && return lb.labels[b]
    return lb.irreps[lb.dominant[i]]
end
_scol(lb::LevelBundle, i) = labelled(lb) && lb.dominant[i] > 0 ?
                            _icol(lb, lb.irreps[lb.dominant[i]]) : UNLABELLED_COLOR

# the irreps a block is made of (its rounded content), else its dominant one: the
# weight outside them is the leakage
function _own_irreps(lb::LevelBundle, b)
    c = lb.content[b]
    own = isempty(c) ? Int[] : findall(x -> round(Int, x) > 0, c)
    return isempty(own) ? [argmax(lb.block_weights[b])] : own
end

# A level that is one irrep and its time-reversal partner, measured closed under T:
# C3v's Γ4 ⊕ Γ5 (Herring type c). One Kramers level, not two irreps by accident.
function _tr_pair(lb::LevelBundle, b)
    p = get(lb.metadata, :tr_partner, nothing)
    (p === nothing || !labelled(lb) || isempty(lb.content[b])) && return false
    own = findall(x -> round(Int, x) > 0, lb.content[b])
    length(own) == 2 && p[own[1]] == own[2] || return false
    return isfinite(lb.tr_closure[b]) && lb.tr_closure[b] <= lb.closure_tol
end

# the irreps a label such as "Γ8 ⊕ Γ9" or "2Γ4 ⊕ Γ5" names; empty for "?" or unknown
function _label_irreps(label::AbstractString, irreps)
    out = Int[]
    for part in split(label, '⊕')
        s = lstrip(strip(part), collect("0123456789"))
        k = findfirst(==(s), irreps)
        k === nothing && return Int[]
        push!(out, k)
    end
    return out
end

function _note!(ax, msg)
    text!(ax, 0.5, 0.5; text = msg, space = :relative, align = (:center, :center),
          color = _MUTED, fontsize = 12)
    return ax
end

# irreps present in some weights, for a legend in table order
function _present(irreps, ws; cut = 1e-3)
    isempty(ws) && return Int[]
    [k for k in eachindex(irreps) if any(w -> !isempty(w) && w[k] > cut, ws)]
end

function _irrep_legend!(pos, irreps, double, ks; title = "irrep", extra = (;))
    els = Any[PolyElement(color = irrep_color(irreps[k], irreps, double)) for k in ks]
    labs = String[irreps[k] for k in ks]
    for (l, e) in pairs(extra)
        push!(els, e); push!(labs, String(l))
    end
    isempty(els) && return nothing
    return Legend(pos, els, labs, title; framevisible = false, labelsize = 11,
                  titlesize = 11, rowgap = 2)
end

# 1-D label repel: positions ≥ `gap` apart, as close to `y` as that allows. Labels
# that would collide form a cluster, spread evenly about the cluster's own mean;
# clusters that then touch merge. A label far from the others stays on its value.
function _repel(y::AbstractVector{<:Real}, gap::Real)
    isempty(y) && return Float64[]
    p = sortperm(y)
    ys = Float64.(y[p])
    cl = [[k] for k in eachindex(ys)]                  # clusters of sorted indices
    centre(c) = sum(ys[c]) / length(c)
    lo(c) = centre(c) - (length(c) - 1) * gap / 2
    hi(c) = centre(c) + (length(c) - 1) * gap / 2
    merged = true
    while merged
        merged = false
        for k in 1:length(cl)-1
            if lo(cl[k+1]) - hi(cl[k]) < gap
                cl[k] = vcat(cl[k], cl[k+1])
                deleteat!(cl, k + 1)
                merged = true
                break
            end
        end
    end
    out = similar(ys)
    for c in cl, (j, k) in enumerate(c)
        out[k] = lo(c) + (j - 1) * gap
    end
    res = similar(out)
    res[p] = out
    return res
end

# ── plot_levels ────────────────────────────────────────────────────────────

"""
    plot_levels(lb; relative = true, size, title) -> Figure

The single-particle ladder, one panel per species. Each level is coloured by its
dominant irrep and bracketed with the states of its block; the marker on the left
says how far to trust the label:

- filled: the block is closed (an invariant subspace, label measured);
- hollow with "?": unclosed, so no label (or the declared one, as a square);
- square: a declared label shown because the measurement gave none;
- red outline: a declared label that the measured one contradicts.

Right of the level: the label and degeneracy, the closure, the largest leakage,
and a stacked bar of which irreps the leakage goes to (its composition; the
number says how much).
"""
function plot_levels(lb::LevelBundle; relative::Bool = true, size = nothing,
                     title = "levels")
    kinds = unique(lb.kind)
    nk = max(length(kinds), 1)
    fig = Figure(size = size === nothing ? (420 * nk + 140, 560) : size)
    _title!(fig[0, 1:nk], title)
    for (n, k) in pairs(kinds)
        bl = [b for b in eachindex(lb.blocks) if lb.kind[first(lb.blocks[b])] === k]
        sel = reduce(vcat, collect.(lb.blocks[bl]); init = Int[])
        E0 = relative ? minimum(lb.energies[sel]) : 0.0
        sc = relative ? 1000.0 : 1.0
        ax = Axis(fig[1, n]; title = _kindname(k),
                  ylabel = relative ? "E − E₀ (meV)" : "E (eV)",
                  ygridcolor = _GRID, xgridvisible = false)
        ys = [(lb.energies[first(lb.blocks[b])] - E0) * sc for b in bl]
        lo, hi = isempty(sel) ? (0.0, 1.0) : extrema((lb.energies[sel] .- E0) .* sc)
        # a panel of one level (or of exactly degenerate ones) still gets an axis:
        # 1 meV (or 1 meV in eV) about it, not the round-off spread of its energies
        span = max(hi - lo, relative ? 1.0 : 1e-3)
        hi - lo < span && ylims!(ax, lo - 0.6span, lo + 0.6span)
        yt = _repel(ys, 0.055 * span)
        for (j, b) in pairs(bl)
            ids = lb.blocks[b]
            lab = lb.labels[b]
            c = labelled(lb) ? _icol(lb, _block_irrep(lb, b)) : UNLABELLED_COLOR
            yy = [(lb.energies[i] - E0) * sc for i in ids]
            for (i, y) in zip(ids, yy)       # each state in its own dominant irrep
                lines!(ax, [-0.4, 0.4], [y, y]; color = _scol(lb, i), linewidth = 2.5)
            end
            # bracket the block: its states' energy range
            y1, y2 = extrema(yy)
            if length(ids) > 1 && y2 - y1 > 1e-3 * span
                lines!(ax, [-0.44, -0.47, -0.47, -0.44], [y1, y1, y2, y2];
                       color = _MUTED, linewidth = 1)
            end
            ym = sum(yy) / length(yy)
            filled = lb.closed[b] && lb.source[b] !== :none
            scatter!(ax, [-0.58], [ym];
                     marker = lb.source[b] === :declared ? :rect : :circle,
                     markersize = 11,
                     color = filled ? c : RGBf(1, 1, 1),
                     strokecolor = lb.disagree[b] ? DISAGREE_COLOR : c,
                     strokewidth = lb.disagree[b] ? 2.5 : 1.5)
            lb.closed[b] || text!(ax, -0.58, ym; text = "?", align = (:center, :center),
                                  fontsize = 8, color = _MUTED)
            # leader from the level to its (repelled) label
            abs(yt[j] - ym) > 1e-12 * span &&
                lines!(ax, [0.42, 0.5], [ym, yt[j]]; color = _GRID, linewidth = 0.8)
            # the leakage's composition, as a stacked bar of fixed length
            leak = labelled(lb) ? maximum(lb.leakage[ids]) : NaN
            x0 = 0.52
            # no bar for round-off, nor for a leakage at or below the method floor
            fl = maximum(lb.floor[ids])
            if labelled(lb) && leak > 1e-12 && !(leak <= fl)
                w = copy(lb.block_weights[b])
                w[_own_irreps(lb, b)] .= 0
                w = max.(w, 0)
                tot = sum(w)
                if tot > 0
                    for kk in eachindex(w)
                        f = w[kk] / tot
                        f < 1e-3 && continue
                        poly!(ax, Rect2f(x0, yt[j] - 0.012span, 0.4f, 0.024span);
                              color = _icol(lb, lb.irreps[kk]))
                        x0 += 0.4f
                    end
                end
            end
            deg = length(ids) > 1 ? " ×$(length(ids))" : ""
            _tr_pair(lb, b) && (deg *= " (T pair)")
            txt = labelled(lb) ?
                  @sprintf("%s%s   c %s   leak %s", lab, deg, _sci(lb.closure[b]), _sci(leak)) :
                  lab * deg
            text!(ax, 0.98, yt[j]; text = txt, align = (:left, :center), fontsize = 10,
                  color = lab == "?" ? _MUTED : _INK)
        end
        xlims!(ax, -0.75, 3.6)
        hidexdecorations!(ax)
        hidespines!(ax, :t, :r)
    end
    ks = _present(lb.irreps, lb.block_weights)
    extra = (; Symbol("closed") => MarkerElement(marker = :circle, color = _MUTED, markersize = 10),
             Symbol("unclosed") => MarkerElement(marker = :circle, color = :white,
                                                 strokecolor = _MUTED, strokewidth = 1.5,
                                                 markersize = 10),
             Symbol("declared") => MarkerElement(marker = :rect, color = _MUTED, markersize = 10),
             Symbol("disagrees") => MarkerElement(marker = :circle, color = :white,
                                                  strokecolor = DISAGREE_COLOR, strokewidth = 2.5,
                                                  markersize = 10))
    _irrep_legend!(fig[1, nk+1], lb.irreps, lb.double, ks; extra)
    return fig
end

# ── plot_irrep_weights ─────────────────────────────────────────────────────

"""
    plot_irrep_weights(lb; per = :state, states = all) -> Figure

The basic breaking figure. Top: each state's (or level's, `per = :level`) irrep
weights stacked to 1. Bottom, on a log axis: every weight but the dominant one,
which is what the stack hides, with the method floor (`level_bundle(…; floor =
true)`) as a black bar per state. A weight below its floor is symmetric to method
accuracy.
"""
function plot_irrep_weights(lb::LevelBundle; per::Symbol = :state, states = nothing,
                            size = nothing, title = "irrep weights")
    per in (:state, :level) || throw(ArgumentError("per must be :state or :level"))
    if per === :state
        idx = states === nothing ? collect(1:length(lb)) : collect(states)
        ws = labelled(lb) ? lb.weights[idx] : [Float64[] for _ in idx]
        fl = lb.floor[idx]
        tick = [labelled(lb) ? _tag(lb, i) * "\n" * _slabel(lb, i) : _tag(lb, i) for i in idx]
    else
        idx = states === nothing ? collect(eachindex(lb.blocks)) : collect(states)
        ws = labelled(lb) ? lb.block_weights[idx] : [Float64[] for _ in idx]
        fl = [maximum(lb.floor[lb.blocks[b]]) for b in idx]
        tick = [string(lb.blocks[b]) * "\n" * lb.labels[b] for b in idx]
    end
    nx = length(idx)
    fig = Figure(size = size === nothing ? (max(480, 46 * nx + 200), 560) : size)
    _title!(fig[0, 1], title)
    ax1 = Axis(fig[1, 1]; ylabel = "weight", xticks = (1:nx, tick),
               xgridvisible = false, ygridcolor = _GRID, xticklabelsize = 10)
    ax2 = Axis(fig[2, 1]; ylabel = "weight (log)", yscale = log10,
               xticks = (1:nx, tick), xgridvisible = false, ygridcolor = _GRID,
               xticklabelsize = 10)
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1; grid = false)
    if labelled(lb)
        xs = Int[]; ys = Float64[]; grp = Int[]
        for (n, w) in pairs(ws), k in eachindex(w)
            push!(xs, n); push!(ys, max(w[k], 0)); push!(grp, k)
        end
        barplot!(ax1, xs, ys; stack = grp, color = [_icol(lb, lb.irreps[k]) for k in grp],
                 gap = 0.25, strokecolor = :white, strokewidth = 1)
        for (n, w) in pairs(ws)
            d = argmax(w)
            for k in eachindex(w)
                (k == d || w[k] <= _LOGMIN) && continue
                scatter!(ax2, [n], [_clip(w[k])]; color = _icol(lb, lb.irreps[k]),
                         markersize = 9, strokecolor = :white, strokewidth = 1)
            end
        end
        if any(isfinite, fl)
            scatter!(ax2, findall(isfinite, fl), _clip.(fl[isfinite.(fl)]);
                     marker = :hline, markersize = 22, color = _INK)
        else
            text!(ax2, 0.99, 0.02; text = "no method floor (level_bundle(…; floor = true))",
                  space = :relative, align = (:right, :bottom), fontsize = 9, color = _MUTED)
        end
    else
        barplot!(ax1, 1:nx, ones(nx); color = UNLABELLED_COLOR, gap = 0.25)
        _note!(ax2, "no analysis: no weights")
    end
    ylims!(ax1, 0, 1)
    ylims!(ax2, _LOGMIN, 2)
    extra = labelled(lb) && any(isfinite, fl) ?
            (; Symbol("method floor") => MarkerElement(marker = :hline, color = _INK,
                                                       markersize = 16)) : (;)
    _irrep_legend!(fig[1:2, 2], lb.irreps, lb.double, _present(lb.irreps, ws; cut = 1e-12);
                   extra)
    rowsize!(fig.layout, 1, Relative(0.45))
    return fig
end

# ── plot_operation_map ─────────────────────────────────────────────────────

"""
    plot_operation_map(lb; tol = 1e-6) -> Figure

Blocks × element classes, coloured by the closure of each class on each block (log
scale), with the value in the cell; cells above `tol` (broken) are outlined. A row
with the rotations at round-off and the mirrors not says the shape keeps Cₙ and
breaks the σ.
"""
function plot_operation_map(lb::LevelBundle; tol::Real = 1e-6, size = nothing,
                            title = "operation map")
    nb, nc = length(lb.blocks), length(lb.classes)
    fig = Figure(size = size === nothing ? (max(420, 62 * max(nc, 3) + 260), 34 * nb + 160) : size)
    rows = [string(lb.blocks[b]) * "  " * lb.labels[b] for b in 1:nb]
    ax = Axis(fig[1, 1]; title = title, yreversed = true,
              xticks = (1:max(nc, 1), nc == 0 ? Any["—"] : _class_ticks(lb.classes)),
              yticks = (1:nb, rows), xgridvisible = false, ygridvisible = false)
    if nc == 0
        _note!(ax, "no analysis: no classes")
        xlims!(ax, 0.5, 1.5)
        ylims!(ax, nb + 0.5, 0.5)
        return fig
    end
    L = log10.(max.(lb.class_closure, _LOGMIN))
    hm = heatmap!(ax, 1:nc, 1:nb, permutedims(L); colormap = _SEQ,
                  colorrange = (log10(_LOGMIN), 0))
    for b in 1:nb, c in 1:nc
        x = lb.class_closure[b, c]
        dark = L[b, c] > -6
        text!(ax, c, b; text = _sci(x), align = (:center, :center), fontsize = 9,
              color = dark ? :white : _INK)
        x > tol && poly!(ax, Rect2f(c - 0.5, b - 0.5, 1, 1); color = :transparent,
                         strokecolor = DISAGREE_COLOR, strokewidth = 1.5)
    end
    Colorbar(fig[1, 2], hm; label = "log₁₀ closure", ticks = -16:4:0)
    Label(fig[2, 1], @sprintf("outlined: closure > %.0e (broken)", tol);
          fontsize = 10, color = _MUTED, tellwidth = false)
    return fig
end

# ── plot_symmetry_quality ──────────────────────────────────────────────────

"""
    plot_symmetry_quality(lb) -> Figure

Whether the labels can be used. Left, per block on a log axis: the closure
(filled, coloured by irrep) against `closure_tol` (dashed), the time-reversal
defect (diamond) and the class spread (cross). Right: the charge gap against
the gap a selection rule needs (100×): above it, `symmetry = :use` is safe.
"""
function plot_symmetry_quality(lb::LevelBundle; size = nothing, title = "symmetry quality")
    nb = length(lb.blocks)
    fig = Figure(size = size === nothing ? (max(560, 40 * nb + 360), 440) : size)
    _title!(fig[0, 1:2], title)
    tick = [string(lb.blocks[b]) * "\n" * lb.labels[b] for b in 1:nb]
    ax = Axis(fig[1, 1]; yscale = log10, ylabel = "defect", xticks = (1:nb, tick),
              xgridvisible = false, ygridcolor = _GRID, xticklabelsize = 10)
    ag = Axis(fig[1, 2]; yscale = log10, ylabel = "charge gap (×)", xticks = ([1], ["rule"]),
              xgridvisible = false, ygridcolor = _GRID)
    colsize!(fig.layout, 2, Fixed(110))
    ylims!(ax, _LOGMIN, 10)
    xlims!(ax, 0.4, nb + 0.6)
    if !labelled(lb)
        _note!(ax, "no analysis: no closures")
        _note!(ag, "–")
        ylims!(ag, 1, 1e4)
        return fig
    end
    cols = [_icol(lb, _block_irrep(lb, b)) for b in 1:nb]
    hlines!(ax, [lb.closure_tol]; color = _MUTED, linestyle = :dash, linewidth = 1)
    text!(ax, nb + 0.4, lb.closure_tol; text = "closure tol", align = (:right, :bottom),
          fontsize = 9, color = _MUTED)
    scatter!(ax, 1:nb, _clip.(lb.closure); color = cols, markersize = 11,
             strokecolor = :white, strokewidth = 1)
    any(isfinite, lb.tr_closure) &&
        scatter!(ax, (1:nb) .+ 0.18, _clip.(lb.tr_closure); marker = :diamond,
                 color = :white, strokecolor = cols, strokewidth = 1.5, markersize = 10)
    scatter!(ax, (1:nb) .- 0.18, _clip.(lb.class_spread); marker = :xcross, color = cols,
             markersize = 9)
    xlims!(ax, 0.4, nb + 0.6)
    g = lb.rule.gap
    need = lb.rule_needed
    hlines!(ag, [need]; color = _MUTED, linestyle = :dash, linewidth = 1)
    if isfinite(g)
        ok = g >= need
        scatter!(ag, [1], [g]; color = ok ? _INK : DISAGREE_COLOR, markersize = 12)
        text!(ag, 1, g; text = ok ? "  usable" : "  NOT usable", align = (:left, :center),
              fontsize = 10, color = _INK)
        ylims!(ag, min(g, need) / 10, max(g, need) * 10)
    else
        _note!(ag, "no charges")
        ylims!(ag, 1, 1e4)
    end
    xlims!(ag, 0.5, 2.2)
    Legend(fig[2, 1:2],
           [MarkerElement(marker = :circle, color = _MUTED, markersize = 10),
            MarkerElement(marker = :diamond, color = :white, strokecolor = _MUTED,
                          strokewidth = 1.5, markersize = 10),
            MarkerElement(marker = :xcross, color = _MUTED, markersize = 9),
            LineElement(color = _MUTED, linestyle = :dash)],
           ["closure", "time reversal", "class spread", "threshold"];
           orientation = :horizontal, framevisible = false, labelsize = 10)
    return fig
end

# ── plot_charges ───────────────────────────────────────────────────────────

"""
    plot_charges(lb) -> Figure

The eigenphases of the adapting rotation on the closed blocks, on the unit circle
(electrons on the outer ring, holes on the inner), coloured by irrep. Partners of
one irrep sit at its characteristic phases; charges that should coincide and do
not show the rule's tolerance at work.
"""
function plot_charges(lb::LevelBundle; size = (520, 540), title = nothing)
    fig = Figure(size = size)
    ttl = title === nothing ?
          (isempty(lb.adapt) ? "charges" : "charges: eigenphases of " * lb.adapt) : title
    ax = Axis(fig[1, 1]; title = ttl, aspect = DataAspect(), xgridvisible = false,
              ygridvisible = false)
    hidedecorations!(ax); hidespines!(ax)
    θ = range(0, 2π; length = 361)
    kinds = unique(lb.kind)
    radius(k) = _ishole(k) && length(kinds) > 1 ? 0.78 : 1.0
    for k in kinds
        r = radius(k)
        lines!(ax, r .* cos.(θ), r .* sin.(θ); color = _GRID, linewidth = 1)
    end
    lines!(ax, [-1.15, 1.15], [0, 0]; color = _GRID, linewidth = 0.8)
    lines!(ax, [0, 0], [-1.15, 1.15]; color = _GRID, linewidth = 0.8)
    have = findall(!ismissing, lb.charges)
    if isempty(have)
        _note!(ax, "no charges (no closed, adapted block)")
    else
        for i in have
            b = findfirst(bb -> i in bb, lb.blocks)
            λ = lb.charges[i]
            r = radius(lb.kind[i])
            scatter!(ax, [r * real(λ)], [r * imag(λ)]; color = _scol(lb, i),
                     markersize = 12, strokecolor = :white, strokewidth = 1)
        end
        # one label per distinct charge, listing its states
        groups = Tuple{Symbol,Vector{Int}}[]
        for k in kinds
            sel = [i for i in have if lb.kind[i] === k]
            sort!(sel; by = i -> mod(angle(lb.charges[i]), 2π))
            for i in sel
                if !isempty(groups) && groups[end][1] === k &&
                   abs(lb.charges[i] - lb.charges[last(groups[end][2])]) < 0.05
                    push!(groups[end][2], i)
                else
                    push!(groups, (k, [i]))
                end
            end
        end
        for (k, ids) in groups
            λ = lb.charges[first(ids)]
            r = radius(k) + (_ishole(k) ? -0.13 : 0.13)
            txt = join((_tag(lb, i) for i in ids), ",")
            length(ids) > 4 && (txt = _tag(lb, first(ids)) * "…" * _tag(lb, last(ids)))
            text!(ax, r * real(λ), r * imag(λ); text = txt, align = (:center, :center),
                  fontsize = 9, color = _INK)
        end
    end
    limits!(ax, -1.35, 1.35, -1.35, 1.35)
    ks = _present(lb.irreps, [lb.block_weights[b] for b in eachindex(lb.blocks) if lb.closed[b]])
    _irrep_legend!(fig[1, 2], lb.irreps, lb.double, ks)
    return fig
end

# ── plot_parent_mixing ─────────────────────────────────────────────────────

"""
    plot_parent_mixing(lb) -> Figure

The broken blocks × the reference levels: each block's average weight on each
level of the symmetric reference (`level_bundle(…; reference = (ref, ref_a))`),
with a last column for the weight outside the reference basis. Rows and columns
are ordered by irrep, so a block that mixes parents of different irreps shows as
weight off the irrep's own band.
"""
function plot_parent_mixing(lb::LevelBundle; size = nothing, title = "parent mixing")
    p = lb.parent
    p === nothing && throw(ArgumentError(
        "the bundle has no parent decomposition; build it with " *
        "level_bundle(ks, a; reference = (ref_ks, ref_analysis))"))
    nb, nl = Base.size(p.mixing)
    ord(l) = something(findfirst(==(l), lb.irreps), length(lb.irreps) + 1)
    rows = sortperm(collect(1:nb); by = b -> (ord(_block_irrep(lb, b)), b))
    cols = sortperm(collect(1:nl); by = l -> (ord(p.ref_labels[l]), l))
    out = [sum(p.outside[lb.blocks[b]]) / length(lb.blocks[b]) for b in 1:nb]
    M = hcat(p.mixing[rows, cols], out[rows])
    fig = Figure(size = size === nothing ? (max(460, 44 * (nl + 1) + 260), 34 * nb + 200) : size)
    xt = vcat([string(p.ref_levels[l]) * "\n" * p.ref_labels[l] for l in cols], ["outside"])
    yt = [string(lb.blocks[b]) * "  " * lb.labels[b] for b in rows]
    ax = Axis(fig[1, 1]; title = title, xlabel = "reference level", ylabel = "block",
              yreversed = true, xticks = (1:nl+1, xt), yticks = (1:nb, yt),
              xticklabelsize = 10)
    hm = heatmap!(ax, 1:nl+1, 1:nb, permutedims(M); colormap = _SEQ, colorrange = (0, 1))
    for r in 1:nb, c in 1:nl+1
        M[r, c] < 1e-3 && continue
        text!(ax, c, r; text = @sprintf("%.3f", M[r, c]), align = (:center, :center),
              fontsize = 9, color = M[r, c] > 0.45 ? :white : _INK)
    end
    vlines!(ax, [nl + 0.5]; color = _INK, linewidth = 1)
    Colorbar(fig[1, 2], hm; label = "weight")
    return fig
end

# ── plot_field_irreps ──────────────────────────────────────────────────────

_trivial(t) = something(findfirst(i -> !t.double[i] &&
                                      all(c -> isapprox(c, 1; atol = 1e-8), t.chars[i, :]),
                                  eachindex(t.dims)), 1)

# the invariant irrep's position among the single-valued ones (field fractions' order)
_trivial_single(t) = findfirst(==(_trivial(t)), findall(.!t.double))

"""
    plot_field_irreps(lb) -> Figure
    plot_field_irreps(results) -> Figure

The irrep content of the fields that break the symmetry (`field_content` results,
or those in a bundle built with `fields`). Left: per field, the fractions over the
box and weighted by the states' density, stacked to 1. Right, log: every fraction
but the invariant one, box filled and density-weighted hollow — the part that can
break the levels.
"""
plot_field_irreps(lb::LevelBundle; kwargs...) = plot_field_irreps(lb.fields; kwargs...)

function plot_field_irreps(rs::AbstractVector; size = nothing, title = "field irreps")
    nf = length(rs)
    fig = Figure(size = size === nothing ? (820, 70 * max(nf, 1) + 200) : size)
    _title!(fig[0, 1:3], title)
    # A field with interfaces is decomposed in the exact subgroup (`C6v∩grid`), whose
    # table is another group's: its row says so, and its legend entries name it.
    kx = findfirst(r -> r.exact, rs)
    main = isempty(rs) ? "" : rs[kx === nothing ? 1 : kx].frame.group
    grp(r) = r.frame.group == main ? "" : " (" * r.frame.group * ")"
    rowlab = String[]
    for r in rs
        push!(rowlab, string(r.field) * " box" * grp(r), string(r.field) * " ρ-weighted" * grp(r))
    end
    nr = 2nf
    ax1 = Axis(fig[1, 1]; yreversed = true, xlabel = "fraction",
               yticks = (1:max(nr, 1), nr == 0 ? ["—"] : rowlab), ygridvisible = false,
               xgridcolor = _GRID)
    ax2 = Axis(fig[1, 2]; yreversed = true, xscale = log10, xlabel = "fraction (log), non-invariant",
               yticks = (1:max(nr, 1), nr == 0 ? ["—"] : rowlab), ygridvisible = false,
               xgridcolor = _GRID)
    linkyaxes!(ax1, ax2)
    hideydecorations!(ax2; grid = false)
    if nf == 0
        _note!(ax1, "no fields")
        _note!(ax2, "")
        return fig
    end
    # legend: one entry per irrep drawn, keyed by its own group, in table order
    seen = Tuple{String,Int}[]
    entry(g, a) = (g, a) in seen || push!(seen, (g, a))
    tabs = Dict{String,Any}()
    for (j, r) in pairs(rs)
        t = r.frame.table
        g = r.frame.group
        tabs[g] = t
        single = findall(.!t.double)
        triv = _trivial(t)      # an index into the table; `single` maps the fractions to it
        for (row, v, hollow) in ((2j - 1, r.box, false), (2j, r.weighted, true))
            all(isnan, v) && continue
            x0 = 0.0
            for (k, a) in pairs(single)
                f = max(v[k], 0)
                f > 1e-3 && entry(g, a)
                f <= 0 && continue
                poly!(ax1, Rect2f(x0, row - 0.35, f, 0.7); color = _icol(t, t.labels[a]),
                      strokecolor = :white, strokewidth = 1)
                x0 += f
                (a == triv || f <= _LOGMIN) && continue
                c = _icol(t, t.labels[a])
                scatter!(ax2, [_clip(f)], [row]; color = hollow ? RGBf(1, 1, 1) : c,
                         strokecolor = c, strokewidth = 1.5, markersize = 10)
                f > 1e-12 && entry(g, a)
            end
        end
    end
    xlims!(ax1, 0, 1)
    xlims!(ax2, _LOGMIN, 2)
    sort!(seen; by = e -> (e[1] != main, e[1], e[2]))
    els = Any[PolyElement(color = _icol(tabs[g], tabs[g].labels[a])) for (g, a) in seen]
    labs = [tabs[g].labels[a] * (g == main ? "" : " (" * g * ")") for (g, a) in seen]
    isempty(els) || Legend(fig[1, 3], els, labs, "irrep"; framevisible = false, labelsize = 11,
                           titlesize = 11, rowgap = 2)
    return fig
end

# ── plot_sweep ─────────────────────────────────────────────────────────────

"""
    plot_sweep(sw::SymmetrySweep; tracks = all, ncols = 3) -> Figure

Per tracked level, its irrep weights stacked to 1 against the breaking parameter
(small multiples). Below, sharing the parameter axis: every track's leakage (log),
and, when the sweep has fields, each field's non-invariant fraction over the box
(log) — the leakage beside its cause. Crossings are dashed verticals.
"""
function plot_sweep(sw::SymmetrySweep; tracks = nothing, ncols::Integer = 3, size = nothing,
                    title = "symmetry sweep", xlabel = "parameter")
    t = sw.frame.table
    trs = tracks === nothing ? collect(eachindex(sw.labels)) : collect(tracks)
    nt = length(trs)
    nc = min(ncols, max(nt, 1))
    nrow = cld(nt, nc)
    hasf = any(!isempty, sw.fields)
    fig = Figure(size = size === nothing ? (300 * nc + 160, 210 * nrow + (hasf ? 440 : 230)) : size)
    _title!(fig[0, 1:nc], title)
    x = sw.parameter
    cross = unique(x[s] for (s, _, _) in sw.crossings)
    _rng(v) = length(v) > 1 && v == first(v):last(v) ? "$(first(v)):$(last(v))" : join(v, ",")
    name = [_rng(sw.states[tr, 1]) * "  " * sw.labels[tr] for tr in eachindex(sw.labels)]
    present = Set{Int}()
    for (n, tr) in pairs(trs)
        r, c = fldmod1(n, nc)
        ax = Axis(fig[r, c]; title = name[tr], titlesize = 12, xgridvisible = false,
                  ygridcolor = _GRID)
        lo = zeros(length(x))
        for k in eachindex(t.dims)
            w = [max(sw.weights[tr, s][k], 0) for s in eachindex(x)]
            maximum(w) < 1e-6 && continue
            push!(present, k)
            band!(ax, x, lo, lo .+ w; color = _icol(t, t.labels[k]))
            lo = lo .+ w
        end
        isempty(cross) || vlines!(ax, cross; color = _INK, linestyle = :dash, linewidth = 0.8)
        ylims!(ax, 0, 1)
        c == 1 || hideydecorations!(ax; grid = false)
        hidexdecorations!(ax; grid = false)
    end
    axl = Axis(fig[nrow+1, 1:nc]; yscale = log10, ylabel = "leakage",
               xlabel = hasf ? "" : xlabel, xgridvisible = false, ygridcolor = _GRID)
    ends = Float64[]
    lall = Float64[]                       # every plotted log value: the axis range
    for tr in trs
        d = argmax(sw.weights[tr, 1])
        col = _icol(t, t.labels[d])
        # a track of several irreps (Γ8 ⊕ Γ9) leaks only outside all of them
        own = _label_irreps(sw.labels[tr], t.labels)
        y = isempty(own) ? _clip.(sw.leakage[tr, :]) :
            [_clip(1 - sum(sw.weights[tr, s][k] for k in own)) for s in eachindex(x)]
        lines!(axl, x, y; color = col, linewidth = 2)
        scatter!(axl, x, y; color = col, markersize = 8)
        push!(ends, log10(y[end])); append!(lall, log10.(y))
    end
    # end labels, repelled in log space so equal leakages stay readable
    ly = _repel(ends, 0.09 * max(maximum(lall) - minimum(lall), 0.5))   # ~ one text line
    for (j, tr) in pairs(trs)
        text!(axl, x[end], 10.0^ly[j]; text = "  " * name[tr], align = (:left, :center),
              fontsize = 9, color = _INK)
    end
    isempty(cross) || vlines!(axl, cross; color = _INK, linestyle = :dash, linewidth = 0.8)
    if hasf
        hidexdecorations!(axl; grid = false)
        axf = Axis(fig[nrow+2, 1:nc]; yscale = log10, ylabel = "field, non-invariant",
                   xlabel = xlabel, xgridvisible = false, ygridcolor = _GRID)
        names = unique(reduce(vcat, [collect(keys(d)) for d in sw.fields]))
        for (j, nm) in pairs(names)
            y = [haskey(sw.fields[s], nm) ?
                 _clip(1 - sw.fields[s][nm].box[_trivial_single(sw.fields[s][nm].frame.table)]) : NaN
                 for s in eachindex(x)]
            lines!(axf, x, y; color = _MUTED, linewidth = 1.5,
                   linestyle = (:solid, :dash, :dot, :dashdot)[mod1(j, 4)])
            k = findlast(isfinite, y)
            k === nothing || text!(axf, x[k], y[k]; text = "  " * string(nm),
                                   align = (:left, :center), fontsize = 9, color = _INK)
        end
        isempty(cross) || vlines!(axf, cross; color = _INK, linestyle = :dash, linewidth = 0.8)
        linkxaxes!(axl, axf)
    end
    # room on the right for the end labels
    Δ = length(x) > 1 ? x[end] - x[1] : 1.0
    xlims!(axl, x[1] - 0.03Δ, x[end] + 0.2Δ)
    _irrep_legend!(fig[1:nrow, nc+1], t.labels, t.double, sort!(collect(present)))
    return fig
end
