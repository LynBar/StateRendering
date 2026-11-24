# ---------------------------------------------------------------------------
# The radiative cascade figure: a `CICalc.CascadeBundle` (XX → X → 0, or any chain
# of sectors that each remove one e–h pair).
#
# LEFT, THE PATHWAYS. One column per sector (XX | X | 0), its levels stacked by
# energy, highest on top (the spacing is schematic, one row per level; the label
# carries the real energy), so many levels grow the figure downwards, not sideways. Levels of one fine-structure manifold (within `manifold_tol`)
# share a shaded box. Arrows are the radiative lines: width ∝ photons per cascade,
# colour = the axial share of the polarisation in the frame's axes. Only the
# `max_arrows` strongest are drawn, and the `label_arrows` strongest per step carry
# their photon energy and yield, written under the level at the arrow's unique end
# ("in" / "out"), so a fan of arrows never stacks its labels. The flow into levels the bundle
# does not show goes to a grey "others" slot, so nothing disappears silently.
#
# RIGHT, THE SPECTRUM, SPLIT BY CHANNEL. One row per irrep of the intermediate
# sector's level (the X level for XX → X → 0), sharing the photon-energy axis. In a
# row, the last step's lines (X → 0) point up and the earlier steps' (XX → X) point
# down, so the two photons of a cascade through one X level face each other. The
# height is photons per cascade, on a log axis by default: a few strong lines and
# many weak ones are the normal case, and a linear axis hides the weak ones.
#
# Colour on the level marks and row labels is the irrep, as in every other figure;
# colour on the arrows and sticks is polarisation, with its own colour bar, so the
# two never share a scale.
# ---------------------------------------------------------------------------

const _POL_GRAD = cgrad([RGBf(0.13, 0.40, 0.75), RGBf(0.55, 0.45, 0.55),
                         RGBf(0.85, 0.42, 0.08)])

_fmt_shift(dE) = abs(dE) < 1e-3 ? @sprintf("%+.1f µeV", 1e6 * dE) : @sprintf("%+.2f meV", 1e3 * dE)
function _fmt_tau(τ)                     # τ in ns
    isfinite(τ) || return ""
    for (s, u) in ((1.0, "ns"), (1e3, "µs"), (1e6, "ms"))
        τ < 1000s && return @sprintf("τ %.3g %s", τ / s, u)
    end
    return @sprintf("τ %.3g s", τ / 1e9)
end

# a level's lifetime: one value, or the range when its states differ (a merged
# fine-structure level of bright and dark states)
function _fmt_tau_range(τs)
    t = filter(isfinite, τs)
    isempty(t) && return ""
    lo, hi = extrema(t)
    hi <= 1.01lo && return _fmt_tau(lo)
    return _fmt_tau(lo) * " … " * _fmt_tau(hi)[4:end]
end

# per level pair (f, i) of step k: photons per cascade, photon energy, axial share
function _level_lines(cb::CascadeBundle, k::Integer)
    Y = line_yield(cb, k)
    out = NamedTuple[]
    for (li, ri) in enumerate(cb.levels[k]), (lf, rf) in enumerate(cb.levels[k+1])
        y = sum(Y[rf, ri])
        # the polarisation split weighted by what is actually emitted
        w = [sum(Y[f, i] / max(cb.Γ[k][f, i], eps()) * cb.Γpol[k][j][f, i]
                 for f in rf, i in ri) for j in 1:3]
        s = sum(w)
        push!(out, (; k, li, lf, y, ħω = cb.ħω[k][first(rf), first(ri)],
                    axial = s > 0 ? w[3] / s : 0.0))
    end
    return out
end

_level_label(cb, k, l) = isempty(cb.level_labels[k][l]) ? "?" : cb.level_labels[k][l]

# consecutive levels of a tier closer than `tol` (eV): the fine-structure manifolds
function _manifolds(E::AbstractVector, tol::Real)
    out = UnitRange{Int}[]
    lo = 1
    for l in 2:length(E)+1
        if l > length(E) || E[l] - E[l-1] > tol
            push!(out, lo:l-1)
            lo = l
        end
    end
    return out
end

"""
    plot_cascade(cb::CascadeBundle; size = nothing, min_yield = 1e-4, max_arrows = 16,
                 label_arrows = 3, manifold_tol = 1e-3, scale = :log, split = :irrep,
                 title = nothing) -> Figure

The radiative cascade: pathways between levels (left) and the emission spectrum
they produce (right).

  * `min_yield` — lines below this many photons per cascade are not drawn (and set
    the floor of the log axis);
  * `max_arrows` — at most this many pathway arrows, the strongest; `label_arrows`
    of them per step carry their photon energy and yield;
  * `manifold_tol` (eV) — levels this close share a shaded box (fine structure);
  * `scale` — `:log` or `:linear` spectrum heights;
  * `split` — `:irrep`: one spectrum row per irrep of the intermediate level; `:none`:
    a single row.

Everything shown is radiative: lifetimes are 1/Γ into the whole next sector, and
populations assume no relaxation or spin flips between levels of a sector (see
`CICalc.cascade_bundle`).
"""
function plot_cascade(cb::CascadeBundle; size = nothing, min_yield::Real = 1e-4,
                      max_arrows::Integer = 16, label_arrows::Integer = 3,
                      manifold_tol::Real = 1e-3, scale::Symbol = :log,
                      split::Symbol = :irrep, title = nothing)
    scale in (:log, :linear) || throw(ArgumentError("scale must be :log or :linear"))
    split in (:irrep, :none) || throw(ArgumentError("split must be :irrep or :none"))
    K = length(cb.names)
    lines_all = reduce(vcat, [_level_lines(cb, k) for k in 1:K-1])
    shown = filter(s -> s.y >= min_yield, lines_all)

    # the tier that keys the spectrum rows: the intermediate sector, else the first
    pivot = K >= 3 ? 2 : 1
    # a line's channel: the pivot level it reaches (its final) or leaves (its initial)
    function key(s)
        split === :none && return ""
        t, l = s.k + 1 == pivot ? (s.k + 1, s.lf) : (s.k, s.li)
        return _level_label(cb, t, l)
    end
    keys = unique(key.(shown))
    order(l) = (i = findfirst(==(first(Base.split(l, r"[ ⊕+]"; keepempty = false))), cb.irreps);
                i === nothing ? typemax(Int) : i)
    split === :irrep && sort!(keys; by = l -> (order(l), l))
    nrow = max(length(keys), 1)

    unshown = get(cb.metadata, :unshown, Float64[])
    others = [k > 1 && k - 1 <= length(unshown) && unshown[k-1] > min_yield for k in 1:K]
    nrows = maximum(length(cb.levels[k]) + others[k] for k in 1:K)
    fig = Figure(size = size === nothing ?
                 (1250, max(560, 200 + 62 * nrows, 160 + 150 * nrow)) : size)
    ttl = title === nothing ?
        @sprintf("radiative cascade %s   (E_P = %.3g eV, n_r = %.3g)",
                 join(cb.names, " → "), cb.E_P, cb.n_r) : title
    Label(fig[0, 1:3], ttl; fontsize = 15, tellwidth = false)

    _pathways!(fig[1, 1], cb, lines_all, others, unshown; min_yield, max_arrows,
               label_arrows, manifold_tol)
    _channel_spectrum!(fig[1, 2], cb, shown, keys, key, pivot; min_yield, scale)
    Colorbar(fig[1, 3]; colormap = _POL_GRAD, limits = (0, 1),
             label = "axial share (" * cb.polnames[3] * ")", width = 12)
    colsize!(fig.layout, 1, Relative(0.5))

    nhid = count(s -> s.y >= min_yield, lines_all) - min(max_arrows,
                                                         count(s -> s.y >= min_yield, lines_all))
    note = "radiative only: no relaxation between levels of a sector"
    nhid > 0 && (note *= "  ·  $nhid weaker pathway(s) not drawn as arrows (all are in the spectrum)")
    Label(fig[2, 1:3], note; fontsize = 10, color = _MUTED, tellwidth = false)
    return fig
end

# the row of each level in a column: highest energy on top, one unit per level and
# an extra gap between fine-structure manifolds; "others" last. Centred on 0.
function _column_rows(E::AbstractVector, others::Bool, manifold_tol::Real)
    n = length(E)
    y = zeros(n + others)
    ord = sortperm(E; rev = true)
    cur = 0.0
    for (j, l) in enumerate(ord)
        j > 1 && (cur -= 1.0 + (E[ord[j-1]] - E[l] > manifold_tol ? 0.45 : 0.0))
        y[l] = cur
    end
    others && (y[end] = cur - 1.45)
    return y .- (maximum(y) + minimum(y)) / 2
end

function _pathways!(pos_, cb, lines_all, others, unshown; min_yield, max_arrows,
                    label_arrows, manifold_tol)
    K = length(cb.names)
    ax = Axis(pos_; title = "pathways (width ∝ photons per cascade)")
    hidedecorations!(ax); hidespines!(ax)
    L = 0.5                                     # length of a level line; columns 1 apart
    pos = Vector{Vector{Tuple{Float64,Float64}}}(undef, K)       # left end of each line
    ymax = 0.0
    for k in 1:K
        x = Float64(k - 1)
        E = [cb.energies[k][first(r)] for r in cb.levels[k]]
        y = _column_rows(E, others[k], manifold_tol)
        pos[k] = [(x, y[l]) for l in eachindex(E)]
        ymax = max(ymax, maximum(abs, y; init = 0.0))
        # the fine-structure manifolds: one shaded box behind their rows
        for m in _manifolds(sort(E), manifold_tol)
            length(m) > 1 || continue
            ids = sortperm(E)[m]
            ys = [y[l] for l in ids]
            poly!(ax, Rect2f(x - 0.06, minimum(ys) - 0.3, L + 0.12, maximum(ys) - minimum(ys) + 0.85);
                  color = (_GRID, 0.7), strokewidth = 0)
        end
        for (l, r) in enumerate(cb.levels[k])
            yl = y[l]
            lines!(ax, [x, x + L], [yl, yl]; color = _INK, linewidth = 2.5)
            lab = _level_label(cb, k, l)
            scatter!(ax, [x - 0.03], [yl]; markersize = 9, strokewidth = 0,
                     color = irrep_color(lab, cb.irreps))
            deg = length(r)
            Eabs = round(E[l]; digits = 6) + 0.0                 # no "-0.0000"
            en = l == 1 ? @sprintf("%.4f eV", Eabs) : _fmt_shift(E[l] - E[1])
            τ = _fmt_tau_range(cb.lifetime[k][r])
            txt = string(lab, deg > 1 ? " ×$deg" : "", "
", en,
                         isempty(τ) ? "" : " · " * τ)
            text!(ax, x, yl + 0.08; text = txt, align = (:left, :bottom), fontsize = 8,
                  color = _MUTED)
        end
        if others[k]
            yo = y[end]
            lines!(ax, [x, x + L], [yo, yo]; color = UNLABELLED_COLOR, linewidth = 2,
                   linestyle = :dash)
            text!(ax, x, yo + 0.08; align = (:left, :bottom), fontsize = 8, color = _MUTED,
                  text = @sprintf("others: %.2g%% of the flow", 100 * unshown[k-1]))
        end
    end
    # the sector names above their columns
    for k in 1:K
        text!(ax, k - 1 + L / 2, ymax + 0.9; text = cb.names[k], align = (:center, :bottom),
              fontsize = 18, color = _INK)
    end
    drawn = filter(s -> s.y >= min_yield, lines_all)
    drawn = sort(drawn; by = s -> -s.y)[1:min(max_arrows, length(drawn))]
    labelled = Set{Any}()
    for k in 1:K-1
        dk = filter(s -> s.k == k, drawn)
        foreach(s -> push!(labelled, (s.k, s.li, s.lf)), dk[1:min(label_arrows, length(dk))])
    end
    for s in reverse(drawn)                                      # strongest on top
        (xa, ya), (xb, yb) = pos[s.k][s.li], pos[s.k+1][s.lf]
        x0, y0, x1, y1 = xa + L + 0.02, ya, xb - 0.1, yb
        c = _POL_GRAD[clamp(s.axial, 0, 1)]
        lw = 0.8 + 9 * s.y
        lines!(ax, [x0, x1], [y0, y1]; color = (c, 0.35 + 0.5 * min(1, 10s.y)), linewidth = lw)
        θ = atan(y1 - y0, x1 - x0)
        scatter!(ax, [x1], [y1]; marker = :utriangle, rotation = θ - π / 2,
                 markersize = 8 + 2 * lw, color = c, strokewidth = 0)
        if (s.k, s.li, s.lf) in labelled
            # under the level at the end that is unique to this line: the target when
            # its sector shows at least as many levels as the source, else the source
            at_target = length(cb.levels[s.k+1]) >= length(cb.levels[s.k])
            t = @sprintf("%s %.4f eV · %.3g%%", at_target ? "in" : "out", s.ħω, 100 * s.y)
            xl, yl = at_target ? (xb, yb) : (xa, ya)
            # "in" on the first line under the level, "out" on the second
            text!(ax, xl, yl - 0.07; text = t, align = (:left, :top), fontsize = 8, color = c,
                  offset = (0, at_target ? 0 : -10))
        end
    end
    xlims!(ax, -0.15, K - 1 + L + 0.15)
    ylims!(ax, -ymax - 0.7, ymax + 1.5)
    return ax
end

function _channel_spectrum!(pos_, cb, shown, keys, key, pivot; min_yield, scale)
    K = length(cb.names)
    gl = GridLayout(pos_)
    if isempty(shown)
        ax = Axis(gl[1, 1]; title = "emission spectrum")
        text!(ax, 0.5, 0.5; text = "no line above min_yield", space = :relative,
              align = (:center, :center), color = _MUTED)
        return gl
    end
    # x: meV from the strongest line of the last step (the X line)
    fin = filter(s -> s.k == K - 1, shown)
    ref = isempty(fin) ? shown[argmax([s.y for s in shown])].ħω :
          fin[argmax([s.y for s in fin])].ħω
    # heights: log decades above the floor, or linear
    lo = log10(min_yield) - 0.5          # so a line at min_yield still has a length
    h(y) = scale === :log ? log10(y) - lo : y
    top = maximum(h(s.y) for s in shown)
    if scale === :log
        d0 = ceil(Int, lo)
        decs = reverse(0:-max(1, cld(-d0, 3)):d0)
        decs = filter(d -> d - lo > 0.75, collect(decs))
        tv = [d - lo for d in decs]
        tl = ["1e$d" for d in decs]
        tl[end] = "1"
    end
    xs = [1e3 * (s.ħω - ref) for s in shown]
    xr = extrema(xs)
    pad = max(0.08 * (xr[2] - xr[1]), 0.5)
    axs = Axis[]
    for (r, kname) in enumerate(keys)
        ax = Axis(gl[r, 2]; xgridvisible = false,
                  title = r == 1 ? "emission spectrum by channel" : "",
                  xlabel = r == length(keys) ? @sprintf("photon energy − %.4f eV (meV)", ref) : "")
        push!(axs, ax)
        hlines!(ax, [0.0]; color = _MUTED, linewidth = 0.8)
        for s in filter(s -> key(s) == kname, shown)
            sgn = s.k == K - 1 ? 1.0 : -1.0          # last step up, earlier steps down
            c = _POL_GRAD[clamp(s.axial, 0, 1)]
            x = 1e3 * (s.ħω - ref)
            lines!(ax, [x, x], [0, sgn * h(s.y)]; color = c, linewidth = 2.2)
            scatter!(ax, [x], [sgn * h(s.y)]; color = c, markersize = 6,
                     marker = s.k == K - 1 ? :rect : :circle)
        end
        ylims!(ax, -1.12top, 1.12top)
        if scale === :log
            ax.yticks = (vcat(-reverse(tv), tv), vcat(reverse(tl), tl))
        end
        xlims!(ax, xr[1] - pad, xr[2] + pad)
        r < length(keys) && hidexdecorations!(ax; grid = false)
        # the channel's name, with its irrep mark (text stays in ink)
        if !isempty(kname)
            Label(gl[r, 1], rich(rich("● "; color = irrep_color(kname, cb.irreps)),
                                 rich(kname; color = _INK));
                  rotation = 0, fontsize = 12, tellheight = false)
        end
        if r == 1
            up = K >= 2 ? cb.names[K-1] * " → " * cb.names[K] : ""
            dn = K >= 3 ? join(cb.names[1:K-1], " → ") : ""
            text!(ax, 0.01, 0.97; space = :relative, align = (:left, :top), fontsize = 9,
                  color = _MUTED, text = "↑ " * up * (isempty(dn) ? "" : "   ↓ " * dn))
        end
    end
    linkxaxes!(axs...)
    Label(gl[:, 3], "photons per cascade" * (scale === :log ? " (log)" : "");
          rotation = π / 2, fontsize = 11, tellheight = false)
    split_note = pivot <= K && !isempty(first(keys)) ?
        "rows: irrep of the " * cb.names[pivot] * " level" : ""
    isempty(split_note) || Label(gl[length(keys)+1, 2], split_note; fontsize = 9,
                                 color = _MUTED, tellwidth = false)
    rowgap!(gl, 4)
    return gl
end
