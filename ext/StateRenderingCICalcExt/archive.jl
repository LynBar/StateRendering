# ---------------------------------------------------------------------------
# Figures from a results archive.
#
# The presentation stage: everything here reads a `CICalc.CICalcResults`, whether it
# was built in memory or loaded with `load_results`, and draws from what it stores.
# Nothing here recomputes physics. A figure that needs a product the archive does not
# hold throws an `ArgumentError` naming the product to add to the run spec. Cheap
# derived numbers (a level's lifetime and brightness from stored rates) are fine.
#
# The figure functions keep taking their bundles; the archive methods only pick the
# bundle out.
# ---------------------------------------------------------------------------

_missing_product(what, how) = throw(ArgumentError("this archive has no $what. $how"))

function _archive_levels(res::CICalcResults)
    res.levels === nothing && _missing_product("states block",
        "Store the states (`store!(res, ks, analysis)`); every run spec does.")
    return res.levels
end

for f in (:plot_levels, :plot_irrep_weights, :plot_operation_map, :plot_symmetry_quality,
          :plot_charges, :plot_parent_mixing, :plot_field_irreps)
    @eval $f(res::CICalcResults; kwargs...) = $f(_archive_levels(res); kwargs...)
end

function _pick(d::AbstractDict, name, what, how)
    isempty(d) && _missing_product(what, how)
    if name === nothing
        length(d) == 1 && return only(values(d))
        throw(ArgumentError("this archive has several of $what: " *
                            join(sort!(collect(keys(d))), ", ") * "; pass one by name"))
    end
    haskey(d, name) || throw(ArgumentError("no $what named \"$name\"; the archive has " *
                                           join(sort!(collect(keys(d))), ", ")))
    return d[name]
end

"""
    plot_cascade(res::CICalcResults, name = nothing; kwargs...) -> Figure

The stored cascade `name` (the only one when `nothing`), drawn by
`plot_cascade(::CascadeBundle)`.
"""
plot_cascade(res::CICalcResults, name = nothing; kwargs...) =
    plot_cascade(_pick(res.cascades, name, "cascade",
                       "Add `cascade:` to the run spec's `compute` (or store a " *
                       "`cascade_bundle`)."); kwargs...)

# the stored rates out of a sector, if any: initial = this sector, final one pair lower
function _rates_from(res::CICalcResults, s::StoredSector)
    for k in sort!(collect(keys(res.rates)))
        r = res.rates[k]
        r.initial == s.sector && r.final == s.sector .- 1 && return r
    end
    return nothing
end

"""
    plot_sector(res::CICalcResults, name; nlevels = 24, by_irrep = true, size, title)
        -> Figure

The lowest `nlevels` levels of a stored sector.

With `by_irrep` (the default) the spectrum has one column per irrep, in character-table
order, with "?" last. Each state goes in the column of its own irrep
(`state_labels`), so a level that the energy tolerance merged, such as `Γ5 ⊕ Γ6 ×4`,
shows as its `Γ5 ×2` and `Γ6 ×2` side by side, each with its own energy and lifetime.
`by_irrep = false` draws every level in one column, labelled by its content.

When the archive holds rates out of this sector (e.g. "X→0" for X), each line shows
its radiative lifetime, and its width and colour show its brightness and axial share.
Lines beyond the stored rates (states whose vectors were not kept) are dashed. A line
whose states differ in brightness shows the range of their lifetimes and is drawn by
its brightest state.
"""
function plot_sector(res::CICalcResults, name::AbstractString; nlevels::Integer = 24,
                     by_irrep::Bool = true, size = nothing, title = nothing)
    s = _pick(res.sectors, String(name), "sector",
              "Add it to the run spec's `compute: sectors:`.")
    r = _rates_from(res, s)
    irreps = res.orbitals === nothing || !haskey(res.orbitals, :table) ? String[] :
             collect(String, res.orbitals.table.labels)
    L = s.levels[1:min(nlevels, length(s.levels))]
    E0 = s.energies[1]
    nr = r === nothing ? 0 : Base.size(r.Γ, 2)
    rate(i) = i <= nr ? sum(r.Γ[:, i]) : NaN
    llab(j) = isempty(s.labels) || isempty(s.labels[j]) ? "?" : s.labels[j]
    slab(i, j) = isempty(s.state_labels) || isempty(s.state_labels[i]) ? llab(j) : s.state_labels[i]

    # the lines: (column label, states), one per level, or per irrep within a level
    lines_ = Tuple{String,Vector{Int}}[]
    for (j, l) in enumerate(L)
        if by_irrep
            for lab in unique(slab(i, j) for i in l)
                push!(lines_, (lab, [i for i in l if slab(i, j) == lab]))
            end
        else
            push!(lines_, (llab(j), collect(l)))
        end
    end
    # per line: the states' total rates (the range is kept), the brightest for the
    # width, the axial share of the emission
    srates = [all(i -> i <= nr, st) ? rate.(st) : Float64[] for (_, st) in lines_]
    bright = [isempty(v) ? NaN : maximum(v) for v in srates]
    ax_share = map(lines_) do (_, st)
        all(i -> i <= nr, st) || return NaN
        t = sum(rate, st)
        t > 0 ? sum(sum(r.Γpol[3][:, i]) for i in st) / t : 0.0
    end
    top = maximum(filter(isfinite, bright); init = 0.0)
    y = [1e3 * (sum(s.energies[st]) / length(st) - E0) for (_, st) in lines_]

    # columns: the irreps in table order, then compound or unknown labels, "?" last
    labs = unique(first.(lines_))
    rank(lab) = lab == "?" ? (3, 0) :
                (k = findfirst(==(lab), irreps); k === nothing ? (2, findfirst(==(lab), labs)) : (1, k))
    cols = by_irrep ? sort(labs; by = rank) : ["all"]
    col(lab) = by_irrep ? findfirst(==(lab), cols) : 1
    nc = length(cols)
    W = by_irrep ? 1.0 : 2.4                       # column width, in axis units

    fig = Figure(size = something(size, by_irrep ? (150 + 300nc, 680) : (640, 620)))
    ttl = title === nothing ?
        @sprintf("%s %s: lowest %d of %d levels%s", s.name, string(s.sector), length(L),
                 length(s.levels), by_irrep ? ", by irrep" : "") : title
    ax = Axis(fig[1, 1]; title = ttl, ylabel = "E − E₁ (meV)", xgridvisible = false,
              ygridcolor = _GRID)
    if by_irrep
        ax.xticks = ((0:nc-1) .* W .+ 0.11, cols)
        ax.xticklabelcolor = _INK
        hidespines!(ax, :t, :r)
        ax.xticksvisible = false
    else
        hidexdecorations!(ax)
    end
    span = isempty(y) ? 1.0 : max(maximum(y) - minimum(y), 1e-9)
    for c in 1:nc
        x0 = (c - 1) * W
        idx = [k for k in eachindex(lines_) if col(first(lines_[k])) == c]
        by_irrep && c > 1 && vlines!(ax, x0 - 0.06; color = _GRID, linewidth = 0.8)
        for k in idx
            lab, st = lines_[k]
            if isfinite(bright[k]) && top > 0
                # brightness on a log scale: 1e-6 of the brightest and below is thin
                w = clamp(1 + log10(max(bright[k] / top, 1e-6)) / 6, 0, 1)
                lines!(ax, [x0, x0 + 0.22W], [y[k], y[k]]; linewidth = 1 + 5w,
                       color = _POL_GRAD[clamp(ax_share[k], 0, 1)])
            else
                lines!(ax, [x0, x0 + 0.22W], [y[k], y[k]]; linewidth = 1.2, linestyle = :dash,
                       color = UNLABELLED_COLOR)
            end
            scatter!(ax, [x0 - 0.03W], [y[k]]; markersize = 8, strokewidth = 0,
                     color = irrep_color(lab, irreps))
        end
        # the labels, kept apart within the column
        txt = map(idx) do k
            lab, st = lines_[k]
            τ = isempty(srates[k]) || !(bright[k] > 0) ? "" :
                "  " * _fmt_tau_range([v > 0 ? 1 / v : Inf for v in srates[k]])
            dE = 1 in st ? @sprintf("%.4f eV", E0) : _fmt_shift(1e-3 * y[k])
            # a pair the symmetry would keep degenerate, split by its breaking
            lo, hi = extrema(s.energies[st])
            sp = hi - lo > 1e-7 ? " (split " * _fmt_shift(hi - lo)[2:end] * ")" : ""
            string(by_irrep ? "" : lab * " ", length(st) > 1 ? "×$(length(st))  " : "", dE, sp, τ)
        end
        yt = _repel(y[idx], 0.028 * span)
        for (k, yk, t) in zip(idx, yt, txt)
            abs(yk - y[k]) > 1e-9 * span &&
                lines!(ax, [x0 + 0.23W, x0 + 0.27W], [y[k], yk]; color = _GRID, linewidth = 0.8)
            text!(ax, x0 + 0.28W, yk; text = t, align = (:left, :center), fontsize = 9, color = _INK)
        end
    end
    xlims!(ax, -0.12, nc * W - 0.06)
    if r !== nothing && top > 0
        Colorbar(fig[1, 2]; colormap = _POL_GRAD, limits = (0, 1), width = 10,
                 label = "axial share of the emission (" * r.polnames[3] * ")")
        Label(fig[2, 1:2], @sprintf("width: radiative rate to %s on a log scale (1e-6 of the brightest and below is thinnest); dashed: no stored rate",
                                     string(r.final)); fontsize = 9, color = _MUTED, tellwidth = false)
    end
    return fig
end

# a file or folder name: letters, digits and _ . - + kept ("X+", "XX→X→0" → "XX_X_0")
_fname(s) = replace(string(s), r"[^A-Za-z0-9_.+\-]+" => "_")

"""
    save_figures(res::CICalcResults; out = "figures", format = "png", verbose = true)
        -> Vector{String}

Every figure the archive supports, written under `out`, one folder per kind:

    labels/     levels, weights_state, weights_level, operation_map, quality, charges,
                fields, parent                       (the single-particle labelling)
    orbitals/   orbital_<k>: the lowest electron and hole orbital of the render tier
    sectors/<name>/
                levels           the sector's levels by irrep, with lifetimes
                configurations_1 the ground state's make-up: a biexciton in excitons, a
                                 trion as electrons ⊗ holes
                configurations_1_species  a biexciton as electron pair ⊗ hole pair
                densities_<i>    each state whose densities were stored
    cascades/   one figure per stored cascade

A figure the archive cannot support is skipped, and the reason is printed. Returns the
paths written. Needs a Makie backend that can save `format` (CairoMakie, or GLMakie
for png).
"""
function save_figures(res::CICalcResults; out::AbstractString = "figures",
                      format::AbstractString = "png", verbose::Bool = true)
    mkpath(out)
    done, skipped = String[], Pair{String,String}[]
    # `path` is folder/…/name, each part made a safe file name
    function one(path, build)
        try
            parts = _fname.(split(path, '/'))
            dir = mkpath(joinpath(out, parts[1:end-1]...))
            p = joinpath(dir, parts[end] * "." * format)
            save(p, build())
            push!(done, p)
        catch e
            e isa ArgumentError || rethrow()
            push!(skipped, path => sprint(showerror, e))
        end
    end
    if res.levels !== nothing
        lb = res.levels
        one("labels/levels", () -> plot_levels(lb))
        one("labels/weights_state", () -> plot_irrep_weights(lb))
        one("labels/weights_level", () -> plot_irrep_weights(lb; per = :level))
        # broken = above ten times the worst method floor, when the floor was computed
        fl = filter(isfinite, lb.floor)
        one("labels/operation_map",
            () -> plot_operation_map(lb; tol = max(1e-6, 10 * maximum(fl; init = 0.0))))
        one("labels/quality", () -> plot_symmetry_quality(lb))
        one("labels/charges", () -> plot_charges(lb))
        isempty(lb.fields) || one("labels/fields", () -> plot_field_irreps(lb))
        lb.parent === nothing || one("labels/parent", () -> plot_parent_mixing(lb))
    else
        push!(skipped, "labels" => "no states block stored")
    end
    # the lowest electron and hole orbitals of the render tier
    ro = get(res.extra, "render/orbitals", nothing)
    if ro !== nothing
        for kind in (:e, :h)
            k = findfirst(==(kind), ro.kind)
            k === nothing || one("orbitals/orbital_$k", () -> plot_orbital(res, k))
        end
    end
    for n in sort!(collect(keys(res.sectors)); by = n -> (length(res.sectors[n]), n))
        s = res.sectors[n]
        one("sectors/$n/levels", () -> plot_sector(res, n))
        # the ground state's make-up, and every state whose densities were stored
        s.nkept >= 1 && length(s.basis) > 1 &&
            one("sectors/$n/configurations_1", () -> plot_configurations(res, n, 1))
        s.sector == (2, 2) && last(s.levels[1]) <= s.nkept &&
            one("sectors/$n/configurations_1_species",
                () -> plot_configurations(res, n, 1; basis = :species))
        for i in sort!(collect(keys(s.densities)))
            one("sectors/$n/densities_$i", () -> plot_densities(res, n, i))
        end
    end
    for n in sort!(collect(keys(res.cascades)))
        one("cascades/" * n, () -> plot_cascade(res.cascades[n]))
    end
    if verbose
        println("save_figures: wrote ", length(done), " figure(s) to ", out)
        for (n, why) in skipped
            println("  skipped ", n, ": ", why)
        end
    end
    return done
end
