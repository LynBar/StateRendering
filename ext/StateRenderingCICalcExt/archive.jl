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
    plot_sector(res::CICalcResults, name; nlevels = 24, size = (640, 620), title = nothing)
        -> Figure

The lowest `nlevels` levels of a stored sector, by irrep. When the archive holds rates
out of this sector (e.g. "X→0" for X), each level shows its radiative lifetime, and its
line width and colour show its brightness and axial share. Levels beyond the stored
rates (those whose vectors were not kept) are dashed. A level whose states differ in
brightness (a merged fine-structure manifold) shows the range of their lifetimes and
is drawn by its brightest state.
"""
function plot_sector(res::CICalcResults, name::AbstractString; nlevels::Integer = 24,
                     size = (640, 620), title = nothing)
    s = _pick(res.sectors, String(name), "sector",
              "Add it to the run spec's `compute: sectors:`.")
    r = _rates_from(res, s)
    irreps = res.orbitals === nothing || !haskey(res.orbitals, :table) ? String[] :
             collect(String, res.orbitals.table.labels)
    L = s.levels[1:min(nlevels, length(s.levels))]
    E0 = s.energies[1]
    nr = r === nothing ? 0 : Base.size(r.Γ, 2)
    # per level: the states' total rates (a merged fine-structure level can mix bright
    # and dark states, so the range is kept), the brightest for the width, the axial share
    srates = map(L) do l
        all(i -> i <= nr, l) || return Float64[]
        [sum(r.Γ[:, i]) for i in l]
    end
    rate = [isempty(v) ? NaN : maximum(v) for v in srates]
    ax_share = map(L) do l
        all(i -> i <= nr, l) || return NaN
        t = sum(sum(r.Γ[:, i]) for i in l)
        t > 0 ? sum(sum(r.Γpol[3][:, i]) for i in l) / t : 0.0
    end
    top = maximum(filter(isfinite, rate); init = 0.0)

    fig = Figure(size = size)
    ttl = title === nothing ?
        @sprintf("%s %s: lowest %d of %d levels", s.name, string(s.sector), length(L),
                 length(s.levels)) : title
    ax = Axis(fig[1, 1]; title = ttl, ylabel = "E − E₁ (meV)")
    hidexdecorations!(ax)
    ys, txts = Float64[], String[]
    for (j, l) in enumerate(L)
        y = 1e3 * (s.energies[first(l)] - E0)
        lab = isempty(s.labels) || isempty(s.labels[j]) ? "?" : s.labels[j]
        if isfinite(rate[j]) && top > 0
            f = rate[j] / top
            # brightness on a log scale: 1e-6 of the brightest and below is thin
            w = clamp(1 + log10(max(f, 1e-6)) / 6, 0, 1)
            lines!(ax, [0.0, 1.0], [y, y]; linewidth = 1 + 5w,
                   color = _POL_GRAD[clamp(ax_share[j], 0, 1)])
        else
            lines!(ax, [0.0, 1.0], [y, y]; linewidth = 1.2, linestyle = :dash,
                   color = UNLABELLED_COLOR)
        end
        scatter!(ax, [-0.08], [y]; markersize = 9, strokewidth = 0,
                 color = irrep_color(lab, irreps))
        τ = if !isfinite(rate[j]) || rate[j] <= 0
            ""
        else
            lo, hi = extrema(srates[j])
            hi <= 1.01lo ? "  " * _fmt_tau(1 / hi) :
                "  " * _fmt_tau(1 / hi) * " … " * (lo > 0 ? _fmt_tau(1 / lo)[4:end] : "∞")
        end
        push!(ys, y)
        push!(txts, string(lab, length(l) > 1 ? " ×$(length(l))" : "", "  ",
                           j == 1 ? @sprintf("%.4f eV", E0) : _fmt_shift(s.energies[first(l)] - E0), τ))
    end
    span = isempty(ys) ? 1.0 : max(maximum(ys) - minimum(ys), 1e-9)
    for (y, yt, t) in zip(ys, _repel(ys, 0.028 * span), txts)
        abs(yt - y) > 1e-9 * span && lines!(ax, [1.02, 1.1], [y, yt]; color = _GRID, linewidth = 0.8)
        text!(ax, 1.12, yt; text = t, align = (:left, :center), fontsize = 10, color = _INK)
    end
    xlims!(ax, -0.2, 2.4)
    if r !== nothing && top > 0
        Colorbar(fig[1, 2]; colormap = _POL_GRAD, limits = (0, 1), width = 10,
                 label = "axial share of the emission (" * r.polnames[3] * ")")
        Label(fig[2, 1:2], @sprintf("width: radiative rate to %s on a log scale (1e-6 of the brightest and below is thinnest); dashed: no stored rate",
                                     string(r.final)); fontsize = 9, color = _MUTED, tellwidth = false)
    end
    return fig
end

_fname(s) = replace(string(s), r"[^A-Za-z0-9_.\-]+" => "_")

"""
    save_figures(res::CICalcResults; out = "figures", format = "png", verbose = true)
        -> Vector{String}

Every figure the archive supports, written to `out`: the labelling figures (from
`res.levels`), per sector its level diagram (`plot_sector`), its ground state's
configurations and the densities of every state that has them, one figure per stored
cascade, and the lowest electron and hole orbitals of the render tier. A figure the archive cannot support is skipped, and the reason is
printed. Returns the paths written. Needs a Makie backend that can save `format`
(CairoMakie, or GLMakie for png).
"""
function save_figures(res::CICalcResults; out::AbstractString = "figures",
                      format::AbstractString = "png", verbose::Bool = true)
    mkpath(out)
    done, skipped = String[], Pair{String,String}[]
    function one(name, build)
        try
            p = joinpath(out, _fname(name) * "." * format)
            save(p, build())
            push!(done, p)
        catch e
            e isa ArgumentError || rethrow()
            push!(skipped, name => sprint(showerror, e))
        end
    end
    if res.levels !== nothing
        lb = res.levels
        one("levels", () -> plot_levels(lb))
        one("weights_state", () -> plot_irrep_weights(lb))
        one("weights_level", () -> plot_irrep_weights(lb; per = :level))
        # broken = above ten times the worst method floor, when the floor was computed
        fl = filter(isfinite, lb.floor)
        one("operation_map", () -> plot_operation_map(lb; tol = max(1e-6, 10 * maximum(fl; init = 0.0))))
        one("quality", () -> plot_symmetry_quality(lb))
        one("charges", () -> plot_charges(lb))
        isempty(lb.fields) || one("fields", () -> plot_field_irreps(lb))
        lb.parent === nothing || one("parent", () -> plot_parent_mixing(lb))
    else
        push!(skipped, "labelling figures" => "no states block stored")
    end
    for n in sort!(collect(keys(res.sectors)); by = n -> (length(res.sectors[n]), n))
        s = res.sectors[n]
        one("sector_" * n, () -> plot_sector(res, n))
        # the ground state's make-up, and every state whose densities were stored
        s.nkept >= 1 && length(s.basis) > 1 &&
            one("configurations_$(n)1", () -> plot_configurations(res, n, 1))
        for i in sort!(collect(keys(s.densities)))
            one("densities_$(n)$i", () -> plot_densities(res, n, i))
        end
    end
    for n in sort!(collect(keys(res.cascades)))
        one("cascade_" * n, () -> plot_cascade(res.cascades[n]))
    end
    # the lowest electron and hole orbitals of the render tier
    ro = get(res.extra, "render/orbitals", nothing)
    if ro !== nothing
        for kind in (:e, :h)
            k = findfirst(==(kind), ro.kind)
            k === nothing || one("orbital_$k", () -> plot_orbital(res, k))
        end
    end
    if verbose
        println("save_figures: wrote ", length(done), " figure(s) to ", out)
        for (n, why) in skipped
            println("  skipped ", n, ": ", why)
        end
    end
    return done
end
