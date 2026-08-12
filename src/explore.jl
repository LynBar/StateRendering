# ---------------------------------------------------------------------------
# The one interactive figure.
#
# The old package had four near-identical builders (PlotΨ, PlotElectron,
# PlotHole, _plot_state) that differed only in which band labels they hard-coded
# — which is precisely the thing that must not be hard-coded, since the labels
# depend on the operator and not on the band count. There is now one, and it
# reads its labels from the bundle.
#
# Interaction needs a GPU backend (GLMakie or WGLMakie). Everything in
# figures.jl works everywhere, CairoMakie included.
# ---------------------------------------------------------------------------

"""
    explore(b::StateBundle; fields = ScalarField[], strain = nothing,
            vectors = nothing, fraction = 0.5) -> Figure

Interactive browser for a solved bundle:

  * a state menu, labelled with energy and dominant orbital component;
  * one checkbox per spinor band, **named and coloured from the bundle**, so the
    density can be restricted to any subset of bands;
  * an isosurface slider in units of *enclosed probability*, not of peak height,
    so the surface means the same thing for every state;
  * live component weights;
  * toggles for the dot outline, any scalar overlays (`fields`, or a
    [`StrainField`](@ref) via `strain`, which expands to its invariants and
    components), and a [`VectorField`](@ref) via `vectors`.

    using GLMakie, StateRendering
    explore(bundle; strain = ε, vectors = P)

Needs GLMakie or WGLMakie loaded; under CairoMakie the figure renders but the
widgets do not respond.
"""
function explore(b::StateBundle;
                 fields = ScalarField[], strain = nothing, vectors = nothing,
                 fraction::Real = 0.5, size = (1180, 730))
    _warn_static_backend()

    overlays = ScalarField[fields...]
    strain === nothing || append!(overlays, scalar_fields(strain))
    for f in overlays
        _check_grid(f, b, "field \"$(f.name)\"")
    end
    vectors === nothing || _check_grid(vectors, b, "vector field \"$(vectors.name)\"")

    fig = Figure(size = size)

    # ── state selection ────────────────────────────────────────────────────
    state_labels = [@sprintf("%s%d   %.4f eV   %s",
                             b.kind[i] === :hole ? "h" : "e", i, b.energies[i],
                             dominant_component(b, i)[1]) for i in 1:length(b)]
    menu = Menu(fig, options = zip(state_labels, 1:length(b)), tellwidth = false)
    fig[1, 1] = hgrid!(Label(fig, "State", width = nothing), menu)

    gl = GridLayout(fig[2, 1])
    left = GridLayout(gl[1, 1], tellheight = false)
    ax = state_axis3(gl[1, 2])
    right = GridLayout(gl[1, 3], tellheight = false)
    colsize!(gl, 2, Auto(3))

    state_idx = Observable(1)
    on(v -> (v === nothing || (state_idx[] = v)), menu.selection)

    # ── band selection, labelled and coloured from the bundle ──────────────
    bcols = band_palette(b)
    boxes = Checkbox[]
    for (k, lbl) in pairs(b.band_labels)
        push!(boxes, Checkbox(left[k, 1], checked = true,
                              checkboxcolor_checked = bcols[k],
                              checkmarkcolor_checked = :white))
        Label(left[k, 2], lbl, halign = :left, color = bcols[k])
    end
    colgap!(left, 8); rowgap!(left, 3)
    active = lift((cb.checked for cb in boxes)...) do vals...
        collect(Bool, vals)
    end

    dens = lift(state_idx, active) do i, sel
        any(sel) ? state_density(b, i; bands = sel) :
                   zeros(length(b.x), length(b.y), length(b.z))
    end

    # ── density isosurface, in enclosed-probability units ──────────────────
    dctl = GridLayout(right[1, 1])
    Label(dctl[1, 1], "Ψ enclosed", halign = :left)
    frac_sl = Slider(dctl[1, 2], range = 0.05:0.05:0.95, startvalue = fraction,
                     update_while_dragging = false, horizontal = true)
    Label(dctl[1, 3], lift(f -> @sprintf("%.0f%%", 100f), frac_sl.value), halign = :left)
    dens_cb = Checkbox(dctl[1, 4], checked = true)

    dmesh = lift(dens, frac_sl.value) do d, f
        sum(d) > 0 || return _empty_mesh()
        _mesh_or_empty(isosurface_mesh(d, b.x, b.y, b.z, isolevel(d, f)))
    end
    dcolor = lift(i -> (_density_color(b, i, :auto), 0.6), state_idx)
    mesh!(ax, dmesh; color = dcolor, transparency = true, visible = dens_cb.checked)

    # ── geometry ───────────────────────────────────────────────────────────
    row = 2
    if b.geometry !== nothing
        gctl = GridLayout(right[row, 1]); row += 1
        Label(gctl[1, 1], "geometry", halign = :left)
        galpha = Slider(gctl[1, 2], range = 0.0:0.02:0.6, startvalue = 0.12,
                        update_while_dragging = true, horizontal = true)
        gcb = Checkbox(gctl[1, 3], checked = true)
        gmesh = isosurface_mesh(smooth(b.geometry; σ = 1.0), b.x, b.y, b.z, 0.5)
        gmesh === nothing || mesh!(ax, gmesh; color = lift(a -> (:grey, a), galpha.value),
                                   transparency = true, visible = gcb.checked)
    end

    # ── scalar overlays ────────────────────────────────────────────────────
    if !isempty(overlays)
        row = _add_overlay_controls!(fig, ax, right, row, b, overlays)
    end

    # ── vector overlay ─────────────────────────────────────────────────────
    if vectors !== nothing
        row = _add_vector_controls!(fig, ax, right, row, b, vectors)
    end

    # ── live component weights ─────────────────────────────────────────────
    bax = Axis(left[length(b.band_labels) + 1, 1:2]; title = "components",
               titlesize = 12, height = 130)
    cols = component_palette(b.component_labels)
    nc = length(cols)
    wobs = lift(i -> component_weights(b, i), state_idx)
    barplot!(bax, 1:nc, wobs; color = cols, direction = :x)
    text!(bax, lift(w -> Point2f.(w .+ 0.02, 1:nc), wobs);
          text = lift(w -> [@sprintf("%.3f", v) for v in w], wobs),
          align = (:left, :center), fontsize = 10)
    bax.yticks = (1:nc, b.component_labels)
    xlims!(bax, 0, 1.2)
    hidespines!(bax, :t, :r)

    notify(menu.selection)
    return fig
end

function _add_overlay_controls!(fig, ax, right, row, b::StateBundle, overlays)
    ctl = GridLayout(right[row, 1])
    names = [f.name for f in overlays]
    Label(ctl[1, 1], "field", halign = :left)
    fmenu = Menu(fig, options = zip(names, 1:length(overlays)), tellwidth = false)
    ctl[1, 2] = fmenu
    mmenu = Menu(fig, options = ["iso", "volume"], tellwidth = false)
    ctl[1, 3] = mmenu
    cb = Checkbox(ctl[1, 4], checked = false)
    lvl = Slider(ctl[2, 1:4], range = 0.05:0.05:1.0, startvalue = 0.5,
                 update_while_dragging = false, horizontal = true)

    sel = Observable(1)
    on(v -> (v === nothing || (sel[] = v)), fmenu.selection)

    # normalised to peak magnitude so one slider spans every field in the menu
    normed = lift(sel) do s
        f = overlays[s]
        p = maximum(abs, f.data)
        p > 0 ? f.data ./ p : f.data
    end
    signed = lift(s -> overlays[s].signed, sel)

    iso_on = lift((m, v) -> v && m == "iso", mmenu.selection, cb.checked)
    vol_on = lift((m, v) -> v && m == "volume", mmenu.selection, cb.checked)

    for (sgn, colour) in ((1, :red), (-1, :blue))
        m = lift(normed, lvl.value, signed) do n, l, sg
            (sgn < 0 && !sg) && return _empty_mesh()
            _mesh_or_empty(isosurface_mesh(n, b.x, b.y, b.z, sgn * l))
        end
        mesh!(ax, m; color = (colour, 0.3), transparency = true, visible = iso_on)
    end
    volume!(ax, extrema(b.x), extrema(b.y), extrema(b.z), normed;
            algorithm = :absorption, absorption = 2.0f0,
            colormap = diverging_map(0.4), colorrange = (-1.0, 1.0),
            transparency = true, visible = vol_on)
    return row + 1
end

function _add_vector_controls!(fig, ax, right, row, b::StateBundle, v::VectorField)
    ctl = GridLayout(right[row, 1])
    Label(ctl[1, 1], v.name, halign = :left)
    pmenu = Menu(fig, options = zip(["x cut", "y cut", "z cut"], [:x, :y, :z]),
                 default = "z cut", tellwidth = false)
    ctl[1, 2] = pmenu
    cb = Checkbox(ctl[1, 3], checked = false)
    # fractional position, so the slider needs no rescaling when the plane changes
    pos = Slider(ctl[2, 1:3], range = 0:0.02:1, startvalue = 0.5,
                 update_while_dragging = true, horizontal = true)

    plane = Observable(:z)
    on(p -> (p === nothing || (plane[] = p)), pmenu.selection)

    coords = (b.x, b.y, b.z)
    mag = magnitude(v)
    peak = max(maximum(mag), eps())
    step = 4
    ls = Float32(step * minimum(c -> length(c) > 1 ? c[2] - c[1] : 1.0, coords) * 2)

    data = lift(plane, pos.value) do p, t
        a = _axis_of(p)
        idx = clamp(round(Int, 1 + t * (length(coords[a]) - 1)), 1, length(coords[a]))
        ranges = [1:step:length(coords[d]) for d in 1:3]
        ranges[a] = idx:idx
        pts = Point3f[]; dirs = Vec3f[]; cs = Float64[]
        for k in ranges[3], j in ranges[2], i in ranges[1]
            push!(pts, Point3f(b.x[i], b.y[j], b.z[k]))
            push!(dirs, Vec3f(v.x[i, j, k], v.y[i, j, k], v.z[i, j, k]) / peak)
            push!(cs, mag[i, j, k])
        end
        (pts, dirs, cs)
    end
    arrows3d!(ax, lift(first, data), lift(d -> d[2], data);
              lengthscale = ls, color = lift(d -> d[3], data),
              colormap = :viridis, colorrange = (0.0, peak), visible = cb.checked)
    return row + 1
end

function _warn_static_backend()
    be = string(Makie.current_backend())
    occursin("Cairo", be) && @warn """
        The active Makie backend is $be, which draws a static image: `explore`'s \
        menus and sliders will not respond. Load GLMakie for interaction, or use \
        the static figures (plot_state, plot_slices, plot_bands, ...) which are \
        designed for CairoMakie."""
    return nothing
end
