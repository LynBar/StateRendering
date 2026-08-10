module StateRendering

using GLMakie, LaTeXStrings, LinearAlgebra
using HDF5

export PlotΨ, PlotH5, PlotElectron, PlotHole, PlotCIΨ

# Pie-component decomposition for a given number of spinor bands. Returns
# (labels, colors, groups) where each group lists the band indices summed into
# one orbital component. Band orderings assumed:
#   2 -> [s↑, s↓]
#   6 -> [pₓ↑, p_y↑, p_z↑, pₓ↓, p_y↓, p_z↓]
#   8 -> [s↑, pₓ↑, p_y↑, p_z↑, s↓, pₓ↓, p_y↓, p_z↓]
function band_decomposition(nbands::Integer)
    nbands == 2 && return ([L"s"], [:red], [[1, 2]])
    nbands == 6 && return ([L"p_x", L"p_y", L"p_z"], [:blue, :green, :black], [[1, 4], [2, 5], [3, 6]])
    nbands == 8 && return ([L"s", L"p_x", L"p_y", L"p_z"], [:red, :blue, :green, :black], [[i, i + 4] for i in 1:4])
    error("No component decomposition defined for $nbands bands")
end

function PlotElectron(Ψes::AbstractVector{<:AbstractArray{Float64, 4}}; energies = nothing)
    band_labels = [L"s\uparrow", L"s\downarrow"]
    comp_labels, comp_colors, comp_groups = band_decomposition(2)
    return _plot_state(Ψes, band_labels, comp_labels, comp_colors, comp_groups; energies, state_prefix = "e")
end

function PlotHole(Ψhs::AbstractVector{<:AbstractArray{Float64, 4}}; energies = nothing)
    band_labels = [L"p_x\uparrow", L"p_y\uparrow", L"p_z\uparrow", L"p_x\downarrow", L"p_y\downarrow", L"p_z\downarrow"]
    comp_labels, comp_colors, comp_groups = band_decomposition(6)
    return _plot_state(Ψhs, band_labels, comp_labels, comp_colors, comp_groups; energies, state_prefix = "h")
end

function ΨDensity(Ψ, bands)
    slices = [Ψb for (Ψb, b) in zip(eachslice(Ψ; dims=1), bands) if b]
    isempty(slices) && return zeros(size(Ψ, 2), size(Ψ, 3), size(Ψ, 4))
    Ψ_ = reduce(+, slices)
    return Ψ_ ./ maximum(Ψ_)
end

function add_band_selector!(band_gl, Ψ_obs, band_labels, comp_groups)
    checkboxes = [Checkbox(band_gl[i, 1], checked = true) for (i, _) in enumerate(band_labels)]
    [Label(band_gl[i, 2], l, halign = :left) for (i, l) in enumerate(band_labels)]
    colgap!(band_gl, 8)
    rowgap!(band_gl, 4)

    bands_active = lift((cb.checked for cb in checkboxes)...) do vals...
        collect(vals)
    end

    density_obs = lift(Ψ_obs, bands_active) do ψ, b
        ΨDensity(ψ, b)
    end

    components_obs = lift(Ψ_obs, bands_active) do ψ, b
        components = [sum(ψi) * active for (active, ψi) in zip(b, eachslice(ψ, dims = 1))]
        return [sum(components[i] for i in group) for group in comp_groups]
    end

    return density_obs, components_obs
end

function gaussian_smooth(A::AbstractArray{<:Real, 3}; σ = 1.0)
    r = ceil(Int, 2σ)
    kernel_1d = [exp(-x^2 / (2σ^2)) for x in -r:r]
    kernel_1d ./= sum(kernel_1d)
    out = copy(Float64.(A))
    for dim in 1:3
        perm = circshift(1:3, 1-dim)
        iperm = invperm(perm)
        P = permutedims(out, perm)
        B = similar(P)
        n = size(P, 1)
        for I in CartesianIndices(size(P)[2:3])
            for i in 1:n
                s = 0.0
                for (ki, k) in enumerate(-r:r)
                    j = clamp(i + k, 1, n)
                    s += kernel_1d[ki] * P[j, I]
                end
                B[i, I] = s
            end
        end
        out .= permutedims(B, iperm)
    end
    return out
end

function add_geometry!(ax, geom_gl, geometry)
    cb = Checkbox(geom_gl[end+1, 1], checked = true)
    Label(geom_gl[end, 2], "Geom", halign = :center)
    geom_controls = GridLayout(geom_gl[end, 3])
    alpha_slider = Slider(geom_controls[1, 1], range = 0.0:0.05:1.0, startvalue = 0.2,
        update_while_dragging = true, horizontal = true)
    Label(geom_controls[1, 2], "α", halign = :center)
    smoothed = gaussian_smooth(geometry)
    volume!(ax, smoothed, algorithm = :iso, alpha = alpha_slider.value, visible = cb.checked, transparency = true)
end

function add_piezo!(ax, geom_gl, piezo)
    piezo ./= maximum(abs, piezo)

    abs_cb = Checkbox(geom_gl[end+1, 1], checked = false)
    Label(geom_gl[end, 2], "Piezo (vol)", halign = :center)
    vol_controls = GridLayout(geom_gl[end, 3])
    alpha_slider = Slider(vol_controls[1, 1], range = 0.0:0.05:1.0, startvalue = 0.4,
        update_while_dragging = true, horizontal = true)
    Label(vol_controls[1, 2], "α", halign = :center)
    absorption_slider = Slider(vol_controls[1, 3], range = 0.0:0.05:4.0, startvalue = 2.0,
        update_while_dragging = true, horizontal = true)
    Label(vol_controls[1, 4], "abs", halign = :center)

    cmap_obs = lift(alpha_slider.value) do a
        [RGBAf(0, 0.3, 1, a), RGBAf(0,0,0,0), RGBAf(1, 0.2, 0, a)]
    end

    volume!(ax, piezo.^3, algorithm = :absorption,
        absorption = absorption_slider.value,
        visible = abs_cb.checked, transparency = true,
        colormap = cmap_obs, colorrange = (-1, 1))

    iso_cb = Checkbox(geom_gl[end+1, 1], checked = false)
    Label(geom_gl[end, 2], "Piezo (iso)", halign = :center)
    iso_controls = GridLayout(geom_gl[end, 3])
    iso_slider = Slider(iso_controls[1, 1], range = 0.1:0.05:1.0, startvalue = 0.5,
        update_while_dragging = true, horizontal = true)
    neg_iso = @lift(-$(iso_slider.value))

    volume!(ax, piezo, algorithm = :iso, alpha = 0.3,
        visible = iso_cb.checked, transparency = true,
        isovalue = iso_slider.value, isorange = 0.05,
        colormap = [:red])
    volume!(ax, piezo, algorithm = :iso, alpha = 0.3,
        visible = iso_cb.checked, transparency = true,
        isovalue = neg_iso, isorange = 0.05,
        colormap = [:blue])
end

function add_polarization!(ax, fig, geom_gl, polarization)
    cb = Checkbox(geom_gl[end+1, 1], checked = false)
    Label(geom_gl[end, 2], "Polarization", halign = :center)
    pol_controls = GridLayout(geom_gl[end, 3])

    pol_menu = Menu(fig, options = zip(["yz", "xz", "xy"], 1:3), tellwidth = false)
    pol_controls[1, 1] = pol_menu

    plane_norm = Observable{Any}(1)

    Pnorm = maximum(norm, polarization |> stack |> x -> eachslice(x, dims = (1, 2, 3)))
    Px, Py, Pz = polarization ./ Pnorm
    dims = size(Px)

    Pmag = [norm([Px[i,j,k], Py[i,j,k], Pz[i,j,k]])
        for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
    Pmag_max = maximum(Pmag)

    arrow_step = 4
    plane_slider = Slider(pol_controls[1, 2], range = 1:1:dims[1], startvalue = 1,
        update_while_dragging = true, horizontal = true)

    on(pol_menu.selection) do s
        plane_norm[] = s
        set_close_to!(plane_slider, first(plane_slider.range[]))
        plane_slider.range[] = 1:1:dims[s]
    end
    notify(pol_menu.selection)

    surface_data = lift(plane_slider.value) do val
        d = plane_norm[]
        if d == 1
            sx = fill(Float32(val), dims[2], dims[3])
            sy = [Float32(j) for j in 1:dims[2], _ in 1:dims[3]]
            sz = [Float32(k) for _ in 1:dims[2], k in 1:dims[3]]
            sc = Pmag[val, :, :]
        elseif d == 2
            sx = [Float32(i) for i in 1:dims[1], _ in 1:dims[3]]
            sy = fill(Float32(val), dims[1], dims[3])
            sz = [Float32(k) for _ in 1:dims[1], k in 1:dims[3]]
            sc = Pmag[:, val, :]
        else
            sx = [Float32(i) for i in 1:dims[1], _ in 1:dims[2]]
            sy = [Float32(j) for _ in 1:dims[1], j in 1:dims[2]]
            sz = fill(Float32(val), dims[1], dims[2])
            sc = Pmag[:, :, val]
        end
        return (sx, sy, sz, sc)
    end

    surface!(ax,
        @lift($surface_data[1]),
        @lift($surface_data[2]),
        @lift($surface_data[3]),
        color = @lift($surface_data[4]),
        colormap = :viridis,
        colorrange = (0, Pmag_max),
        transparency = true,
        alpha = 0.6,
        visible = cb.checked)

    arrow_data = lift(plane_slider.value) do val
        d = plane_norm[]
        ranges = [1:arrow_step:dims[1], 1:arrow_step:dims[2], 1:arrow_step:dims[3]]
        ranges[d] = val:1:val
        pts = vec([Point3f(i, j, k) for i in ranges[1], j in ranges[2], k in ranges[3]])
        vs = vec([Point3f(Px[i,j,k], Py[i,j,k], Pz[i,j,k]) for i in ranges[1], j in ranges[2], k in ranges[3]])
        cs = vec([Pmag[i,j,k] for i in ranges[1], j in ranges[2], k in ranges[3]])
        return (pts, vs, cs)
    end

    arrows3d!(ax,
        @lift($arrow_data[1]),
        @lift($arrow_data[2]);
        lengthscale = 2.0,
        color = @lift($arrow_data[3]),
        colormap = :viridis,
        colorrange = (0, Pmag_max),
        visible = cb.checked)
end

function add_strain!(ax, fig, geom_gl, strain)
    hydrostatic = strain[1] .+ strain[4] .+ strain[6]
    biaxial = strain[6] .- (strain[1] .+ strain[4]) ./ 2

    components = Dict(
        "εxx" => strain[1], "εxy" => strain[2], "εxz" => strain[3],
        "εyy" => strain[4], "εyz" => strain[5], "εzz" => strain[6],
        "Hydro" => hydrostatic, "Biax" => biaxial
    )

    strain_data = Observable{Any}(hydrostatic)

    names = ["Hydro", "Biax", "εxx", "εyy", "εzz", "εxy", "εxz", "εyz"]

    abs_cb = Checkbox(geom_gl[end+1, 1], checked = false)
    Label(geom_gl[end, 2], "Strain (vol)", halign = :center)
    vol_controls = GridLayout(geom_gl[end, 3])
    strain_menu = Menu(fig, options = zip(names, names), tellwidth = false)
    vol_controls[1, 1] = strain_menu
    alpha_slider = Slider(vol_controls[1, 2], range = 0.0:0.05:4.0, startvalue = 0.4,
        update_while_dragging = true, horizontal = true)
    Label(vol_controls[1, 3], "α", halign = :center)
    absorption_slider = Slider(vol_controls[1, 4], range = 0.0:0.05:4.0, startvalue = 2.0,
        update_while_dragging = true, horizontal = true)
    Label(vol_controls[1, 5], "abs", halign = :center)
    thresh_slider = Slider(vol_controls[1, 6], range = 0.0:0.01:0.5, startvalue = 0.05,
        update_while_dragging = true, horizontal = true)
    Label(vol_controls[1, 7], "thresh", halign = :center)

    normed = lift(strain_data, thresh_slider.value) do s, t
        m = maximum(abs, s)
        m == 0 && return s
        n = s ./ m
        return map(v -> abs(v) < t ? zero(v) : v, n)
    end

    on(strain_menu.selection) do s
        strain_data[] = components[s]
    end

    cmap_obs = lift(alpha_slider.value) do a
        [RGBAf(0, 0.3, 1, a), RGBAf(0,0,0,0), RGBAf(1, 0.2, 0, a)]
    end
    volume!(ax, @lift($normed .^ 1), algorithm = :absorption,
        absorption = absorption_slider.value,
        visible = abs_cb.checked, transparency = true,
        colormap = cmap_obs, colorrange = (-1, 1))

    iso_cb = Checkbox(geom_gl[end+1, 1], checked = false)
    Label(geom_gl[end, 2], "Strain (iso)", halign = :center)
    iso_controls = GridLayout(geom_gl[end, 3])
    iso_slider = Slider(iso_controls[1, 1], range = 0.1:0.05:1.0, startvalue = 0.5,
        update_while_dragging = true, horizontal = true)
    neg_iso = @lift(-$(iso_slider.value))

    volume!(ax, normed, algorithm = :iso, alpha = 0.3,
        visible = iso_cb.checked, transparency = true,
        isovalue = iso_slider.value, isorange = 0.05,
        colormap = [:red])
    volume!(ax, normed, algorithm = :iso, alpha = 0.3,
        visible = iso_cb.checked, transparency = true,
        isovalue = neg_iso, isorange = 0.05,
        colormap = [:blue])
end

function add_density!(ax, geom_gl, density_obs)
    cb = Checkbox(geom_gl[1, 1], checked = true)
    Label(geom_gl[1, 2], "Ψ density", halign = :center)
    density_controls = GridLayout(geom_gl[1, 3])
    iso_slider = Slider(density_controls[1, 1], range = 0:0.01:1, startvalue = 0.7,
        update_while_dragging = true, horizontal = true)
    Label(density_controls[1, 2], "iso", halign = :center)
    volume!(ax, density_obs, algorithm = :iso, isovalue = iso_slider.value, isorange = 0.1, visible = cb.checked)
end

function add_components!(gl, components_obs, orbital_labels, orbital_colors; halign = 0.1, valign = :top, legend_halign = :left, legend_valign = :top, title = "Components", show_legend = true)
    ax_inset = Axis(gl[1, 1],
        width = Relative(0.2),
        height = Relative(0.2),
        aspect = 1,
        halign = halign,
        valign = valign,
        tellwidth = false,
        tellheight = false,
        title = title,
        backgroundcolor = :transparent)
    hidespines!(ax_inset)
    hidedecorations!(ax_inset)

    pie!(ax_inset, components_obs,
        color = orbital_colors,
        label = orbital_labels,
        radius = 4,
        inner_radius = 2,
        strokecolor = :white,
        strokewidth = 2,
        overdraw = true)

    show_legend && Legend(gl[1, 1],
        [MarkerElement(color = c, marker = :circle) for c in orbital_colors],
        orbital_labels,
        tellwidth = false,
        halign = legend_halign,
        valign = legend_valign)
end

function _plot_state(Ψs::AbstractVector{<:AbstractArray{Float64,4}}, band_labels, comp_labels, comp_colors, comp_groups; energies = nothing, state_prefix = "")
    n = length(Ψs)
    fig = Figure()

    unicode_sub(i) = join(Char(0x2080 + d) for d in reverse(digits(i)))
    if isnothing(energies)
        labels_menu = ["$i" for i in 1:n]
    else
        labels_menu = ["$(state_prefix)$(unicode_sub(i)) -> $(energies[i]) eV" for i in 1:n]
    end
    menu = Menu(fig, options = zip(labels_menu, 1:n), tellwidth = false)
    fig[1, 1] = vgrid!(Label(fig, "State", width = nothing), menu)

    gl = GridLayout(fig[2, 1])

    band_gl = GridLayout(gl[1, 1], tellheight = false)
    ax = Axis3(gl[1, 2])
    hidedecorations!(ax)
    geom_gl = GridLayout(gl[1, 3], tellheight = false)
    colsize!(gl, 2, Auto(1))

    Ψ_obs = Observable{Any}(Ψs[1])

    density_obs, components_obs = add_band_selector!(band_gl, Ψ_obs, band_labels, comp_groups)

    add_density!(ax, geom_gl, density_obs)
    add_components!(gl[1, 2], components_obs, comp_labels, comp_colors)

    on(menu.selection) do s
        Ψ_obs[] = Ψs[s]
    end
    notify(menu.selection)

    return fig
end

function PlotΨ(Ψs::AbstractVector{<:AbstractArray{Float64,4}}; geometry = nothing, piezo = nothing, polarization = nothing, ne::Int = 0, nh::Int = 0, energies::AbstractVector, strain = nothing)
    n = size(Ψs, 1)
    fig = Figure()

    unicode_sub(i) = join(Char(0x2080 + d) for d in reverse(digits(i)))
    if ne == nh == 0
        labels_menu = ["$i" for i in 1:n]
    else
        labels_menu = append!(["e$(unicode_sub(i)) -> $(energies[i]) eV" for i in 1:ne], ["h$(unicode_sub(i)) -> $(energies[i]) eV" for i in 1:nh])
    end
    menu = Menu(fig, options = zip(labels_menu, 1:n), tellwidth = false)
    fig[1, 1] = vgrid!(Label(fig, "State", width = nothing), menu)

    gl = GridLayout(fig[2, 1])

    band_gl = GridLayout(gl[1, 1], tellheight = false)
    ax = Axis3(gl[1, 2])
    hidedecorations!(ax)
    geom_gl = GridLayout(gl[1, 3], tellheight = false)
    colsize!(gl, 2, Auto(1))

    Ψ_obs = Observable{Any}(Ψs[1])
    e_obs = Observable{Any}(energies[1])

    band_labels = [
        L"s\uparrow", L"s\downarrow", L"p_x\uparrow", L"p_x\downarrow", L"p_y\uparrow", L"p_y\downarrow", L"p_z\uparrow", L"p_z\downarrow"
    ]
    permute!(band_labels, [1,3,5,7,2,4,6,8])
    comp_labels, comp_colors, comp_groups = band_decomposition(8)

    density_obs, components_obs = add_band_selector!(band_gl, Ψ_obs, band_labels, comp_groups)


    add_density!(ax, geom_gl, density_obs)
    add_components!(gl[1,2], components_obs, comp_labels, comp_colors)
    !isnothing(geometry)     && add_geometry!(ax, geom_gl, geometry)
    !isnothing(piezo)        && add_piezo!(ax, geom_gl, piezo)
    !isnothing(polarization) && add_polarization!(ax, fig, geom_gl, polarization)
    !isnothing(strain)       && add_strain!(ax, fig, geom_gl, strain)

    on(menu.selection) do s
        Ψ_obs[] = Ψs[s]
        e_obs[] = energies[s]
    end
    notify(menu.selection)

    return fig
end

function PlotH5(estruct::HDF5.File)
    read_transposed(path) = Array(estruct[path]) |> x -> real(permutedims(x, (3, 2, 1)))

    geometry = read_transposed("geometry/composition")
    piezo = read_transposed("piezo/potential")

    el = Array(estruct["estruct/el_eigenvectors_5d"])
    el_energies = Array(estruct["estruct/el_eigenvalues"])
    el = permutedims(el, (1, 2, 5, 4, 3))
    Ψes = eachslice(abs2.(el), dims = 1)
    ne = size(Ψes, 1)

    ho = Array(estruct["estruct/ho_eigenvectors_5d"])
    ho_energies = Array(estruct["estruct/ho_eigenvalues"])
    ho = permutedims(ho, (1, 2, 5, 4, 3))
    Ψhs = eachslice(abs2.(ho), dims = 1)
    nh = size(Ψhs, 1)

    px = read_transposed("piezo/Px")
    py = read_transposed("piezo/Py")
    pz = read_transposed("piezo/Pz")

    strain = [read_transposed("strain/e"*ij) for ij in ("XX", "XY", "XZ", "YY", "YZ", "ZZ")]

    PlotΨ(append!(Array.(Ψes), Array.(Ψhs)); geometry, piezo, polarization = (px, py, pz), ne, nh, energies = append!(el_energies, ho_energies), strain)
end

function add_ci_density!(ax, geom_gl, e_density_obs, h_density_obs)
    e_cb = Checkbox(geom_gl[1, 1], checked = true)
    Label(geom_gl[1, 2], "Electron", halign = :center)
    h_cb = Checkbox(geom_gl[2, 1], checked = true)
    Label(geom_gl[2, 2], "Hole", halign = :center)

    iso_controls = GridLayout(geom_gl[3, 3])
    iso_slider = Slider(iso_controls[1, 1], range = 0:0.01:1, startvalue = 0.7,
        update_while_dragging = true, horizontal = true)
    Label(iso_controls[1, 2], "iso", halign = :center)

    volume!(ax, e_density_obs, algorithm = :iso, isovalue = iso_slider.value, isorange = 0.1,
        visible = e_cb.checked, transparency = true, alpha = 0.5, colormap = [:dodgerblue])
    volume!(ax, h_density_obs, algorithm = :iso, isovalue = iso_slider.value, isorange = 0.1,
        visible = h_cb.checked, transparency = true, alpha = 0.5, colormap = [:crimson])
end

function PlotCIΨ(Ψe::AbstractVector{<:AbstractArray{Float64,4}}, Ψh::AbstractVector{<:AbstractArray{Float64,4}}, CIVecs::AbstractVector{<:AbstractVector{ComplexF64}}, CIEnergies::AbstractVector{Float64}; e_energies = nothing, h_energies = nothing, geometry = nothing, piezo = nothing, polarization = nothing, strain = nothing)
    ne = length(Ψe)
    nh = length(Ψh)
    r = min(ne, nh)
    CIVecs = reshape.(CIVecs, Ref((nh, ne)))
    nstates = length(CIVecs)

    # Natural orbital for a Schmidt pair: complex-weighted combination of the
    # single-particle states, turned into a band-resolved density via |·|².
    natural_orbital(Ψ, coeffs) = abs2.(sum(c .* ψ for (c, ψ) in zip(coeffs, Ψ)))
    comp_values(ψ, groups) = begin
        c = [sum(ψi) for ψi in eachslice(ψ, dims = 1)]
        [sum(c[i] for i in g) for g in groups]
    end

    e_labels, e_colors, e_groups = band_decomposition(size(Ψe[1], 1))
    h_labels, h_colors, h_groups = band_decomposition(size(Ψh[1], 1))

    fig = Figure()

    unicode_sub(i) = join(Char(0x2080 + d) for d in reverse(digits(i)))
    state_labels = ["X$(unicode_sub(i)) -> $(round(CIEnergies[i]; digits = 4)) eV" for i in 1:nstates]

    # Selection state kept independent of the (repurposed) menus below.
    state_sel = Observable(1)   # which exciton (CI eigenstate)
    pair_idx  = Observable(1)   # which Schmidt pair
    e_idx     = Observable(1)   # which bare electron eigenstate
    h_idx     = Observable(1)   # which bare hole eigenstate

    C_obs = lift(s -> CIVecs[s], state_sel)
    svd_obs = lift(svd, C_obs)
    pair_labels = lift(svd_obs) do F
        ["pair $k -> λ=$(round(abs2(F.S[k]); digits = 3))" for k in 1:r]
    end

    # Toggle between the correlated exciton picture (Schmidt natural orbitals)
    # and the bare single-particle picture.  The two selector menus are
    # repurposed rather than duplicated, so the exciton-specific menus are
    # simply absent in single-particle mode.
    mode_menu = Menu(fig, options = ["Exciton", "Single particle"], tellwidth = false)
    sp_obs = lift(s -> s == "Single particle", mode_menu.selection)

    sp_label(prefix, i, energies) = isnothing(energies) ? "$prefix$(i - 1)" :
        "$prefix$(i - 1) -> $(round(energies[i]; digits = 4)) eV"

    exc_optsA = collect(zip(state_labels, 1:nstates))
    sp_optsA  = collect(zip([sp_label("e", i, e_energies) for i in 1:ne], 1:ne))
    sp_optsB  = collect(zip([sp_label("h", i, h_energies) for i in 1:nh], 1:nh))

    menuA = Menu(fig, options = exc_optsA, tellwidth = false)
    menuB = Menu(fig, options = collect(zip(pair_labels[], 1:r)), tellwidth = false)

    labelA = lift(sp -> sp ? "Electron" : "State", sp_obs)
    labelB = lift(sp -> sp ? "Hole" : "Schmidt pair", sp_obs)

    fig[1, 1] = hgrid!(
        vgrid!(Label(fig, "Mode", width = nothing), mode_menu),
        vgrid!(Label(fig, labelA, width = nothing), menuA),
        vgrid!(Label(fig, labelB, width = nothing), menuB),
    )

    gl = GridLayout(fig[2, 1])
    ax = Axis3(gl[1, 1])
    hidedecorations!(ax)
    right_gl = GridLayout(gl[1, 2])
    comp_gl = GridLayout(right_gl[1, 1])
    geom_gl = GridLayout(right_gl[2, 1], tellheight = false)
    colsize!(gl, 1, Auto(1))

    # Configuration-weight matrix |C_ij|² (hole index on x, electron index on y)
    # for the selected CI state.  Square cells, integer ticks and a light-background
    # colormap keep the (usually sparse) matrix readable; dominant cells are
    # annotated with their weight.
    # left-align the (square) heatmap and keep the colourbar snug against it
    plot_gl = GridLayout(comp_gl[1, 1], halign = :left, tellwidth = true)
    checker_ax = Axis(plot_gl[1, 1], title = "Configuration weights",
        xlabel = "hole index", ylabel = "electron index",
        aspect = DataAspect(),
        xticks = 1:nh, yticks = 1:ne,
        xgridvisible = false, ygridvisible = false,
        xminorgridvisible = true, yminorgridvisible = true,
        xminorticks = 0.5:1:(nh + 0.5), yminorticks = 0.5:1:(ne + 0.5),
        xminorgridcolor = (:white, 0.6), yminorgridcolor = (:white, 0.6),
        height = 200)
    weight_obs = lift(C -> abs2.(C), C_obs)
    crange_obs = lift(w -> (0.0, max(maximum(w), eps())), weight_obs)
    hm = heatmap!(checker_ax, 1:nh, 1:ne, weight_obs,
        colormap = :dense, colorrange = crange_obs)
    Colorbar(plot_gl[1, 2], hm, label = L"|C_{ij}|^2")
    colsize!(plot_gl, 1, Aspect(1, nh / ne))

    # annotate the cells carrying meaningful weight (white text reads well on
    # the dark, high-weight end of the colormap)
    annot_obs = lift(weight_obs, crange_obs) do w, cr
        pts = Point2f[]
        strs = String[]
        for I in CartesianIndices(w)
            if w[I] > 0.05 * cr[2]
                push!(pts, Point2f(I[1], I[2]))
                push!(strs, string(round(w[I]; digits = 2)))
            end
        end
        (pts, strs)
    end
    text!(checker_ax, lift(a -> a[1], annot_obs);
        text = lift(a -> a[2], annot_obs),
        align = (:center, :center), color = :white, fontsize = 11)

    # Exciton mode: Schmidt natural orbitals of the selected pair.
    # Single-particle mode: the bare electron/hole eigenstate densities.
    e_no = lift(sp_obs, svd_obs, pair_idx, e_idx) do sp, F, k, ei
        sp ? Ψe[ei] : natural_orbital(Ψe, F.V[:, k])
    end
    h_no = lift(sp_obs, svd_obs, pair_idx, h_idx) do sp, F, k, hi
        sp ? Ψh[hi] : natural_orbital(Ψh, F.U[:, k])
    end

    e_density = lift(ψ -> ΨDensity(ψ, trues(size(ψ, 1))), e_no)
    h_density = lift(ψ -> ΨDensity(ψ, trues(size(ψ, 1))), h_no)
    e_comps = lift(ψ -> comp_values(ψ, e_groups), e_no)
    h_comps = lift(ψ -> comp_values(ψ, h_groups), h_no)

    add_ci_density!(ax, geom_gl, e_density, h_density)
    add_components!(gl[1, 1], e_comps, e_labels, e_colors; halign = 0.02, valign = :top, title = "Electron", show_legend = false)
    add_components!(gl[1, 1], h_comps, h_labels, h_colors; halign = 0.98, valign = :top, title = "Hole", show_legend = false)
    # both pies share the orbital colour scheme, so a single legend suffices;
    # place it at the top, between the two pie charts
    Legend(gl[1, 1],
        [MarkerElement(color = c, marker = :circle) for c in e_colors],
        e_labels, "Orbital",
        tellwidth = false, tellheight = false,
        halign = :center, valign = :top, orientation = :horizontal)
    !isnothing(geometry)     && add_geometry!(ax, geom_gl, geometry)
    !isnothing(piezo)        && add_piezo!(ax, geom_gl, piezo)
    !isnothing(polarization) && add_polarization!(ax, fig, geom_gl, polarization)
    !isnothing(strain)       && add_strain!(ax, fig, geom_gl, strain)

    # Menu A/B drive different observables depending on the mode.
    on(menuA.selection) do v
        isnothing(v) && return
        sp_obs[] ? (e_idx[] = v) : (state_sel[] = v)
    end
    on(menuB.selection) do v
        isnothing(v) && return
        sp_obs[] ? (h_idx[] = v) : (pair_idx[] = v)
    end

    # Repopulate both menus on a mode switch, restoring each mode's last choice.
    on(mode_menu.selection) do _
        if sp_obs[]
            menuA.options[] = sp_optsA
            menuB.options[] = sp_optsB
            menuA.i_selected[] = clamp(e_idx[], 1, ne)
            menuB.i_selected[] = clamp(h_idx[], 1, nh)
        else
            menuA.options[] = exc_optsA
            menuB.options[] = collect(zip(pair_labels[], 1:r))
            menuA.i_selected[] = clamp(state_sel[], 1, nstates)
            menuB.i_selected[] = clamp(pair_idx[], 1, r)
        end
    end

    # Keep the Schmidt-pair labels in sync with the selected exciton
    # (exciton mode only; leaves the hole menu untouched in SP mode).
    on(pair_labels) do labels
        sp_obs[] && return
        keep = clamp(pair_idx[], 1, r)
        menuB.options[] = collect(zip(labels, 1:r))
        menuB.i_selected[] = keep
    end

    return fig
end

function PlotH5(estruct::HDF5.File, CI::HDF5.File)

    read_transposed(path) = Array(estruct[path]) |> x -> real(permutedims(x, (3, 2, 1)))

    geometry = read_transposed("geometry/composition")
    piezo = read_transposed("piezo/potential")

    el = Array(estruct["estruct/el_eigenvectors_5d"])
    el_energies = Array(estruct["estruct/el_eigenvalues"])
    el = permutedims(el, (1, 2, 5, 4, 3))
    Ψes = eachslice(abs2.(el), dims = 1)
    ne = size(Ψes, 1)

    ho = Array(estruct["estruct/ho_eigenvectors_5d"])
    ho_energies = Array(estruct["estruct/ho_eigenvalues"])
    ho = permutedims(ho, (1, 2, 5, 4, 3))
    Ψhs = eachslice(abs2.(ho), dims = 1)
    nh = size(Ψhs, 1)

    px = read_transposed("piezo/Px")
    py = read_transposed("piezo/Py")
    pz = read_transposed("piezo/Pz")

    strain = [read_transposed("strain/e"*ij) for ij in ("XX", "XY", "XZ", "YY", "YZ", "ZZ")]

    CIvecs = Array(CI["ci/exciton/eigenvectors"]) |> eachrow .|> collect
    CIenergies = Array(CI["ci/exciton/eigenvalues"])

    CIne, CInh = Array(CI["ci/exciton/basis_indices"]) |> Basis -> (Basis[1,end], Basis[2,end])

    return PlotCIΨ(Array.(Ψes)[1:CIne], Array.(Ψhs)[1:CInh], CIvecs, CIenergies;
        e_energies = el_energies[1:CIne], h_energies = ho_energies[1:CInh],
        geometry, piezo, polarization = (px, py, pz), strain)
end

end # module StateRendering

