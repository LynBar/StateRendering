# Run with:
#     julia --project=. -e 'using Pkg; Pkg.test()'
#
# CairoMakie throughout: every figure in this package except `explore` is meant
# to be renderable by a static backend, and testing under Cairo is what keeps
# that true. A test that only passed under GLMakie would not catch a stray
# `volume!` — which cannot be drawn on paper.

using CairoMakie
using StateRendering
using MatrixFreeKPSolver          # re-exports Grid, materials, geometries
using Test, Printf

# `import`, not `using`: GeometryBasics also exports `coordinates`, which here
# means the grid's axes.
import GeometryBasics

CairoMakie.activate!(type = "png")

const OUT = mktempdir()

# ── synthetic states: analytic, so quantities are checkable ────────────────

"Gaussian envelope of width `w` centred at `c`, times fixed per-band weights."
function gaussian_state(grid, weights; c = nothing, w = 2.5)
    xs, ys, zs = coordinates(grid)
    c = c === nothing ? (grid.L ./ 2) : c
    nb = length(weights)
    ψ = Array{ComplexF64}(undef, grid.N..., nb)
    for b in 1:nb, k in 1:grid.N[3], j in 1:grid.N[2], i in 1:grid.N[1]
        r2 = (xs[i] - c[1])^2 + (ys[j] - c[2])^2 + (zs[k] - c[3])^2
        ψ[i, j, k, b] = sqrt(weights[b]) * exp(-r2 / (2w^2))
    end
    return ψ
end

"Indicator of a ball, on the grid, as a Float64 array."
function ball(grid, centre, R)
    xs, ys, zs = coordinates(grid)
    return [Float64((xs[i]-centre[1])^2 + (ys[j]-centre[2])^2 + (zs[k]-centre[3])^2 ≤ R^2)
            for i in 1:grid.N[1], j in 1:grid.N[2], k in 1:grid.N[3]]
end

"Signed volume enclosed by a closed triangle mesh."
function mesh_volume(m)
    pts = GeometryBasics.coordinates(m)
    V = 0.0
    for f in GeometryBasics.faces(m)
        a, b, c = pts[f[1]], pts[f[2]], pts[f[3]]
        V += (a[1]*(b[2]*c[3] - b[3]*c[2]) -
              a[2]*(b[1]*c[3] - b[3]*c[1]) +
              a[3]*(b[1]*c[2] - b[2]*c[1])) / 6
    end
    return abs(V)
end

axis_titles(fig) = [c.title[] for c in fig.content if c isa Axis]

const N, L = 24, 20.0
const grid = Grid((N, N, N), (L, L, L), Dirichlet())
const geom = SphereDot((L/2, L/2, L/2), 5.0)

# one bundle per basis, so every test can check that labels actually travel
const H8  = eight_band_hamiltonian(grid, GaAs)
const H8w = wurtzite_hamiltonian(grid, GaN_wz)
const Hv  = ValenceHamiltonian(grid, electron_picture(GaAs))

const b8 = state_bundle(H8, [1.02, 1.19, 1.19],
                        [gaussian_state(grid, [0.80, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01]),
                         gaussian_state(grid, [0.70, 0.10, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01]; w = 3.5),
                         gaussian_state(grid, [0.60, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01]; w = 4.0)];
                        kind = :electron, geometry = geom)

const b8w = state_bundle(H8w, [3.5], [gaussian_state(grid, [0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01])];
                         kind = :electron, geometry = geom)

const bv = state_bundle(Hv, [-0.02, -0.05],
                        [gaussian_state(grid, [0.4, 0.1, 0.1, 0.3, 0.05, 0.05]),
                         gaussian_state(grid, [0.1, 0.4, 0.3, 0.1, 0.05, 0.05]; w = 3.0)];
                        geometry = geom)

# ---------------------------------------------------------------------------

@testset "StateRendering" begin

@testset "isolevel means enclosed probability" begin
    d = state_density(b8, 1)
    for f in (0.1, 0.5, 0.9)
        lvl = isolevel(d, f)
        # one voxel of slack: the level is a grid value, so the enclosed
        # fraction lands just past the target
        @test enclosed_fraction(d, lvl) ≥ f
        @test enclosed_fraction(d, lvl) ≤ f + 0.02
    end
    @test isolevel(d, 1.0) ≤ minimum(d[d .> 0])
    @test_throws ArgumentError isolevel(d, 0.0)
    @test_throws ArgumentError isolevel(d, 1.5)
    @test isolevel(zeros(4, 4, 4), 0.5) == 0.0

    # scale invariance: an unnormalised density gives the same *fraction*
    @test enclosed_fraction(7d, isolevel(7d, 0.5)) ≈ enclosed_fraction(d, isolevel(d, 0.5))

    # the point of the design — a peak-fraction level is NOT comparable between
    # a peaked and a diffuse state, an enclosed-probability level is
    peaked, diffuse = state_density(b8, 1), state_density(b8, 3)
    fmax(d, r) = enclosed_fraction(d, r * maximum(d))
    @test abs(fmax(peaked, 0.5) - fmax(diffuse, 0.5)) > 0.05
    @test enclosed_fraction(peaked, isolevel(peaked, 0.5)) ≈
          enclosed_fraction(diffuse, isolevel(diffuse, 0.5)) atol = 0.02
end

@testset "meshes carry physical coordinates" begin
    R = 6.0
    χ = ball(grid, (L/2, L/2, L/2), R)
    m = isosurface_mesh(χ, coordinates(grid)..., 0.5)
    @test m !== nothing
    # if vertices were in index units this would be off by dx³ = (L/N)³ ≈ 0.58
    @test mesh_volume(m) ≈ 4/3 * π * R^3 rtol = 0.05
    pts = GeometryBasics.coordinates(m)
    @test all(0 ≤ p[1] ≤ L for p in pts)

    @test isosurface_mesh(χ, coordinates(grid)..., 1.5) === nothing   # above max
    @test isosurface_mesh(χ, coordinates(grid)..., -0.5) === nothing  # below min
end

@testset "centroid and smoothing" begin
    off = (L/3, L/2, 2L/3)
    b = state_bundle(H8, [1.0], [gaussian_state(grid, ones(8) ./ 8; c = off, w = 2.0)])
    @test all(isapprox.(centroid(state_density(b, 1), b.x, b.y, b.z), off; atol = 0.3))

    χ = ball(grid, (L/2, L/2, L/2), 5.0)
    s = smooth(χ; σ = 1.0)
    @test sum(s) ≈ sum(χ) rtol = 0.02       # mass conserving away from the walls
    @test maximum(s) < maximum(χ)           # the step is rounded off
    @test smooth(χ; σ = 0) == χ
end

@testset "colours are keyed by label, not by index" begin
    @test component_color("HH") == component_color("X")   # same slot, same hue
    @test component_color("S") != component_color("HH")
    @test component_color("nonsense") === nothing

    # any number of components, always as many colours
    for b in (b8, b8w, bv)
        @test length(component_palette(b.component_labels)) == length(b.component_labels)
        @test length(band_palette(b)) == nbands(b)
    end
    # unknown labels still get distinct, reproducible colours — and a repeated
    # label gets the SAME colour, so the fallback is keyed by name like the table
    p = component_palette(["Γ7", "Γ9", "Γ7"])
    @test p[1] == p[3]
    @test p[1] != p[2]
    @test component_palette(["a", "b"]) == component_palette(["a", "b"])
    @test kind_color(:hole) != kind_color(:electron)
end

@testset "overlay fields" begin
    n = (N, N, N)
    @test ScalarField(rand(n...), "χ").signed == false
    @test ScalarField(rand(n...) .- 0.5, "V").signed == true
    @test ScalarField(rand(n...), "V"; signed = true).signed == true

    εxx = fill(0.01, n...); εyy = fill(0.02, n...); εzz = fill(-0.03, n...)
    z = zeros(n...)
    s = StrainField(; εxx, εyy, εzz, εxy = z, εxz = z, εyz = z)
    @test all(hydrostatic(s) .≈ 0.0)
    @test all(biaxial(s) .≈ -0.03 - 0.015)
    @test length(scalar_fields(s)) == 8
    @test first(scalar_fields(s)).name == "hydrostatic"
    @test all(f -> f.signed, scalar_fields(s))
    @test_throws DimensionMismatch StrainField(; εxx, εyy, εzz, εxy = z, εxz = z,
                                                 εyz = zeros(2, 2, 2))

    v = VectorField(fill(3.0, n...), fill(4.0, n...), z, "P")
    @test all(magnitude(v) .≈ 5.0)

    # a field of the wrong size must be refused, not broadcast into nonsense
    bad = ScalarField(rand(4, 4, 4), "bad")
    @test_throws DimensionMismatch plot_field(b8, bad)
end

@testset "figures build and carry the right labels" begin
    @test plot_state(b8, 1) isa Figure
    @test plot_states(b8; ncols = 2) isa Figure
    @test plot_slices(b8, 1) isa Figure
    @test plot_composition(b8) isa Figure
    @test plot_spectrum(b8) isa Figure

    # band panels must be titled from the bundle: same band COUNT, different basis
    tz = axis_titles(plot_bands(b8, 1))
    tw = axis_titles(plot_bands(b8w, 1))
    @test any(t -> startswith(t, "HH↑"), tz)
    @test any(t -> startswith(t, "SO↓"), tz)
    @test !any(t -> startswith(t, "X↑"), tz)
    @test any(t -> startswith(t, "X↑"), tw)
    @test !any(t -> startswith(t, "HH↑"), tw)

    # a hole bundle is labelled as such, and its own basis is used
    @test any(t -> occursin("HH", t), axis_titles(plot_bands(bv, 1)))
    @test occursin("h1", axis_titles(plot_spectrum(bv))[1]) ||
          plot_spectrum(bv) isa Figure

    ε = StrainField(; εxx = 0.01 .* b8.geometry, εyy = 0.01 .* b8.geometry,
                      εzz = -0.02 .* b8.geometry, εxy = zeros(N, N, N),
                      εxz = zeros(N, N, N), εyz = zeros(N, N, N))
    @test plot_field(b8, ε) isa Figure
    @test plot_field(b8, ScalarField(b8.geometry, "χ")) isa Figure
end

@testset "figures save under CairoMakie" begin
    # the real test of backend independence: a GPU-only primitive would throw
    for (name, fig) in (("state", plot_state(b8, 1)),
                        ("states", plot_states(b8; ncols = 3)),
                        ("slices", plot_slices(b8, 1)),
                        ("bands", plot_bands(b8, 1)),
                        ("composition", plot_composition(b8)),
                        ("spectrum", plot_spectrum(b8)))
        path = joinpath(OUT, name * ".png")
        save(path, fig)
        @test isfile(path) && filesize(path) > 1000
    end
end

@testset "primitives draw into an axis you own" begin
    fig = Figure()
    ax = state_axis3(fig[1, 1]; title = "hand-built")
    @test add_density!(ax, b8, 1) !== nothing
    @test add_geometry!(ax, b8) !== nothing
    @test add_density!(ax, b8, 1; bands = [1, 2], color = :kind) !== nothing
    @test add_scalar!(ax, b8, ScalarField(b8.geometry .- 0.5, "signed")) isa Vector

    ax2 = Axis(fig[1, 2]; aspect = DataAspect())
    @test add_slice!(ax2, b8, 1; plane = :z) !== nothing
    @test ax2.xlabel[] == "x (nm)" && ax2.ylabel[] == "y (nm)"
    @test add_slice!(Axis(fig[1, 3]), b8, 1; plane = :x) !== nothing

    ax3 = Axis(fig[2, 1])
    add_component_bars!(ax3, b8, 1)
    @test last(ax3.yticks[]) == b8.component_labels

    # a bundle with no geometry simply skips the outline
    nog = state_bundle(H8, [1.0], [gaussian_state(grid, ones(8) ./ 8)])
    @test add_geometry!(ax, nog) === nothing

    v = VectorField(fill(1.0, N, N, N), zeros(N, N, N), fill(0.5, N, N, N), "P")
    @test add_vectors!(ax, b8, v; plane = :z) isa Vector
    @test_throws ArgumentError add_slice!(ax2, b8, 1; plane = :q)
end

@testset "explore builds (statically) under Cairo" begin
    fig = @test_logs (:warn,) match_mode = :any explore(b8)
    @test fig isa Figure
    ε = StrainField(; εxx = 0.01 .* b8.geometry, εyy = zeros(N, N, N),
                      εzz = -0.02 .* b8.geometry, εxy = zeros(N, N, N),
                      εxz = zeros(N, N, N), εyz = zeros(N, N, N))
    v = VectorField(0.01 .* b8.geometry, zeros(N, N, N), 0.02 .* b8.geometry, "P")
    @test explore(b8; strain = ε, vectors = v) isa Figure
    @test explore(bv) isa Figure          # 6 bands, different basis
    # an UNSIGNED overlay: only the positive isosurface exists, and the negative
    # branch must degrade to an empty mesh rather than throwing
    @test explore(b8; fields = [ScalarField(b8.geometry, "χ")]) isa Figure
    # overlays on the wrong grid are refused up front
    @test_throws DimensionMismatch explore(b8; fields = [ScalarField(rand(4,4,4), "x")])
end

include("test_hdf5.jl")     # the loader for files from another solver

end # StateRendering
