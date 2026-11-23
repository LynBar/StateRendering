#=
The exciton figures and the `labels` channel.

The bundle is built by hand from its own fields rather than by running a CI:
these tests are about the drawing contract, and a synthetic bundle states
exactly what that contract is. CICalc's own suite checks that a real solve
produces a bundle satisfying it.
=#

using Test
using CairoMakie
using StateRendering
using CICalc: ExcitonBundle
using MatrixFreeKPSolver: Grid, Dirichlet, BandBasis, state_bundle

const NX = 13

function toy_bundle(; ne = 2, nh = 2, states = nothing, labelled = true,
                    bright = true, geometry = true)
    xs = collect(((0:NX-1) .- NX / 2) .* 0.8)
    gauss(c, w) = [exp(-((xs[i] - c[1])^2 + (xs[j] - c[2])^2 + (xs[k] - c[3])^2) /
                       (2w^2)) for i in 1:NX, j in 1:NX, k in 1:NX]
    nrm(d) = d ./ sum(d)

    nX = ne * nh
    sel = states === nothing ? collect(1:nX) : collect(states)
    C = [ComplexF64[(e == mod1(k, ne) && h == mod1(k, nh)) ? 1.0 : 0.0
                    for e in 1:ne, h in 1:nh] for k in 1:nX]
    E = collect(range(1.50; step = 0.004, length = nX))
    levels = [k:k for k in 1:nX]

    osc = zeros(Float64, nX, 3)
    bright && (osc[1, 3] = 1.0; osc[2, 1] = 0.4)
    tot = vec(sum(osc; dims = 2))

    return ExcitonBundle(
        E, C,
        labelled ? [isodd(k) ? "Γ1" : "Γ3" for k in 1:nX] : fill("", nX),
        Vector{Float64}[], labelled ? ["Γ1", "Γ2", "Γ3"] : String[],
        levels, ones(Int, nX),
        fill(true, nX), fill("", nX), zeros(Float64, nX),
        osc, tot, tot .> 1e-6,
        collect(1.0:ne), collect(0.1:0.1:0.1nh),
        labelled ? ["Γ4", "Γ4"][1:ne] : fill("", ne),
        labelled ? ["Γ5", "Γ6", "Γ4", "Γ4"][1:nh] : fill("", nh),
        [nrm(gauss((0.0, 0.0, 0.0), 1.6 + 0.1k)) for k in sel],
        [nrm(gauss((0.4, 0.0, 0.0), 1.1 + 0.1k)) for k in sel],
        xs, xs, xs,
        geometry ? gauss((0.0, 0.0, 0.0), 2.4) : nothing,
        Dict{Symbol,Any}(:density_states => sel, :n_e => ne, :n_h => nh))
end

# a minimal single-particle bundle, for the `labels` channel
function toy_states_bundle(n = 3)
    g = Grid(NX, 10.0, Dirichlet())
    basis = BandBasis(["S↑", "S↓"], "S" => [1, 2])
    ψ = [ComplexF64.(reshape(1:(NX^3 * 2), NX, NX, NX, 2) .+ k) for k in 1:n]
    return state_bundle(basis, g, collect(1.0:n), ψ; kind = :electron)
end

@testset "exciton figures build" begin
    xb = toy_bundle()
    @test plot_exciton(xb, 1) isa Figure
    @test plot_exciton(xb, 2; fraction = 0.9, geometry = false) isa Figure
    @test plot_exciton_spectrum(xb) isa Figure
    @test plot_exciton_spectrum(xb; relative = false) isa Figure
    @test plot_configurations(xb, 1) isa Figure
    @test plot_configurations(xb, 3) isa Figure
end

@testset "they survive an unlabelled, dark, geometry-less bundle" begin
    xb = toy_bundle(; labelled = false, bright = false, geometry = false)
    @test plot_exciton(xb, 1) isa Figure
    @test plot_exciton_spectrum(xb) isa Figure     # nothing bright: no f= text
    @test plot_configurations(xb, 1) isa Figure
end

@testset "a state whose density was not stored says so" begin
    xb = toy_bundle(; states = 2:3)
    @test plot_exciton(xb, 2) isa Figure
    @test plot_exciton(xb, 3) isa Figure
    err = try plot_exciton(xb, 1); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("no stored density", err.msg)
    # the level diagram and the configuration map need no densities at all
    @test plot_exciton_spectrum(xb) isa Figure
    @test plot_configurations(xb, 1) isa Figure
end

@testset "the labels channel reaches the single-particle figures" begin
    b = toy_states_bundle(3)
    labs = ["Γ6", "Γ6", "Γ4"]
    @test plot_spectrum(b) isa Figure
    @test plot_spectrum(b; labels = labs) isa Figure
    @test plot_composition(b; labels = labs) isa Figure

    ax = Axis(Figure()[1, 1])
    @test add_levels!(ax, b; labels = labs) === ax
    @test add_levels!(ax, b; labels = ["", "", ""]) === ax    # blank suppresses

    # a wrong count is a mistake, not something to recycle or truncate
    @test_throws DimensionMismatch plot_spectrum(b; labels = ["Γ6"])
    @test_throws DimensionMismatch plot_composition(b; labels = ["Γ6"])
    @test_throws DimensionMismatch add_levels!(Axis(Figure()[1, 1]), b;
                                               labels = ["a", "b"])
end
