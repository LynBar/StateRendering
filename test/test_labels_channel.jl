#=
The `labels` channel of the single-particle figures: point-group labels reach the
spectrum, composition and level primitives through a keyword, and a wrong count is
refused.
=#

using Test
using CairoMakie
using StateRendering
using MatrixFreeKPSolver: Grid, Dirichlet, BandBasis, state_bundle

const NX = 13

# a minimal single-particle bundle, for the `labels` channel
function toy_states_bundle(n = 3)
    g = Grid(NX, 10.0, Dirichlet())
    basis = BandBasis(["S↑", "S↓"], "S" => [1, 2])
    ψ = [ComplexF64.(reshape(1:(NX^3 * 2), NX, NX, NX, 2) .+ k) for k in 1:n]
    return state_bundle(basis, g, collect(1.0:n), ψ; kind = :electron)
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
