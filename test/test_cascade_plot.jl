#=
The radiative cascade figure (CICalc.cascade_bundle): XX → X → 0 of synthetic C3v
[111] states, labelled and unlabelled, saved as PNG.
=#

module CascadePlotTests

using Test
using CairoMakie
using StateRendering
using CICalc
using CICalc.Unitful: @u_str

include(joinpath(dirname(pathof(CICalc)), "..", "test", "symmetry", "synthetic_states.jl"))

const CF = SymmetryFrame("C3v"; axis = [1, 1, 1], inplane = [1, 1, -2])
const CD = double_irreps(CF)
const CKS0 = symmetric_states(CF; irreps_e = CD[1:2], irreps_h = CD[[1, 3, 2]], n = 13,
                              seed = 8, basis = :jmj)

function cascade(ks; labels = nothing)
    o = OrbitalSet(ks; labels)
    V = screened(unwrap(coulomb_integrals(o)), 5.0)
    sols = [solve_sector(o, V; Ne = n, Nh = n) for n in 2:-1:0]
    return cascade_bundle(DipoleOperator(o; form = :momentum), sols; E_P = 22u"eV",
                          n_r = 3.5)
end

@testset "the cascade figure" begin
    ks, a = analyse_symmetry(CKS0, CF)
    cb = cascade(ks; labels = a)
    f = plot_cascade(cb)
    @test f isa Figure
    p = joinpath(mktempdir(), "cascade.png")
    save(p, f)
    @test filesize(p) > 10_000
    # everything filtered away: the spectrum says so rather than failing
    @test plot_cascade(cb; min_yield = 2.0) isa Figure
    # one spectrum row, linear heights, few arrows; and the refusals
    @test plot_cascade(cb; split = :none, scale = :linear, max_arrows = 2,
                       label_arrows = 1) isa Figure
    @test_throws ArgumentError plot_cascade(cb; scale = :sqrt)
    @test_throws ArgumentError plot_cascade(cb; split = :step)
    # a crowded cascade: every X level that receives any flow, and an "others" slot
    o = OrbitalSet(ks; labels = a)
    V = screened(unwrap(coulomb_integrals(o)), 5.0)
    sols = [solve_sector(o, V; Ne = n, Nh = n) for n in 2:-1:0]
    busy = cascade_bundle(DipoleOperator(o; form = :momentum), sols; E_P = 22u"eV",
                          n_r = 3.5, min_branch = 1e-12, max_levels = 4)
    @test length(busy.levels[2]) == 4
    @test plot_cascade(busy; min_yield = 1e-8) isa Figure
    # unlabelled: grey "?" marks, Cartesian polarisation, same layout
    cu = cascade(ks)
    @test all(isempty, cu.level_labels[1]) && plot_cascade(cu) isa Figure
end

end # module CascadePlotTests
