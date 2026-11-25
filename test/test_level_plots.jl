#=
The labelling figures, drawn from a real analysis of
synthetic C4v states (CICalc's own test generator, through the sibling layout
that [sources] already assumes), broken by a B1 shape, against the symmetric
reference. Each figure must build labelled, and keep its layout unlabelled.
=#

# its own module: CICalc and MatrixFreeKPSolver both export names like BandBasis
module LevelPlotTests

using Test
using CairoMakie
using StateRendering
using CICalc
using CICalc: level_bundle, LevelBundle, symmetry_sweep, Field

include(joinpath(dirname(pathof(CICalc)), "..", "test", "symmetry", "synthetic_states.jl"))

const LF = SymmetryFrame("C4v")
const LKS = symmetric_states(LF; irreps_e = double_irreps(LF), irreps_h = double_irreps(LF),
                             n = 13, seed = 21)
const LX = LKS.coords[1]

function _deformed(e)
    fac = [exp(e * (LX[i]^2 - LX[j]^2) / 2.25) for i in eachindex(LX), j in eachindex(LX)]
    ψ = [p .* reshape(fac, size(fac)..., 1, 1) for p in LKS.ψ]
    return kp_states(ψ, LKS.energies; basis = :jmj2, h = LKS.h, coords = LKS.coords,
                     centre = 0.0, kind = LKS.kind)
end
_shape(e) = Field(:shape, :scalar,
                  [[exp(-(x^2 + y^2) / 4) * (1 + e * (x^2 - y^2)) for x in LX, y in LX, z in LX]],
                  CICalc.Unitful.NoUnits, false, ntuple(d -> copy(LKS.coords[d]), 3), "")

@testset "the irrep palette" begin
    t = LF.table
    @test irrep_color("?", t.labels, t.double) == UNLABELLED_COLOR
    @test irrep_color("nonsense", t.labels, t.double) == UNLABELLED_COLOR
    # single- and double-valued irreps are numbered separately, from the first slot
    s1 = t.labels[findfirst(.!t.double)]
    d1 = t.labels[findfirst(t.double)]
    @test irrep_color(s1, t.labels, t.double) == irrep_color(d1, t.labels, t.double)
    @test length(unique(values(irrep_palette(t.labels[.!t.double])))) == count(.!t.double)
    @test irrep_color(d1 * "+" * t.labels[end], t.labels, t.double) ==
          irrep_color(d1, t.labels, t.double)
end

@testset "labelling figures build" begin
    k0, a0 = analyse_symmetry(LKS, LF)
    kb, ab = analyse_symmetry(_deformed(0.03), LF)
    lb = level_bundle(kb, ab; floor = true, reference = (k0, a0),
                      fields = Dict(:shape => _shape(0.1)),
                      declared = [isodd(i) ? "Γ6" : "Γ7" for i in 1:length(kb)])
    @test any(lb.disagree)
    for f in (plot_levels(lb), plot_levels(lb; relative = false),
              plot_irrep_weights(lb), plot_irrep_weights(lb; per = :level),
              plot_operation_map(lb), plot_symmetry_quality(lb), plot_charges(lb),
              plot_parent_mixing(lb), plot_field_irreps(lb), plot_field_irreps(lb.fields))
        @test f isa Figure
    end
    @test_throws ArgumentError plot_irrep_weights(lb; per = :class)
    ps = [0.0, 0.02, 0.04]
    sw = symmetry_sweep([_deformed(e) for e in ps], LF; parameter = ps,
                        fields = [Dict(:shape => _shape(e)) for e in ps])
    @test plot_sweep(sw) isa Figure
    @test plot_sweep(sw; tracks = 1:2, ncols = 2) isa Figure
end

@testset "and keep their layout unlabelled" begin
    lu = level_bundle(LKS)
    for f in (plot_levels(lu), plot_irrep_weights(lu), plot_operation_map(lu),
              plot_symmetry_quality(lu), plot_charges(lu), plot_field_irreps(lu))
        @test f isa Figure
    end
    # a parent figure needs a reference: refused, and the message says how
    @test_throws ArgumentError plot_parent_mixing(lu)
end

end # module LevelPlotTests
