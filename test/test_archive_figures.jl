#=
Figures from a CICalc results archive (CICalc RESULTS_PIPELINE_PLAN 2a): built on
synthetic C3v [111] states, saved, reloaded, and drawn with `save_figures`; a missing
product is refused with the product to add.
=#

module ArchiveFigureTests

using Test
using CairoMakie
using StateRendering
using CICalc
using CICalc.Unitful: @u_str

include(joinpath(dirname(pathof(CICalc)), "..", "test", "symmetry", "synthetic_states.jl"))

const AF = SymmetryFrame("C3v"; axis = [1, 1, 1], inplane = [1, 1, -2])
const AD = double_irreps(AF)

function archive()
    ks, a = analyse_symmetry(symmetric_states(AF; irreps_e = AD[1:2], irreps_h = AD[[1, 3, 2]],
                                              n = 13, seed = 8, basis = :jmj), AF)
    o = OrbitalSet(ks; labels = a)
    V = screened(unwrap(coulomb_integrals(o)), 5.0)
    XX, X, G0 = [solve_sector(o, V; Ne = n, Nh = n) for n in 2:-1:0]
    d = DipoleOperator(o; form = :momentum)
    res = CICalcResults(; params = Dict("eps_inf" => 5.0))
    store!(res, ks, a)
    store!(res, V)
    store!(res, "XX", XX; keep = 3, densities = [1], ks)
    store!(res, "X", X; densities = 1:2, ks)
    store!(res, "0", G0)
    kept = SectorSolution(res.sectors["XX"])
    store!(res, "XX→X", radiative_rates(d, kept, X; E_P = 22u"eV", n_r = 3.5))
    store!(res, "X→0", radiative_rates(d, X, G0; E_P = 22u"eV", n_r = 3.5))
    store!(res, "XX→X→0", cascade_bundle(d, [XX, X, G0]; E_P = 22u"eV", n_r = 3.5))
    store!(res, "render/orbitals", render_orbitals(ks; stride = 1))
    return res
end

@testset "figures from an archive" begin
    dir = mktempdir()
    p = joinpath(dir, "run.h5")
    save_results(p, archive(); verbose = false)
    res = load_results(p)
    out = joinpath(dir, "figs")
    paths = save_figures(res; out, verbose = false)
    names = Set(splitext(basename(q))[1] for q in paths)
    @test issubset(["levels", "operation_map", "quality", "sector_X", "sector_XX", "sector_0",
                    "cascade_XX_X_0"], names)
    @test all(q -> filesize(q) > 5_000, paths)
    # the archive methods pick the bundle; a sector's levels with and without stored rates
    @test plot_cascade(res) isa Figure && plot_levels(res) isa Figure
    @test plot_sector(res, "X") isa Figure && plot_sector(res, "XX"; nlevels = 40) isa Figure
    @test plot_sector(res, "0") isa Figure
    # one state: densities, configurations (exciton heatmap, XX bar chart), orbitals
    @test plot_densities(res, "X", 1) isa Figure
    @test plot_densities(res, "XX", 1; geometry = false) isa Figure
    @test plot_configurations(res, "X", 2) isa Figure
    @test plot_configurations(res, "XX", 1; top = 5) isa Figure
    @test plot_orbital(res, 1) isa Figure && plot_orbital(res, 3; bands = false) isa Figure
    msg(f) = try f(); "" catch e; sprint(showerror, e) end
    @test occursin("densities:", msg(() -> plot_densities(res, "X", 3)))
    @test occursin("keep:", msg(() -> plot_configurations(res, "XX", 5)))
    @test occursin("out of range", msg(() -> plot_orbital(res, 99)))
    # an archive built in memory draws the same, its heavy values not yet written
    mem = archive()
    @test plot_densities(mem, "X", 1) isa Figure && plot_orbital(mem, 1) isa Figure
    @test plot_configurations(mem, "X", 1) isa Figure
    # a missing product names what to add
    empty = CICalcResults()
    e = try plot_cascade(empty); nothing catch err; err end
    @test e isa ArgumentError && occursin("run spec", sprint(showerror, e))
    @test_throws ArgumentError plot_sector(res, "X⁻")
    @test_throws ArgumentError plot_levels(empty)
    @test occursin("orbitals:", msg(() -> plot_orbital(empty, 1)))
    # an archive with nothing drawable still runs, and says what it skipped
    @test isempty(save_figures(empty; out = joinpath(dir, "none"), verbose = false))
end

end # module ArchiveFigureTests
