#=
Figures from a CICalc results archive: built on
synthetic C3v [111] states, saved, reloaded, and drawn with `save_figures`; a missing
product is refused with the product to add.
=#

module ArchiveFigureTests

using Test
using CairoMakie
using StateRendering
using StateRendering: node_name, select!, focus!, back!, forward!, step!, follow!, find_levels,
                      decay_tree, feeders, availability
using CICalc
using CICalc.Unitful: @u_str

include(joinpath(dirname(pathof(CICalc)), "..", "test", "symmetry", "synthetic_states.jl"))

const AF = SymmetryFrame("C3v"; axis = [1, 1, 1], inplane = [1, 1, -2])
const AD = double_irreps(AF)

function archive(; amplitudes = false)
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
    store!(res, "render/orbitals", render_orbitals(ks; stride = 1, amplitudes))
    return res
end

@testset "figures from an archive" begin
    dir = mktempdir()
    p = joinpath(dir, "run.h5")
    save_results(p, archive(); verbose = false)
    res = load_results(p)
    out = joinpath(dir, "figs")
    paths = save_figures(res; out, verbose = false)
    # one folder per kind of figure
    names = Set(first(splitext(replace(relpath(q, out), '\\' => '/'))) for q in paths)
    @test issubset(["labels/levels", "labels/operation_map", "labels/quality",
                    "sectors/X/levels", "sectors/XX/levels", "sectors/0/levels",
                    "sectors/XX/configurations_1", "sectors/XX/configurations_1_species",
                    "sectors/X/densities_1",
                    "orbitals/orbital_1", "cascades/XX_X_0"], names)
    @test all(q -> filesize(q) > 5_000, paths)
    # the archive methods pick the bundle; a sector's levels with and without stored rates
    @test plot_cascade(res) isa Figure && plot_levels(res) isa Figure
    @test plot_sector(res, "X") isa Figure && plot_sector(res, "XX"; nlevels = 40) isa Figure
    @test plot_sector(res, "0") isa Figure
    @test plot_sector(res, "X"; by_irrep = false) isa Figure            # one column
    # one state: densities, configurations (exciton heatmap, XX bar chart), orbitals
    @test plot_densities(res, "X", 1) isa Figure
    @test plot_densities(res, "XX", 1; geometry = false) isa Figure
    @test plot_configurations(res, "X", 2) isa Figure
    # a biexciton in terms of excitons (the archive holds X with its vectors), or by
    # determinants on request
    @test plot_configurations(res, "XX", 1) isa Figure
    @test plot_configurations(res, "XX", 1; basis = :determinants, top = 5) isa Figure
    @test plot_configurations(res, "XX", 1; basis = :species) isa Figure   # e pair ⊗ h pair
    @test_throws ArgumentError plot_configurations(res, "0", 1; basis = :species)
    noX = archive(); delete!(noX.sectors, "X")
    @test plot_configurations(noX, "XX", 1) isa Figure                  # falls back
    @test_throws ArgumentError plot_configurations(noX, "XX", 1; basis = :excitons)
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

@testset "panels: the figures drawn into one layout" begin
    res = archive()
    fig = Figure(size = (1600, 1000))
    ps = [draw_sector!(fig[1, 1], res, "X"), draw_configurations!(fig[1, 2], res, "XX", 1),
          draw_densities!(fig[2, 1], res, "X", 1), draw_cascade!(fig[2, 2], res),
          draw_orbital!(fig[1, 3], res, 1)]
    @test all(p -> p.layout isa GridLayout && length(p.size) == 2, ps)
    @test !isempty(ps[1].lines)
    p = joinpath(mktempdir(), "panels.png")
    save(p, fig)
    @test filesize(p) > 20_000
    # a figure of one panel takes the panel's own size, unless given one
    @test Tuple(size(plot_sector(res, "X").scene)) == ps[1].size
    @test Tuple(size(plot_sector(res, "X"; size = (500, 400)).scene)) == (500, 400)
end

@testset "the explorer model" begin
    dir = mktempdir()
    save_results(joinpath(dir, "run.h5"), archive(); verbose = false)
    res = load_results(joinpath(dir, "run.h5"))
    m = explorer_model(res)
    @test m.sectors == ["0", "X", "XX"]                           # in charge order
    # the nodes are the by-irrep lines: every state of the levels shown, once
    for n in m.sectors
        s = res.sectors[n]
        st = sort!(reduce(vcat, [x.states for x in m.nodes if x.sector == n]))
        @test st == collect(1:last(s.levels[min(40, end)]))
    end
    @test m.nodes[m.selection[]].sector == "X" && m.nodes[m.selection[]].level == 1

    # the cascade through a node: the XX ground level's decay reaches X and then the
    # vacuum in full
    xx = first(find_levels(m, "XX 1"))
    t = decay_tree(m, xx)
    into(n) = sum((f for (k, f) in t if m.nodes[k].sector == n); init = 0.0)
    @test into("X") ≈ 1 atol = 1e-4
    @test into("0") ≈ 1 atol = 1e-4
    g = only(find_levels(m, "0"))
    @test feeders(m, g)[xx] ≈ 1 atol = 1e-4
    @test all(e -> e.rate > 0 && 0 <= e.axial <= 1 + 1e-9, m.lines)

    # navigation, with a history like a browser's
    select!(m, xx)
    follow!(m)
    @test m.selection[] == m.lines[first(m.out[xx])].to && m.nodes[m.selection[]].sector == "X"
    back!(m)
    @test m.selection[] == xx
    forward!(m)
    @test m.nodes[m.selection[]].sector == "X"
    step!(m, :next)
    @test m.nodes[m.selection[]].sector == "XX" && m.nodes[m.selection[]].level == 1
    step!(m, :previous); step!(m, :previous)
    @test m.nodes[m.selection[]].sector == "0"
    a = first(find_levels(m, "X 1"))
    select!(m, a); step!(m, :up)
    b = m.nodes[m.selection[]]
    @test b.id == a || (b.sector == "X" && b.column == m.nodes[a].column && b.dE >= m.nodes[a].dE)
    @test all(k -> m.nodes[k].sector == "X" && m.nodes[k].level == 1, find_levels(m, "X 1"))
    lab = m.nodes[a].label
    @test a in find_levels(m, "X $lab") && all(k -> m.nodes[k].label == lab, find_levels(m, lab))
    @test occursin("X1", node_name(m, a)) && occursin("0 #1", node_name(m, g))

    # what is stored, and the spec key for what is not
    @test availability(m, xx).vectors === true && availability(m, xx).rates === true
    deep = last([x.id for x in m.nodes if x.sector == "XX"])
    @test occursin("keep:", string(availability(m, deep).vectors))
    @test availability(m, g).rates isa String
    @test_throws ArgumentError step!(m, :sideways)
    @test_throws ArgumentError select!(m, 0)

    # a focused state within the selected level
    multi = first(x for x in m.nodes if length(x.states) > 1)
    select!(m, multi.id)
    @test m.focus[] == first(multi.states)
    select!(m, multi.id; state = last(multi.states))
    @test m.focus[] == last(multi.states) && m.selection[] == multi.id
    @test_throws ArgumentError select!(m, multi.id; state = 10_000)
    lvl = [x for x in m.nodes if x.sector == multi.sector && x.level == multi.level]
    for y in lvl, i in y.states                      # any state of the level, any irrep part
        focus!(m, i)
        @test m.focus[] == i && i in m.nodes[m.selection[]].states
    end
    other = first(x for x in m.nodes if x.sector != multi.sector)
    @test_throws ArgumentError focus!(m, other.states[1] + 10_000)
    select!(m, other.id); back!(m)
    @test m.focus[] == first(m.nodes[m.selection[]].states)
end

@testset "the explorer" begin
    res = archive()
    e = explore(res; display = false)
    m = e.model
    X = Base.get_extension(StateRendering, :StateRenderingCICalcExt)
    @test !isempty(e.segments[]) && e.lens[] == "recombination"
    # every lens, every group the frame allows, with a selection in each sector
    groups = e.state[:menus][2].options[]
    @test first(groups) == "C3v" && "C3" in groups
    for lens in ("recombination", "symmetry", "composition"), g in groups
        e.lens[] = lens
        e.group[] = g
        for n in m.sectors
            select!(m, first(find_levels(m, n)))
        end
    end
    e.group[] = "C3v"; e.lens[] = "recombination"
    # a click on a line of the map selects it; one far from every line does nothing
    s = first(x for x in e.segments[] if m.nodes[x.id].sector == "XX")
    @test X._click_map!(e, (s.x + 0.1, s.y)) == s.id && m.selection[] == s.id
    @test X._click_map!(e, (s.x + 0.1, s.y + 1e6)) === nothing
    # keys: Enter follows the strongest line down, Backspace comes back, 2 the lens
    X._key!(e, Keyboard.enter)
    @test m.nodes[m.selection[]].sector == "X"
    X._key!(e, Keyboard.backspace)
    @test m.selection[] == s.id
    X._key!(e, Keyboard._2)
    @test e.lens[] == "symmetry"
    # a line in the detail panel leads to its other end
    e.lens[] = "recombination"
    d = e.state[Symbol("lines out")]
    @test X._click_lines!(e, Symbol("lines out"), (first(d.lines).ħω, 0.0)) == first(d.lines).to
    # the 3-D window: stored densities, an orbital, and a level without either
    select!(m, first(find_levels(m, "X 1")))
    @test e.state[:axis3d] !== nothing
    e.state[:mode3d][] = "orbital"; X._draw_3d!(e)
    @test e.state[:axis3d] !== nothing
    e.state[:mode3d][] = "densities"
    select!(m, last([x.id for x in m.nodes if x.sector == "XX"]))
    @test e.state[:axis3d] === nothing                  # no vectors: a note naming `keep:`
    # a kept state without stored densities: its occupation densities, Σₖ nₖ|ψₖ|²
    k = first(x.id for x in m.nodes if x.sector == "X" && !any(i -> i in (1, 2), x.states))
    select!(m, k)
    @test e.state[:axis3d] !== nothing
    n = X._occupations(m, k)
    @test sum(n.e) ≈ 1 atol = 1e-10                     # one electron
    @test sum(n.h) ≈ 1 atol = 1e-10                     # one hole
    od = X._occupation_densities(m, k)
    T = X._val(res.extra["render/orbitals"].total)
    @test sum(od.e) ≈ sum(n.e[j] * sum(Float64.(T[:, :, :, st.orb]))
                          for (j, st) in enumerate(res.sectors["X"].estates)) rtol = 1e-10
    # the orbital menu follows the focused state: one entry per Kramers pair, the most
    # occupied pair first selected, occupations shown
    e.state[:mode3d][] = "orbital"; X._draw_3d!(e)
    o = e.state[:pick][]
    nf = X._occupations(m, "X", [m.focus[]])
    occ = zeros(length(res.orbitals.energies))
    foreach(((j, st),) -> occ[st.orb] = nf.e[j], enumerate(res.sectors["X"].estates))
    foreach(((j, st),) -> occ[st.orb] = nf.h[j], enumerate(res.sectors["X"].hstates))
    prs = X._orbital_pairs(res)
    @test length(e.state[:pick_items]) == length(prs)
    @test o == argmax([sum(occ[p.members]) for p in prs])
    @test occursin("n ", e.state[:pick_menu].options[][o])
    e.state[:mode3d][] = "densities"; X._draw_3d!(e)
    # without the amplitude tier, natural orbitals and constituents say what to add
    @test occursin("amplitudes: true", availability(m, k).amplitudes)
    e.state[:mode3d][] = "natural orbitals"; X._draw_3d!(e)
    @test e.state[:axis3d] === nothing
    e.state[:mode3d][] = "densities"; X._draw_3d!(e)
    # the fine-structure strip: a click focuses a state, Tab steps through the level, and
    # the 3-D window follows the focused state
    multi = first(x for x in m.nodes if x.sector == "X" && length(X.level_states(select!(m, x.id))) > 1)
    select!(m, multi.id)
    f = last(e.state[:fine])
    @test X._click_fine!(e, (f.x, f.y)) == f.state && m.focus[] == f.state
    st = X.level_states(m)
    X._key!(e, Keyboard.tab)
    @test m.focus[] == st[mod1(findfirst(==(f.state), st) + 1, length(st))]
    @test occursin("state $(m.focus[])", e.fine_axis.title[])
    @test e.state[:axis3d] !== nothing
    dir = mktempdir()
    save(joinpath(dir, "map.png"), e.fig); save(joinpath(dir, "3d.png"), e.fig3d)
    @test filesize(joinpath(dir, "map.png")) > 20_000
    @test_throws ArgumentError explore(res; lens = "phonons", display = false)
end

@testset "Kramers pairs" begin
    X = Base.get_extension(StateRendering, :StateRenderingCICalcExt)
    # orbitals: E1, E2, … and H1, … by degenerate pairs in energy order
    r = CICalcResults()
    r.orbitals = (energies = [1.0, 1.0 + 1e-11, 1.2, 1.3, 1.3, -0.5, -0.5 - 1e-11],
                  kind = [:e, :e, :e, :e, :e, :h, :h])
    prs = X._orbital_pairs(r)
    @test [p.name for p in prs] == ["E1", "E2", "E3", "H1"]
    @test [p.members for p in prs] == [[1, 2], [3], [4, 5], [6, 7]]
    @test X._pair_of_orbital(r, 5) == 3
    # an odd sector's states pair within each level; an even sector's do not
    odd = (sector = (2, 1), levels = [1:2, 3:5], energies = [1.0, 1.0, 2.0, 2.0 + 2e-6, 2.0 + 1e-9])
    d = X._kramers_map(odd)
    @test d[1] == d[2] == [1, 2] && d[3] == d[5] == [3, 5] && d[4] == [4]
    @test isempty(X._kramers_map((sector = (1, 1), levels = [1:2], energies = [1.0, 1.0])))
    # an exciton over paired orbitals: one cell per pair of pairs, with the 2 × 2 inset
    I = zeros(ComplexF64, 8, 8, 8, 8)
    es = [CICalc.SPState(energy = E, band = :e, orb = k) for (k, E) in enumerate([0.8, 0.8, 0.9, 0.9])]
    hs = [CICalc.SPState(energy = E, band = :h, orb = 4 + k) for (k, E) in enumerate([-0.7, -0.7, -0.75, -0.75])]
    res = CICalcResults()
    store!(res, "X", solve_sector(es, hs, I; Ne = 1, Nh = 1))
    f = Figure()
    p = draw_configurations!(f[1, 1], res, "X", 1)
    @test p.kind === :pairs
    @test length(p.axis.yticks[][1]) == 2 && length(p.axis.xticks[][1]) == 2
    @test draw_configurations!(Figure()[1, 1], res, "X", 1; pairs = false).kind === :orbitals
end

@testset "the amplitude tier" begin
    res = archive(; amplitudes = true)
    X = Base.get_extension(StateRendering, :StateRenderingCICalcExt)
    xs, xx = res.sectors["X"], res.sectors["XX"]
    # densities of a state stored and not: the contraction reproduces the stored ones
    rd = reduced_densities(res, "X", 1)
    @test rd.e ≈ X._val(xs.densities[1].e) rtol = 1e-4
    @test rd.h ≈ X._val(xs.densities[1].h) rtol = 1e-4
    @test X._state_densities(res, xs, 1)[2] === :stored
    @test X._state_densities(res, xs, 3)[2] === :amplitudes
    @test plot_densities(res, "X", 3) isa Figure
    # natural orbitals: a figure of every group, or one; the occupations sum to N
    no = natural_orbitals(res, "XX", collect(xx.levels[1]); cut = 0.0)
    @test sum(sum(g.n) for g in no.e) ≈ 2 atol = 1e-8
    @test plot_natural_orbitals(res, "XX", collect(xx.levels[1])) isa Figure
    @test plot_natural_orbitals(res, "X", 1; pick = (:h, 1)) isa Figure
    # constituents: the heaviest pair by default; every pair's weight, summing to what
    # the lowest levels capture
    f = Figure()
    c = draw_constituents!(f[1, 1], res, "XX", 1)
    w = exciton_pair_weights(SectorSolution(xx), SectorSolution(xs); states = [1],
                             xlevels = xs.levels, nlevels = 8)
    @test sum(last, c.cells) ≈ w.captured[1] rtol = 1e-10
    @test c.cell == (c.cells[1][1], c.cells[1][2]) && issorted(last.(c.cells); rev = true)
    @test_throws ArgumentError plot_constituents(res, "X", 1)
    dir = mktempdir()
    paths = save_figures(res; out = dir, verbose = false)
    names = Set(first(splitext(replace(relpath(q, dir), '\\' => '/'))) for q in paths)
    @test issubset(["sectors/X/natural_orbitals_1", "sectors/XX/constituents_1",
                    "sectors/XX/natural_orbitals_1", "sectors/0/levels"], names)
    # the explorer: every mode of the 3-D window on X and XX levels
    e = explore(res; display = false)
    m = e.model
    for name in ("X 1", "XX 1"), mode in X._MODES3D
        select!(m, first(find_levels(m, name)))
        e.state[:mode3d][] = mode; X._draw_3d!(e)
        ok = !(mode == "constituents" && name == "X 1")
        @test (e.state[:axis3d] !== nothing) == ok
    end
    @test availability(m, m.selection[]).densities === true
    # the pick menu: all groups, then one; the densities' level average
    e.state[:mode3d][] = "natural orbitals"; X._follow_pick!(e)
    @test startswith(first(e.state[:pick_menu].options[]), "all")
    e.state[:pick][] = 2; X._draw_3d!(e)
    @test e.state[:axis3d] !== nothing
    e.state[:mode3d][] = "densities"; X._follow_pick!(e)
    e.state[:pick][] = 2; X._draw_3d!(e)                 # the level's average
    @test e.state[:axis3d] !== nothing
    # a click on a cell of XX's make-up (composition lens) shows that pair
    e.lens[] = "composition"
    @test e.state[:config].kind === :excitons
    @test X._click_cell!(e, (1.1, 1.9)) == (1, 2)
    @test e.state[:mode3d][] == "constituents"
    @test e.state[:pick_items][e.state[:pick][]] == (1, 2)
    @test e.state[:axis3d] !== nothing
    @test X._click_cell!(e, (40, 40)) === nothing
    save(joinpath(dir, "3d.png"), e.fig3d)
    @test filesize(joinpath(dir, "3d.png")) > 20_000
end

end # module ArchiveFigureTests
