# Solve an InAs/GaAs dot and produce the whole figure set.
#
#     julia --project=examples --threads=8 examples/quantum_dot.jl
#
# Runs under CairoMakie and writes PNGs, so it works headless — on a cluster, in
# CI, over ssh. Swap in GLMakie and uncomment the last line for the interactive
# browser.

using CairoMakie
using MatrixFreeKPSolver
using StateRendering

CairoMakie.activate!(type = "png")
outdir = joinpath(@__DIR__, "figures")
mkpath(outdir)

# ── the model ──────────────────────────────────────────────────────────────

grid = Grid((32, 32, 32), (24.0, 24.0, 24.0), Dirichlet())     # nm
geom = LensDot((12.0, 12.0, 10.0), 7.0, 4.0)                   # base radius, height
δ    = 1.5 * minimum(grid.dx)                                  # smooth interface

H = eight_band_hamiltonian(grid, geom, InAs, GaAs; δ = δ)

# ── solve both branches ────────────────────────────────────────────────────
#
# Band-edge states are INTERIOR eigenvalues of the coupled operator, so the
# Schur complement is the method of choice; `branch` picks which side of the gap.

@info "electrons…"
Ee, ψe, ie = eigenstates(H, 4; method = SchurComplement(branch = :conduction))
@info "holes…"
Eh, ψh, ih = eigenstates(H, 4; method = SchurComplement(branch = :valence))

electrons = state_bundle(H, Ee, ψe; kind = :electron, geometry = geom, info = ie)
holes     = state_bundle(H, Eh, ψh; kind = :hole,     geometry = geom, info = ih)

display(electrons)
@info "gap" E_gap = Ee[1] - Eh[1]

# ── figures ────────────────────────────────────────────────────────────────

save(joinpath(outdir, "e1.png"),         plot_state(electrons, 1))
save(joinpath(outdir, "electrons.png"),  plot_states(electrons; ncols = 4))
save(joinpath(outdir, "e1_slices.png"),  plot_slices(electrons, 1))
save(joinpath(outdir, "e1_bands.png"),   plot_bands(electrons, 1))
save(joinpath(outdir, "h1_bands.png"),   plot_bands(holes, 1))
save(joinpath(outdir, "character.png"),  plot_composition(holes))
save(joinpath(outdir, "geometry.png"),
     plot_field(electrons, ScalarField(electrons.geometry, "χ (InAs fraction)")))

# Both carriers on one axis, in the same box — the overlap that decides the
# oscillator strength. `color = :kind` overrides the component colouring, which
# would otherwise put an S electron and an HH hole in unrelated hues.
fig = Figure(size = (620, 560))
ax  = state_axis3(fig[1, 1]; title = "e₁ (blue) and h₁ (red), 50% of the charge")
add_density!(ax, electrons, 1; fraction = 0.5, color = :kind)
add_density!(ax, holes,     1; fraction = 0.5, color = :kind)
add_geometry!(ax, electrons)
save(joinpath(outdir, "overlap.png"), fig)

# A level diagram of both ladders at once.
both = state_bundle(H, vcat(Ee, Eh), vcat(ψe, ψh);
                    kind = vcat(fill(:electron, 4), fill(:hole, 4)),
                    geometry = geom)
save(joinpath(outdir, "spectrum.png"), plot_spectrum(both))

@info "wrote figures" outdir

# ── interactive ────────────────────────────────────────────────────────────
#
# using GLMakie; GLMakie.activate!()
# explore(both)
#
# With strain and polarisation from an elasticity solve or a file:
#
# explore(both; strain = ε, vectors = P)
