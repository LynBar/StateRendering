# ---------------------------------------------------------------------------
# The HDF5 loader.
#
# Written against a synthetic file rather than a real one, so the test can assert
# on values it chose: in particular that the axis transpose is undone (a file
# written row-major arrives reversed, and a symmetric test array would hide a
# wrong permutation), and that the band labels are the ones the CALLER passed —
# a file stores an array, not what its band axis means.
# ---------------------------------------------------------------------------

using HDF5

const NX, NY, NZ = 6, 7, 8      # deliberately unequal: a bad transpose throws
const NB, NE, NH = 8, 3, 2

# distinguishable in every direction, so a permutation error is visible
ramp() = [i + 10j + 100k for i in 1:NX, j in 1:NY, k in 1:NZ] .* 1.0

function write_synthetic(path)
    χ = ramp() ./ maximum(ramp())
    ψe = [ComplexF64[(s + i + 2j + 3k + 4b) for i in 1:NX, j in 1:NY, k in 1:NZ, b in 1:NB]
          for s in 1:NE]
    ψh = [ComplexF64[(10s + i - j + k - b) for i in 1:NX, j in 1:NY, k in 1:NZ, b in 1:NB]
          for s in 1:NH]

    # the file's own layout: (nstates, nbands, Nz, Ny, Nx)
    pack(ψs) = permutedims(cat(ψs...; dims = 5), (5, 4, 3, 2, 1))
    tr(a) = permutedims(a, (3, 2, 1))

    h5open(path, "w") do f
        f["estruct/el_eigenvectors_5d"] = pack(ψe)
        f["estruct/el_eigenvalues"] = [1.0, 1.2, 1.35]
        f["estruct/ho_eigenvectors_5d"] = pack(ψh)
        f["estruct/ho_eigenvalues"] = [-0.02, -0.06]
        f["geometry/composition"] = tr(χ)
        f["piezo/potential"] = tr(ramp() .- 400.0)
        f["piezo/Px"] = tr(fill(3.0, NX, NY, NZ))
        f["piezo/Py"] = tr(fill(4.0, NX, NY, NZ))
        f["piezo/Pz"] = tr(zeros(NX, NY, NZ))
        for (ij, v) in (("XX", 0.01), ("YY", 0.02), ("ZZ", -0.03),
                        ("XY", 0.0), ("XZ", 0.0), ("YZ", 0.0))
            f["strain/e" * ij] = tr(fill(v, NX, NY, NZ))
        end
    end
    return χ, ψe, ψh
end

@testset "HDF5 loader" begin
    path = joinpath(OUT, "synthetic.h5")
    χ, ψe, ψh = write_synthetic(path)

    lbl = ["s↑", "pₓ↑", "p_y↑", "p_z↑", "s↓", "pₓ↓", "p_y↓", "p_z↓"]
    basis = BandBasis(lbl, "s" => [1, 5], "pₓ" => [2, 6],
                           "p_y" => [3, 7], "p_z" => [4, 8])

    el, ho, ov = load_h5(path, basis; L = (12.0, 14.0, 16.0))

    @testset "states and axes survive the round trip" begin
        @test length(el) == NE && length(ho) == NH
        @test el.kind == fill(:electron, NE)
        @test ho.kind == fill(:hole, NH)
        @test el.energies ≈ [1.0, 1.2, 1.35]
        @test nbands(el) == NB
        # the transpose is undone: unequal axes make a wrong permutation fatal,
        # and this also pins which axis is which
        @test Base.size(el.density[1]) == (NX, NY, NZ, NB)
        @test el.states !== nothing
        @test el.states[1] ≈ ψe[1] ./ sqrt(sum(abs2, ψe[1])) ||
              el.states[1] ≈ ψe[1]           # amplitudes are stored as read
        @test length(el.x) == NX && length(el.y) == NY && length(el.z) == NZ
        @test el.x[end] ≈ 12.0 * (NX - 1) / NX      # L honoured, in nm
    end

    @testset "labels come from the caller, not the file" begin
        @test el.band_labels == lbl
        @test el.component_labels == ["s", "pₓ", "p_y", "p_z"]
        @test ho.band_labels == lbl
        # and they reach the figure
        @test any(t -> startswith(t, "p_z↓"), axis_titles(plot_bands(el, 1)))
        @test !any(t -> occursin("HH", t), axis_titles(plot_bands(el, 1)))
        @test el.metadata[:operator] == "BandBasis"
    end

    @testset "overlay fields" begin
        @test ov.geometry ≈ χ                      # transpose undone here too
        @test el.geometry ≈ χ
        @test ov.strain isa StrainField
        @test all(hydrostatic(ov.strain) .≈ 0.0)   # 0.01 + 0.02 − 0.03
        @test all(biaxial(ov.strain) .≈ -0.045)
        @test ov.potential isa ScalarField
        @test ov.potential.signed
        @test all(magnitude(ov.polarization) .≈ 5.0)
        # straight into a figure, on the same grid
        @test plot_field(el, ov.strain) isa Figure
        @test explore(el; strain = ov.strain, vectors = ov.polarization) isa Figure
    end

    @testset "missing datasets and missing box" begin
        partial = joinpath(OUT, "partial.h5")
        h5open(partial, "w") do f
            f["geometry/composition"] = permutedims(χ, (3, 2, 1))
        end
        e, h, o = @test_logs (:warn,) match_mode = :any load_h5(partial, basis)
        @test e === nothing && h === nothing        # no states in the file
        @test o.strain === nothing && o.polarization === nothing
        @test o.geometry ≈ χ

        empty = joinpath(OUT, "empty.h5")
        h5open(f -> (f["nothing"] = [1.0]), empty, "w")
        @test_throws ArgumentError load_h5(empty, basis)

        # a basis with the wrong number of bands must be refused
        wrong = BandBasis(lbl[1:6], "s" => [1, 4], "p" => [2, 5], "q" => [3, 6])
        @test_throws DimensionMismatch load_h5(path, wrong; L = (12.0, 14.0, 16.0))
    end
end
