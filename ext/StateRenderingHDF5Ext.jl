module StateRenderingHDF5Ext

# ---------------------------------------------------------------------------
# Reading state files written by another solver.
#
# Only loaded when HDF5 is, so the package itself stays free of it. The one
# non-negotiable argument is the band basis: a file records an array, not what
# its band axis means, and this package refuses to guess. Pass a `BandBasis`
# naming the bands in the file's own order and the labels then travel correctly
# through every figure.
# ---------------------------------------------------------------------------

using HDF5
using StateRendering: ScalarField, VectorField, StrainField
using MatrixFreeKPSolver: BandBasis, Grid, Dirichlet, state_bundle
import StateRendering: load_h5

# HDF5 written from a row-major writer arrives with the axes reversed.
_read3(f, path) = real.(permutedims(Array(f[path]), (3, 2, 1)))

# `haskey` on a slash-separated path throws if an intermediate group is missing,
# so walk it link by link and let an absent dataset simply mean "not supplied".
function _has(f, path::AbstractString)
    node = f
    for part in split(path, '/'; keepempty = false)
        (node isa HDF5.File || node isa HDF5.Group) || return false
        haskey(node, part) || return false
        node = node[part]
    end
    return true
end

"""
    load_h5(path, basis::BandBasis; kwargs...) -> (electrons, holes, overlays)
    load_h5(file::HDF5.File, basis::BandBasis; kwargs...)

Read an electronic-structure file into two [`StateBundle`](@ref)s and a named
tuple of overlay fields.

`basis` must describe the band axis **as the file stores it** — see
[`BandBasis`](@ref). For an SXYZ, spin-outer 8-band file:

    basis = BandBasis(["s↑","pₓ↑","p_y↑","p_z↑","s↓","pₓ↓","p_y↓","p_z↓"],
                      "s" => [1,5], "pₓ" => [2,6], "p_y" => [3,7], "p_z" => [4,8])
    el, ho, ov = load_h5("Basic.h5", basis)
    explore(el; strain = ov.strain, vectors = ov.polarization)

# Keywords

  * `L` — box side lengths in nm. Files rarely record them; without it the grid
    is labelled 1 nm per point and every length in every figure is wrong by a
    constant factor, so pass it if you have it.
  * `paths` — override the dataset names, which are not standardised. The
    defaults match the layout this package was first used with:
    `estruct/el_eigenvectors_5d`, `estruct/el_eigenvalues`, the `ho_` pair,
    `geometry/composition`, `piezo/potential`, `piezo/{Px,Py,Pz}`,
    `strain/e{XX,XY,XZ,YY,YZ,ZZ}`.

Datasets that are absent are skipped, so a file with no strain simply returns
`overlays.strain === nothing`.
"""
function load_h5(path::AbstractString, basis::BandBasis; kwargs...)
    return h5open(f -> load_h5(f, basis; kwargs...), path, "r")
end

function load_h5(f::HDF5.File, basis::BandBasis;
                 L = nothing,
                 paths = NamedTuple(),
                 keep_states::Bool = true)
    p = merge((el_vec = "estruct/el_eigenvectors_5d",
               el_val = "estruct/el_eigenvalues",
               ho_vec = "estruct/ho_eigenvectors_5d",
               ho_val = "estruct/ho_eigenvalues",
               geometry = "geometry/composition",
               potential = "piezo/potential",
               Px = "piezo/Px", Py = "piezo/Py", Pz = "piezo/Pz",
               strain = "strain/e"), paths)

    χ = _has(f, p.geometry) ? _read3(f, p.geometry) : nothing
    el, Ee = _read_states(f, p.el_vec, p.el_val)
    ho, Eh = _read_states(f, p.ho_vec, p.ho_val)

    N = el !== nothing ? Base.size(first(el))[1:3] :
        ho !== nothing ? Base.size(first(ho))[1:3] :
        χ !== nothing  ? Base.size(χ) :
        throw(ArgumentError("file contains neither states nor geometry"))

    Lnm = L === nothing ? Float64.(N) : L
    L === nothing && @warn "no box size given; the grid is labelled 1 nm per " *
                           "point, so every length in every figure is scaled. " *
                           "Pass L = (Lx, Ly, Lz) in nm."
    grid = Grid(N, Tuple(Float64.(Lnm)), Dirichlet())

    bundle(states, energies, kind) = states === nothing ? nothing :
        state_bundle(basis, grid, energies, states;
                     kind = kind, geometry = χ, keep_states = keep_states,
                     source_file = HDF5.filename(f))

    strain = nothing
    if all(_has(f, p.strain * ij) for ij in ("XX", "YY", "ZZ", "XY", "XZ", "YZ"))
        strain = StrainField(; εxx = _read3(f, p.strain * "XX"),
                               εyy = _read3(f, p.strain * "YY"),
                               εzz = _read3(f, p.strain * "ZZ"),
                               εxy = _read3(f, p.strain * "XY"),
                               εxz = _read3(f, p.strain * "XZ"),
                               εyz = _read3(f, p.strain * "YZ"))
    end
    potential = _has(f, p.potential) ?
        ScalarField(_read3(f, p.potential), "piezo potential"; signed = true, unit = "V") :
        nothing
    polarization = all(_has(f, q) for q in (p.Px, p.Py, p.Pz)) ?
        VectorField(_read3(f, p.Px), _read3(f, p.Py), _read3(f, p.Pz), "P") : nothing

    return (bundle(el, Ee, :electron), bundle(ho, Eh, :hole),
            (; geometry = χ, strain, potential, polarization))
end

# Stored as (nstates, nbands, Nz, Ny, Nx); the bundle wants a vector of
# (Nx, Ny, Nz, nbands) arrays.
function _read_states(f, vec_path, val_path)
    (_has(f, vec_path) && _has(f, val_path)) || return (nothing, nothing)
    raw = Array(f[vec_path])
    E   = Array(f[val_path])
    ψ   = ComplexF64.(permutedims(raw, (1, 5, 4, 3, 2)))
    return ([Array(selectdim(ψ, 1, i)) for i in axes(ψ, 1)], E)
end

end # module StateRenderingHDF5Ext
