# ---------------------------------------------------------------------------
# Turning a density into something drawable.
#
# The one idea worth stating: an isosurface should be chosen by ENCLOSED
# PROBABILITY, not as a fraction of the peak. "isovalue = 0.7·max" describes the
# array, not the physics — it encloses wildly different amounts of charge for a
# peaked s-like state and a diffuse p-like one, so two such surfaces cannot be
# compared, between states or between figures. "the surface containing 50% of
# the charge" is the same statement about every state, and is what you want when
# comparing an electron against a hole, or the same state across grid sizes.
# ---------------------------------------------------------------------------

"""
    isolevel(d, fraction = 0.5) -> Float64

The isosurface level that encloses `fraction` of the total weight in `d`: the
largest `λ` with `sum(d[d .≥ λ]) ≥ fraction · sum(d)`.

    lvl = isolevel(state_density(b, 1), 0.5)   # the surface holding half the charge

Comparable across states, grids and materials, which `fraction · maximum(d)` is
not.
"""
function isolevel(d::AbstractArray{<:Real}, fraction::Real = 0.5)
    0 < fraction ≤ 1 || throw(ArgumentError("fraction must be in (0, 1], got $fraction"))
    total = sum(d)
    total > 0 || return 0.0
    s = sort(vec(Float64.(d)); rev = true)
    target = fraction * total
    acc = 0.0
    @inbounds for x in s
        acc += x
        acc ≥ target && return x
    end
    return s[end]
end

"""
    enclosed_fraction(d, level) -> Float64

Fraction of the total weight of `d` inside the `level` isosurface. The inverse of
[`isolevel`](@ref); useful for reporting what an arbitrary level actually means.
"""
function enclosed_fraction(d::AbstractArray{<:Real}, level::Real)
    total = sum(d)
    total > 0 || return 0.0
    return sum(x for x in d if x ≥ level; init = 0.0) / total
end

"""
    smooth(A; σ = 1.0) -> Array{Float64,3}

Separable Gaussian blur with clamped edges, in grid units. Its purpose is
cosmetic and specific: a sharp composition profile χ(r) is a 0/1 step, whose
isosurface is a staircase of grid facets. One grid spacing of blur turns that
into the smooth interface the model is actually describing.

Do not use it on a density before measuring anything from it.
"""
function smooth(A::AbstractArray{<:Real,3}; σ::Real = 1.0)
    σ > 0 || return Array{Float64}(A)
    r = ceil(Int, 2σ)
    k = [exp(-x^2 / (2σ^2)) for x in -r:r]
    k ./= sum(k)

    out = Array{Float64}(A)
    buf = similar(out)
    n = size(out)
    for dim in 1:3
        fill!(buf, 0.0)
        @inbounds for I in CartesianIndices(out)
            t = Tuple(I)
            s = 0.0
            for (ki, off) in pairs(-r:r)
                j = clamp(t[dim] + off, 1, n[dim])
                s += k[ki] * out[ntuple(d -> d == dim ? j : t[d], 3)...]
            end
            buf[I] = s
        end
        copyto!(out, buf)
    end
    return out
end

"""
    isosurface_mesh(d, x, y, z, level) -> mesh or `nothing`

Marching-cubes surface of `d` at `level` as a `GeometryBasics` mesh with normals,
vertices in the physical coordinates `x`, `y`, `z` (nm). `nothing` when the level
lies outside the data, so a caller can simply skip drawing.

Producing a triangle mesh rather than a GPU volume render is what lets these
figures be drawn by any Makie backend, CairoMakie included — that is, saved as
vector graphics for a paper.
"""
function isosurface_mesh(d::AbstractArray{<:Real,3}, x, y, z, level::Real)
    lo, hi = extrema(d)
    (level ≤ lo || level ≥ hi) && return nothing
    verts, faces = Meshing.isosurface(Float64.(d), Meshing.MarchingCubes(iso = Float64(level)),
                                      collect(Float64, x), collect(Float64, y),
                                      collect(Float64, z))
    isempty(faces) && return nothing
    pts = [Point3f(v[1], v[2], v[3]) for v in verts]
    tri = [GLTriangleFace(f[1], f[2], f[3]) for f in faces]
    return normal_mesh(pts, tri)
end

# A zero-area triangle: what an Observable-driven mesh plot shows when the level
# has been dragged outside the data. Swapping in an empty mesh is not reliable
# across backends, and a degenerate one renders as nothing at all.
_empty_mesh() = normal_mesh(fill(Point3f(0), 3), [GLTriangleFace(1, 2, 3)])
_mesh_or_empty(m) = m === nothing ? _empty_mesh() : m

"""
    centroid(d, x, y, z) -> (x̄, ȳ, z̄)

Weight-centre of `d` in nm. Used to place slice planes through the state rather
than through the middle of the box, which for an off-centre state is empty.
"""
function centroid(d::AbstractArray{<:Real,3}, x, y, z)
    total = sum(d)
    total > 0 || return (x[(end+1)÷2], y[(end+1)÷2], z[(end+1)÷2])
    cx = cy = cz = 0.0
    @inbounds for k in axes(d, 3), j in axes(d, 2), i in axes(d, 1)
        w = d[i, j, k]
        cx += w * x[i]; cy += w * y[j]; cz += w * z[k]
    end
    return (cx / total, cy / total, cz / total)
end

"""
    nearest_index(coords, value) -> Int

Index of the grid point closest to `value`.
"""
nearest_index(coords, value) = argmin(abs.(collect(coords) .- value))

# axis index and label for a plane normal, used by the slicing figures
_axis_of(plane::Symbol) = plane === :x ? 1 : plane === :y ? 2 : plane === :z ? 3 :
    throw(ArgumentError("plane must be :x, :y or :z, got :$plane"))

_inplane_axes(plane::Symbol) = plane === :x ? (2, 3) : plane === :y ? (1, 3) : (1, 2)
const _AXIS_NAMES = ("x", "y", "z")
