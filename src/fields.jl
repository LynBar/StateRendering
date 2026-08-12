# ---------------------------------------------------------------------------
# Overlay fields.
#
# The solver produces states, densities and χ(r). Strain, piezoelectric potential
# and polarisation come from somewhere else — a continuum-elasticity solve, an
# atomistic code, or a file written by another package. This module gives them a
# typed home so a plotting call cannot confuse a strain component for a potential,
# and so the sign convention (diverging colour map about zero, or not) travels
# with the data instead of being a keyword the caller has to remember.
# ---------------------------------------------------------------------------

"""
    ScalarField(data, name; signed = any(<(0), data), unit = "")

A scalar field on the state grid: piezoelectric potential, a strain component, a
confinement profile. `signed = true` selects a diverging colour map about zero
and symmetric ± isosurfaces; `false` selects a sequential map.

`data` must have the same size as the states it is drawn with, which is checked
at draw time.
"""
struct ScalarField
    data::Array{Float64,3}
    name::String
    signed::Bool
    unit::String
end

function ScalarField(data::AbstractArray{<:Real,3}, name::AbstractString;
                     signed::Union{Bool,Nothing} = nothing, unit::AbstractString = "")
    d = Array{Float64}(data)
    return ScalarField(d, String(name),
                       signed === nothing ? any(<(0), d) : signed, String(unit))
end

Base.size(f::ScalarField) = size(f.data)
Base.show(io::IO, f::ScalarField) = print(io, "ScalarField(", f.name, ", ",
    join(size(f.data), "×"), f.signed ? ", signed" : "", ")")

"""
    VectorField(x, y, z, name)

A three-component field on the state grid — polarisation, an electric field, a
displacement. Drawn as arrows on a plane, coloured by magnitude.
"""
struct VectorField
    x::Array{Float64,3}
    y::Array{Float64,3}
    z::Array{Float64,3}
    name::String
end

VectorField(x, y, z, name::AbstractString = "P") =
    VectorField(Array{Float64}(x), Array{Float64}(y), Array{Float64}(z), String(name))

VectorField(components::NTuple{3,<:AbstractArray}, name::AbstractString = "P") =
    VectorField(components..., name)

Base.size(v::VectorField) = size(v.x)

"""
    magnitude(v::VectorField) -> Array{Float64,3}
"""
magnitude(v::VectorField) = sqrt.(v.x .^ 2 .+ v.y .^ 2 .+ v.z .^ 2)

Base.show(io::IO, v::VectorField) = print(io, "VectorField(", v.name, ", ",
    join(size(v.x), "×"), ", |v|max=", round(maximum(magnitude(v)); sigdigits = 3), ")")

"""
    StrainField(; εxx, εyy, εzz, εxy, εxz, εyz)

The symmetric strain tensor on the state grid, taken by **keyword** so the six
components cannot be transposed into the wrong slots — the usual failure mode
when a tensor is passed as a bare six-element vector in whatever order the file
happened to use.

Derived quantities: [`hydrostatic`](@ref) and [`biaxial`](@ref), which are what
actually shift band edges, and [`scalar_fields`](@ref) for the full menu.
"""
struct StrainField
    εxx::Array{Float64,3}
    εyy::Array{Float64,3}
    εzz::Array{Float64,3}
    εxy::Array{Float64,3}
    εxz::Array{Float64,3}
    εyz::Array{Float64,3}
end

function StrainField(; εxx, εyy, εzz, εxy, εxz, εyz)
    a = (εxx, εyy, εzz, εxy, εxz, εyz)
    allequal(size.(a)) || throw(DimensionMismatch(
        "strain components have sizes $(size.(a)); all six must match"))
    return StrainField(Array{Float64}.(a)...)
end

Base.size(s::StrainField) = size(s.εxx)

"""
    hydrostatic(s::StrainField) -> Array{Float64,3}

`εxx + εyy + εzz`, the trace. Shifts both band edges and so moves the gap.
"""
hydrostatic(s::StrainField) = s.εxx .+ s.εyy .+ s.εzz

"""
    biaxial(s::StrainField) -> Array{Float64,3}

`εzz − (εxx + εyy)/2`. Splits heavy and light hole, so it is the component that
decides which valence band is on top.
"""
biaxial(s::StrainField) = s.εzz .- (s.εxx .+ s.εyy) ./ 2

"""
    scalar_fields(s::StrainField) -> Vector{ScalarField}

The two invariants first, then the six tensor components — the order a menu
should offer them in, since the invariants are what carry physical meaning.
"""
scalar_fields(s::StrainField) = ScalarField[
    ScalarField(hydrostatic(s), "hydrostatic"; signed = true),
    ScalarField(biaxial(s),     "biaxial";     signed = true),
    ScalarField(s.εxx, "εxx"; signed = true), ScalarField(s.εyy, "εyy"; signed = true),
    ScalarField(s.εzz, "εzz"; signed = true), ScalarField(s.εxy, "εxy"; signed = true),
    ScalarField(s.εxz, "εxz"; signed = true), ScalarField(s.εyz, "εyz"; signed = true),
]

Base.show(io::IO, s::StrainField) = print(io, "StrainField(", join(size(s), "×"),
    ", hydro ∈ ", round.(extrema(hydrostatic(s)); sigdigits = 3), ")")

# A bundle's geometry χ(r) is a scalar field like any other; this lets it be
# drawn through the same path when a caller wants it as a volume rather than an
# isosurface.
geometry_field(b::StateBundle) = b.geometry === nothing ? nothing :
    ScalarField(b.geometry, "χ"; signed = false)

_check_grid(f, b::StateBundle, what) =
    size(f) == (length(b.x), length(b.y), length(b.z)) || throw(DimensionMismatch(
        "$what is $(join(size(f), "×")) but the state grid is " *
        "$(length(b.x))×$(length(b.y))×$(length(b.z))"))
