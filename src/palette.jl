# ---------------------------------------------------------------------------
# Colours keyed by BAND LABEL, never by band index.
#
# The reason is the same one that motivates `StateBundle` carrying its labels:
# component 2 of a zinc-blende operator is HH, of a wurtzite operator is X. A
# palette indexed positionally would silently give the same colour to different
# physics, and two figures side by side would be uncomparable. Here "HH" is the
# same blue in every figure ever drawn, and an unrecognised label falls through
# to a stable, colourblind-safe cycle rather than to an error.
# ---------------------------------------------------------------------------

const _COMPONENT_COLORS = Dict{String,RGBf}(
    # conduction
    "S"   => RGBf(0.84, 0.19, 0.15),
    "s"   => RGBf(0.84, 0.19, 0.15),
    # zinc-blende valence, |j,mⱼ⟩
    "HH"  => RGBf(0.12, 0.47, 0.71),
    "LH"  => RGBf(0.17, 0.63, 0.17),
    "SO"  => RGBf(0.58, 0.40, 0.74),
    # wurtzite / SXYZ valence
    "X"   => RGBf(0.12, 0.47, 0.71),
    "Y"   => RGBf(0.17, 0.63, 0.17),
    "Z"   => RGBf(0.90, 0.62, 0.00),
    "pₓ"  => RGBf(0.12, 0.47, 0.71),
    "p_y" => RGBf(0.17, 0.63, 0.17),
    "p_z" => RGBf(0.90, 0.62, 0.00),
    "px"  => RGBf(0.12, 0.47, 0.71),
    "py"  => RGBf(0.17, 0.63, 0.17),
    "pz"  => RGBf(0.90, 0.62, 0.00),
)

# Wong's colourblind-safe eight, for labels the table does not know.
const _FALLBACK = RGBf[
    RGBf(0.00, 0.45, 0.70), RGBf(0.90, 0.62, 0.00), RGBf(0.00, 0.62, 0.45),
    RGBf(0.80, 0.47, 0.65), RGBf(0.34, 0.71, 0.91), RGBf(0.84, 0.37, 0.00),
    RGBf(0.95, 0.90, 0.25), RGBf(0.35, 0.35, 0.35),
]

"""
    component_color(label) -> RGBf
    component_palette(labels) -> Vector{RGBf}

Colour for an orbital component, keyed by its name. `S`/`s` is red, `HH`/`X` blue,
`LH`/`Y` green, `SO` purple, `Z` amber — so the same component is the same colour
in every figure, and a zinc-blende and a wurtzite figure can be read side by side.

Unknown labels take colours from a colourblind-safe cycle, assigned in order of
first appearance so a palette is reproducible.
"""
component_color(label::AbstractString) = get(_COMPONENT_COLORS, String(label), nothing)

function component_palette(labels)
    out = Vector{RGBf}(undef, length(labels))
    seen = Dict{String,RGBf}()          # one colour per label, not per position
    for (i, l) in pairs(labels)
        c = component_color(l)
        if c === nothing
            key = String(l)
            c = get!(seen, key) do
                _FALLBACK[mod1(length(seen) + 1, length(_FALLBACK))]
            end
        end
        out[i] = c
    end
    return out
end

"""
    band_palette(bundle) -> Vector{RGBf}

One colour per spinor band: the colour of the component it belongs to, with the
spin-down partner darkened. Pair with `bundle.band_labels`.
"""
function band_palette(b::StateBundle)
    comp = component_palette(b.component_labels)
    out  = Vector{RGBf}(undef, nbands(b))
    for (c, group) in pairs(b.component_groups)
        for (k, band) in pairs(sort(group))
            # first band of a component keeps the colour, later ones darken:
            # for a spin pair this reads as ↑ bright, ↓ dark.
            f = 1.0 - 0.35 * (k - 1) / max(length(group) - 1, 1)
            out[band] = RGBf(comp[c].r * f, comp[c].g * f, comp[c].b * f)
        end
    end
    return out
end

"""
    kind_color(kind) -> RGBf

Electron blue, hole red. Used when two carrier densities share an axis, where
component colours would compete with the carrier distinction.
"""
kind_color(kind::Symbol) = kind === :hole ? RGBf(0.86, 0.20, 0.27) :
                                            RGBf(0.16, 0.44, 0.78)

# Diverging map for signed fields (strain, piezoelectric potential): blue for
# negative, transparent at zero, red for positive. Alpha is a parameter because
# these are always drawn on top of something else.
diverging_map(alpha::Real) = [RGBAf(0.0, 0.3, 1.0, alpha),
                              RGBAf(0.0, 0.0, 0.0, 0.0),
                              RGBAf(1.0, 0.2, 0.0, alpha)]
