const _CRED2_TRIGGER_LOW_TAG = UInt16(0x3ffe)
const _CRED2_TRIGGER_HIGH_TAG = UInt16(0x3fff)
const _CRED2_TRIGGER_TAG_PIXEL = 4

"""Expected acquisition rejection from C‑RED 2 image-tag decoding."""
@enum Cred2ChoppedImageKindFailure::UInt8 begin
    Cred2UnexpectedImageTag = 1
end

@doc "Raw pixel 4 is neither the documented low nor high C‑RED 2 tag word."
Cred2UnexpectedImageTag

"""
    cred2_trigger_metadata(metadata, raw_image)

Decode the C‑RED 2 `Trigger_in` tag into in-band [`SampleMetadata`](@ref).
With camera image tags enabled, the fourth transmitted raw pixel is `0x3FFE`
for low and `0x3FFF` for high. The complete word is validated before its least
significant bit is accepted. An unexpected word returns
[`Cred2UnexpectedImageTag`](@ref) and leaves the supplied metadata unchanged.

Call this ingress function before pixel calibration alters or removes the four
camera tag pixels. Downstream Algorithms receive the annotation through the
ordinary image edge; no one-element classification Port is required.

A raw tag does not establish its temporal scope over the full exposure. Before
publishing this annotation as native whole-exposure `triggerClassification`, the
source adapter must qualify that scope. Optical classification needs an explicit
application mapping and its own exposure qualification. This helper does not
qualify a physical camera.
"""
function cred2_trigger_metadata(
    metadata::SampleMetadata,
    raw_image::AbstractArray{UInt16},
)
    Base.require_one_based_indexing(raw_image)
    length(raw_image) >= _CRED2_TRIGGER_TAG_PIXEL || throw(
        DimensionMismatch("C-RED 2 tagged frames require at least four pixels"),
    )
    tag = raw_image[_CRED2_TRIGGER_TAG_PIXEL]
    trigger_high = if tag == _CRED2_TRIGGER_HIGH_TAG
        true
    elseif tag == _CRED2_TRIGGER_LOW_TAG
        false
    else
        return Cred2UnexpectedImageTag
    end
    return SampleMetadata(metadata; trigger_high)
end
