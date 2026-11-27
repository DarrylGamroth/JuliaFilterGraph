"""
Finite-only Float32 pixel validity. Each output is `isfinite(input[i])`.
This operation preserves pixel units and applies no detector calibration,
saturation policy, or static bad-pixel mask. The supplied input schema identifies
those units; the existing pixel-validity schema describes the Boolean output.
"""
@algorithm PixelValidityF32 begin
    label = "pixel-validity-f32"

    """Pixel extent and scientific input schema for finite-only validity."""
    @configuration PixelValidityF32Configuration begin
        extent::Int => graph_rebuild
        input_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), configuration.input_schema),
        output(:validity, Bool, (configuration.extent,), CALIBRATED_PIXEL_VALIDITY_V1;
            metadata_from=:input),
    )

    prepare(configuration) = (; extent=configuration.extent)

    process!(plan) = begin
        Base.mightalias(validity, input) && throw(ArgumentError(
            "pixel validity output must not alias input pixels"))
        for index in eachindex(validity, input)
            validity[index] = isfinite(input[index])
        end
        nothing
    end
end
