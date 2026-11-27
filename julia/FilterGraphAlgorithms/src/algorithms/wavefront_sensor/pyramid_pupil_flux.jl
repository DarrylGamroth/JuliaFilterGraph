"""Prepared detector geometry and exact-sum bound for pyramid pupil flux."""
struct PyramidPupilFluxPlan <: AbstractAlgorithmPlan
    regions::RegionExtractionPlan
end

function PyramidPupilFluxPlan(image_size, pupil_size, pupil_origins)
    regions = RegionExtractionPlan(image_size, pupil_origins, pupil_size)
    pupil_pixels = Base.checked_mul(pupil_size...)
    maximum_sum = Base.Checked.checked_mul(UInt64(pupil_pixels), UInt64(typemax(UInt16)))
    maximum_sum <= UInt64(1) << 53 || throw(ArgumentError(
        "pupil region can exceed the exact-integer range of Float64",
    ))
    return PyramidPupilFluxPlan(regions)
end

function _sum_pyramid_pupil(raw, first_row, first_column, rows, columns)
    total = 0.0
    for row in 1:rows, column in 1:columns
        total += Float64(raw[first_row + row - 1, first_column + column - 1])
    end
    return total
end

function _sum_contiguous_pyramid_pupil(raw, first_row, first_column, rows, columns)
    total = 0.0
    # Preparation bounds every nonnegative sum by 2^53, so
    # reassociation preserves exact counts. Rows are contiguous.
    for column in 1:columns
        @simd for row in 1:rows
            total += Float64(raw[first_row + row - 1, first_column + column - 1])
        end
    end
    return total
end

_sum_pyramid_pupil_flux!(flux, regions, raw::AbstractMatrix) =
    _sum_pyramid_pupil_flux!(flux, regions, raw, _sum_pyramid_pupil)

function _sum_pyramid_pupil_flux!(flux, regions, raw::StridedMatrix{UInt16})
    sum_pupil = stride(raw, 1) == 1 ? _sum_contiguous_pyramid_pupil : _sum_pyramid_pupil
    return _sum_pyramid_pupil_flux!(flux, regions, raw, sum_pupil)
end

function _sum_pyramid_pupil_flux!(flux, regions, raw, sum_pupil::F) where {F}
    rows, columns = regions.region_size
    first_image_row = first(axes(raw, 1))
    first_image_column = first(axes(raw, 2))
    for (pupil, output_index) in enumerate(eachindex(flux))
        first_row, first_column = regions.origins[pupil]
        flux[output_index] = sum_pupil(
            raw, first_image_row + first_row, first_image_column + first_column, rows, columns,
        )
    end
    return nothing
end

"""
Sum raw U16 detector ADU values independently over each configured pyramid
pupil rectangle. Results remain exact F64 ADC counts per exposure.
"""
@algorithm PyramidPupilFluxU16F64 begin
    label = "pyramid-pupil-flux-u16-f64"

    """Detector and four-pupil geometry; origins use zero-based row, column order."""
    @configuration PyramidPupilFluxU16F64Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        pupil_size::NTuple{2,Int} => graph_rebuild
        pupil_origins::Vector{Vector{Int}} => graph_rebuild
    end

    ports(configuration) = (
        input(
            :raw,
            schema_element_type(RAW_DETECTOR_PIXELS_V1),
            configuration.image_size,
            RAW_DETECTOR_PIXELS_V1,
        ),
        output(
            :flux,
            schema_element_type(PYRAMID_PUPIL_FLUX_ADU_F64_V1),
            (PYRAMID_PUPIL_COUNT,),
            PYRAMID_PUPIL_FLUX_ADU_F64_V1;
            name="pupil-flux",
            metadata_from=:raw,
        ),
    )

    prepare(configuration) = begin
        origins = _configured_pyramid_image_origins(configuration.pupil_origins)
        PyramidPupilFluxPlan(
            configuration.image_size,
            configuration.pupil_size,
            origins,
        )
    end

    process!(plan) = begin
        size(raw) == plan.regions.image_size || throw(
            DimensionMismatch("raw detector image shape changed after preparation"),
        )
        length(plan.regions.origins) == PYRAMID_PUPIL_COUNT || throw(
            DimensionMismatch("pyramid pupil geometry must contain four regions"),
        )
        ndims(flux) == 1 && length(flux) == PYRAMID_PUPIL_COUNT || throw(
            DimensionMismatch("pupil-flux output must contain four values"),
        )
        _sum_pyramid_pupil_flux!(flux, plan.regions, raw)
    end
end
