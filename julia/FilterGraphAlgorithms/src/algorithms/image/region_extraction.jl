"""Prepared zero-based origins for equal rectangular detector regions."""
struct RegionExtractionPlan <: AbstractAlgorithmPlan
    image_size::NTuple{2,Int}
    region_size::NTuple{2,Int}
    origins::Vector{NTuple{2,Int}}
end

function RegionExtractionPlan(
    image_size::NTuple{2,Int},
    origins,
    region_size::NTuple{2,Int},
)
    rows, columns = image_size
    region_rows, region_columns = region_size
    rows > 0 && columns > 0 ||
        throw(ArgumentError("image dimensions must be positive"))
    region_rows > 0 && region_columns > 0 ||
        throw(ArgumentError("region dimensions must be positive"))
    isempty(origins) && throw(ArgumentError("region origins must be nonempty"))

    prepared = Vector{NTuple{2,Int}}(undef, length(origins))
    for (index, origin) in enumerate(origins)
        length(origin) == 2 || throw(
            DimensionMismatch("region $(index - 1) origin must contain row and column"),
        )
        first_row = Int(origin[1])
        first_column = Int(origin[2])
        first_row >= 0 && first_column >= 0 || throw(
            ArgumentError("region $(index - 1) origin must be nonnegative"),
        )
        last_row = Base.Checked.checked_add(first_row, region_rows)
        last_column = Base.Checked.checked_add(first_column, region_columns)
        last_row <= rows && last_column <= columns || throw(
            ArgumentError("region $(index - 1) is outside the detector image"),
        )
        prepared[index] = (first_row, first_column)
    end
    return RegionExtractionPlan(image_size, region_size, prepared)
end

function _configured_region_origins(
    values::AbstractVector{<:AbstractVector{<:Integer}},
)
    return map(values) do origin
        Base.require_one_based_indexing(origin)
        length(origin) == 2 || throw(
            DimensionMismatch("each region origin must contain row and column"),
        )
        (origin[1], origin[2])
    end
end

function _replace_region_origins(plan::RegionExtractionPlan, values)
    origins = map(axes(values, 1)) do region
        (Int(values[region, 1]), Int(values[region, 2]))
    end
    return RegionExtractionPlan(plan.image_size, origins, plan.region_size)
end

"""
Extract rectangular Float32 regions from a calibrated detector image, with
each region addressed by its configured origin. Region pixels retain the
input's calibrated-pixel units.
"""
@algorithm RegionExtractionF32 begin
    label = "region-extraction-f32"

    """
    Set image and region rows/columns plus one two-coordinate origin per output
    region. The number of origins fixes the first output dimension.
    """
    @configuration RegionExtractionF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        region_size::NTuple{2,Int} => graph_rebuild
        origins::Vector{Vector{Int}} => graph_rebuild
    end

    ports(configuration) = (
        input(
            :calibrated,
            schema_element_type(CALIBRATED_PIXELS_V1),
            configuration.image_size,
            CALIBRATED_PIXELS_V1,
        ),
        output(
            :regions,
            schema_element_type(REGION_PIXELS_V1),
            (
                length(configuration.origins),
                configuration.region_size[1],
                configuration.region_size[2],
            ),
            REGION_PIXELS_V1;
            metadata_from=:calibrated,
        ),
        parameter(
            :origins,
            schema_element_type(REGION_ORIGINS_V1),
            (length(configuration.origins), 2),
            REGION_ORIGINS_V1;
            replace=(plan, values) -> _replace_region_origins(plan, values),
        ),
    )

    prepare(configuration) = RegionExtractionPlan(
        configuration.image_size,
        _configured_region_origins(configuration.origins),
        configuration.region_size,
    )

    process!(plan) = begin
        Base.mightalias(regions, calibrated) && throw(
            ArgumentError("region output must not alias the detector image"),
        )
        region_rows, region_columns = plan.region_size
        @inbounds for region in eachindex(plan.origins)
            first_row, first_column = plan.origins[region]
            for row in 1:region_rows, column in 1:region_columns
                regions[region, row, column] =
                    calibrated[first_row + row, first_column + column]
            end
        end
        nothing
    end
end
