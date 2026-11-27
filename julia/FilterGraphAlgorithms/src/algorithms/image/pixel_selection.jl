"""
Prepared selection of calibrated detector pixels in logical row-major order.

The Plan copies the mask and prepares row and column coordinates once. It does
not apply saturation, finite-value, or validity policy; those belong to the
Algorithm that produced the input image.
"""
struct PixelSelectionPlan <: AbstractAlgorithmPlan
    image_size::NTuple{2,Int}
    mask::Matrix{Bool}
    selected_rows::Vector{Int}
    selected_columns::Vector{Int}
    input_schema::String
    output_schema::String
    _row_major_runs::Union{Nothing,Vector{UnitRange{Int}}}
    _selected_count::Int
end

PixelSelectionPlan(image_size, mask, selected_rows, selected_columns, input_schema, output_schema) =
    PixelSelectionPlan(
        image_size, mask, selected_rows, selected_columns, input_schema, output_schema, nothing, 0,
    )

function _pixel_selection_mask(values, rows::Int, columns::Int)
    count = Base.Checked.checked_mul(rows, columns)
    length(values) == count || throw(
        DimensionMismatch("pixel-selection mask must contain $count row-major values"),
    )
    mask = Matrix{Bool}(undef, rows, columns)
    for row in 1:rows, column in 1:columns
        mask[row, column] = values[(row - 1) * columns + column]
    end
    return mask
end

function PixelSelectionPlan(
    image_size::NTuple{2,Int},
    mask::AbstractMatrix{Bool},
    input_schema::AbstractString,
    output_schema::AbstractString,
)
    rows, columns = image_size
    rows > 0 && columns > 0 ||
        throw(ArgumentError("image dimensions must be positive"))
    Base.checked_mul(rows, columns)
    Base.require_one_based_indexing(mask)
    size(mask) == image_size ||
        throw(DimensionMismatch("pixel-selection mask shape does not match the image"))

    selected_count = count(mask)
    selected_count > 0 ||
        throw(ArgumentError("pixel-selection mask must select pixels"))
    selected_rows = Vector{Int}(undef, selected_count)
    selected_columns = Vector{Int}(undef, selected_count)
    # Use indexed reads when the mask has too many short runs to copy efficiently.
    max_runs = selected_count ÷ 32
    runs = iszero(max_runs) ? nothing : UnitRange{Int}[]
    selected_index = 1
    for row in 1:rows, column in 1:columns
        if mask[row, column]
            selected_rows[selected_index] = row
            selected_columns[selected_index] = column
            selected_index += 1
            isnothing(runs) && continue
            pixel = (row - 1) * columns + column
            if !isempty(runs) && last(runs[end]) == pixel - 1
                runs[end] = first(runs[end]):pixel
            elseif length(runs) < max_runs
                push!(runs, pixel:pixel)
            else
                runs = nothing
            end
        end
    end
    return PixelSelectionPlan(
        image_size,
        Matrix(mask),
        selected_rows,
        selected_columns,
        String(input_schema),
        String(output_schema),
        runs,
        selected_count,
    )
end

function _replace_pixel_selection_mask(plan::PixelSelectionPlan, values)
    replacement = PixelSelectionPlan(
        plan.image_size,
        values,
        plan.input_schema,
        plan.output_schema,
    )
    length(replacement.selected_rows) == length(plan.selected_rows) || throw(
        DimensionMismatch("replacement mask changes the selected-pixel count"),
    )
    return replacement
end

@inline function _pixel_selection_overlaps(left, right)
    a, b = UInt(pointer(left)), UInt(pointer(right))
    return a <= b ? b - a < UInt(sizeof(left)) : a - b < UInt(sizeof(right))
end

# The exported coordinate vectors can be edited. Verify their contents before
# using the prepared runs, including when the coordinates were read via getfield.
function _pixel_selection_matches_runs(
    selected_rows, selected_columns, runs, columns, pixel_count,
)
    changed = 0
    selected = 0
    previous_end = 0
    count = length(selected_rows)
    for run in runs
        first(run) > previous_end && first(run) <= last(run) <= pixel_count || return false
        length(run) <= count - selected || return false
        previous_end = last(run)
        row, column = divrem(first(run) - 1, columns)
        remaining = length(run)
        while remaining > 0
            n = min(columns - column, remaining)
            @inbounds @simd for i in 1:n
                changed |= xor(selected_rows[selected + i], row + 1) |
                           xor(selected_columns[selected + i], column + i)
            end
            remaining -= n
            selected += n
            row += 1
            column = 0
        end
    end
    return selected == count && iszero(changed)
end

function _can_copy_pixel_selection_runs(output::Vector{Float32}, plan, image, runs)
    n = length(output)
    (size(image) == plan.image_size && n == plan._selected_count &&
     length(plan.selected_rows) == length(plan.selected_columns) == n) || return false
    GC.@preserve output plan begin
        (_pixel_selection_overlaps(output, plan.selected_rows) ||
         _pixel_selection_overlaps(output, plan.selected_columns) ||
         _pixel_selection_overlaps(output, runs)) && return false
    end
    return _pixel_selection_matches_runs(
        plan.selected_rows, plan.selected_columns, runs, size(image, 2), length(image),
    )
end

@inline _select_pixels!(output, plan::PixelSelectionPlan, image) =
    _select_pixels_cartesian!(output, plan, image)

@inline function _select_pixels_cartesian!(output, plan::PixelSelectionPlan, image)
    @inbounds for selected in eachindex(plan.selected_rows)
        output[selected] = image[plan.selected_rows[selected], plan.selected_columns[selected]]
    end
    return nothing
end

function _copy_selected_pixel_runs!(
    output::Ptr{Float32}, image::Ptr{Float32}, runs::Vector{UnitRange{Int}},
)
    for run in runs
        unsafe_copyto!(output, image + (first(run) - 1) * sizeof(Float32), length(run))
        output += length(run) * sizeof(Float32)
    end
    return nothing
end

"""
Select masked Float32 pixels from a complete image into a packed vector in
the prepared mask order. Selected values retain the image pixel's units.
"""
@algorithm PixelSelectionF32 begin
    label = "pixel-selection-f32"

    """
    Set image rows/columns, a Boolean mask of that image, and versioned input and
    output schemas. The number of true mask entries fixes the output extent;
    changing that extent requires graph reconstruction.
    """
    @configuration PixelSelectionF32Configuration begin
        image_size::NTuple{2,Int} => graph_rebuild
        mask::Vector{Bool} => graph_rebuild
        input_schema::String => graph_rebuild
        output_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(
            :image,
            Float32,
            configuration.image_size,
            configuration.input_schema,
        ),
        output(
            :selected_pixels,
            Float32,
            (count(configuration.mask),),
            configuration.output_schema;
            name="selected-pixels",
            metadata_from=:image,
        ),
        parameter(
            :mask,
            schema_element_type(PIXEL_SELECTION_MASK_V1),
            configuration.image_size,
            PIXEL_SELECTION_MASK_V1;
            replace=(plan, values) -> _replace_pixel_selection_mask(plan, values),
        ),
    )

    prepare(configuration) = begin
        rows = configuration.image_size[1]
        columns = configuration.image_size[2]
        PixelSelectionPlan(
            (rows, columns),
            _pixel_selection_mask(configuration.mask, rows, columns),
            configuration.input_schema,
            configuration.output_schema,
        )
    end

    process!(plan) = begin
        Base.mightalias(selected_pixels, image) && throw(
            ArgumentError("selected-pixels output must not alias the detector image"),
        )
        _select_pixels!(selected_pixels, plan, image)
    end
end
