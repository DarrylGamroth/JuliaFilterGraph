import FilterGraphAlgorithms:
    PreparedAlgorithm,
    ShackHartmannImageF32,
    ShackHartmannImagePlan,
    _finish_center_of_gravity,
    complete!,
    enqueue!,
    preflight,
    reset!

const SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE = 32

# Bind the backend reduction during preparation; replay has no runtime selection.
struct _ShackHartmannKernel{K,R}
    kernel!::K
    reduction::R
end

@inline function (measurement::_ShackHartmannKernel)(args...; ndrange)
    measurement.kernel!(args..., measurement.reduction; ndrange)
    return nothing
end

_shack_hartmann_kernel(device, backend) = _ShackHartmannKernel(
    _shack_hartmann_image!(backend, SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE),
    nothing,
)

@inline function _reduce_shack_hartmann_moments!(
    local_x_moments,
    local_y_moments,
    local_fluxes,
    local_index::Int,
    stride::Int,
)
    if local_index <= stride
        @inbounds begin
            local_x_moments[local_index] +=
                local_x_moments[local_index + stride]
            local_y_moments[local_index] +=
                local_y_moments[local_index + stride]
            local_fluxes[local_index] += local_fluxes[local_index + stride]
        end
    end
    return nothing
end

@kernel function _shack_hartmann_image!(
    slopes,
    flux,
    validity,
    frame_status,
    image,
    origin_rows,
    origin_columns,
    x_coordinates,
    y_coordinates,
    reference_x,
    reference_y,
    pixel_thresholds,
    flux_thresholds,
    active,
    weights,
    subaperture_rows,
    subaperture_columns,
    reduction,
)
    local_index = @index(Local, Linear)
    _, subaperture = @index(Group, NTuple)
    local_x_moments = @localmem Float32 (
        SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE,
    )
    local_y_moments = @localmem Float32 (
        SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE,
    )
    local_fluxes = @localmem Float32 (
        SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE,
    )
    @uniform frame_valid = @inbounds iszero(frame_status[1])
    @inbounds begin
        origin_row = origin_rows[subaperture]
        origin_column = origin_columns[subaperture]
        pixel_threshold = pixel_thresholds[subaperture]
    end
    # Per-lane X/Y moments and flux survive the synchronized reduction branch.
    x_moment = @private Float32 (1,)
    y_moment = @private Float32 (1,)
    measured_flux = @private Float32 (1,)
    x_moment[1] = 0.0f0
    y_moment[1] = 0.0f0
    measured_flux[1] = 0.0f0
    if frame_valid
        for row in Int(local_index):SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE:subaperture_rows
            for column in 1:subaperture_columns
                coordinate = (row - 1) * subaperture_columns + column
                @inbounds value = image[
                    origin_row + row,
                    origin_column + column,
                ]
                if !isfinite(value) || value >= pixel_threshold
                    weighted_value = if weights === nothing
                        value
                    else
                        @inbounds weights[subaperture, coordinate] * value
                    end
                    @inbounds begin
                        x_moment[1] += weighted_value * x_coordinates[coordinate]
                        y_moment[1] += weighted_value * y_coordinates[coordinate]
                    end
                    measured_flux[1] += weighted_value
                end
            end
        end
    end
    if reduction === nothing
        @inbounds begin
            local_x_moments[local_index] = x_moment[1]
            local_y_moments[local_index] = y_moment[1]
            local_fluxes[local_index] = measured_flux[1]
        end
        @synchronize

        _reduce_shack_hartmann_moments!(
            local_x_moments,
            local_y_moments,
            local_fluxes,
            Int(local_index),
            16,
        )
        @synchronize
        _reduce_shack_hartmann_moments!(
            local_x_moments,
            local_y_moments,
            local_fluxes,
            Int(local_index),
            8,
        )
        @synchronize
        _reduce_shack_hartmann_moments!(
            local_x_moments,
            local_y_moments,
            local_fluxes,
            Int(local_index),
            4,
        )
        @synchronize
        _reduce_shack_hartmann_moments!(
            local_x_moments,
            local_y_moments,
            local_fluxes,
            Int(local_index),
            2,
        )
        @synchronize
        _reduce_shack_hartmann_moments!(
            local_x_moments,
            local_y_moments,
            local_fluxes,
            Int(local_index),
            1,
        )
        @synchronize
        if local_index == 1
            @inbounds begin
                x_moment[1] = local_x_moments[1]
                y_moment[1] = local_y_moments[1]
                measured_flux[1] = local_fluxes[1]
            end
        end
    else
        x_moment[1], y_moment[1], measured_flux[1] = reduction(x_moment[1], y_moment[1], measured_flux[1])
    end

    if frame_valid && local_index == 1
        @inbounds x_offset, y_offset, measured_flux[1], valid =
            _finish_center_of_gravity(
                x_moment[1],
                y_moment[1],
                measured_flux[1],
                reference_x[subaperture],
                reference_y[subaperture],
                flux_thresholds[subaperture],
            )
        @inbounds begin
            selected = valid && active[subaperture]
            slopes[subaperture, 1] = selected ? x_offset : 0.0f0
            slopes[subaperture, 2] = selected ? y_offset : 0.0f0
            flux[subaperture] = measured_flux[1]
            validity[subaperture] = valid
        end
    end
end

struct ShackHartmannImage{
    Measure,
    Slopes,
    Flux,
    Validity,
    Status,
    Image,
    OriginRows,
    OriginColumns,
    XCoordinates,
    YCoordinates,
    ReferenceX,
    ReferenceY,
    PixelThresholds,
    FluxThresholds,
    Active,
    Weights,
}
    measure!::Measure
    slopes::Slopes
    flux::Flux
    validity::Validity
    frame_status::Status
    image::Image
    origin_rows::OriginRows
    origin_columns::OriginColumns
    x_coordinates::XCoordinates
    y_coordinates::YCoordinates
    reference_x::ReferenceX
    reference_y::ReferenceY
    pixel_thresholds::PixelThresholds
    flux_thresholds::FluxThresholds
    active::Active
    weights::Weights
    subaperture_rows::Int
    subaperture_columns::Int
end

function _prepare_capture(
    ::CompleteTransition,
    device,
    prepared::PreparedAlgorithm{ShackHartmannImageF32,<:ShackHartmannImagePlan},
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    return _prepare_shack_hartmann_image(
        device,
        prepared,
        prepared.plan,
        nothing,
        outputs,
        inputs,
        frame_status,
        node_index,
    )
end

_capture_shack_hartmann_weights(_, ::Nothing) = nothing
_capture_shack_hartmann_weights(device, weights) =
    _capture_array(device, weights)

function _prepare_shack_hartmann_image(
    device,
    prepared,
    plan::ShackHartmannImagePlan,
    weights,
    outputs::Tuple,
    inputs::Tuple,
    frame_status,
    node_index::Int32,
)
    length(outputs) == 3 || throw(
        ArgumentError("Shack-Hartmann capture requires three outputs"),
    )
    length(inputs) == 1 || throw(
        ArgumentError("Shack-Hartmann capture requires one input"),
    )
    slopes, flux, validity = outputs
    image = only(inputs)
    for (port, array) in (
        (:image, image),
        (:slopes, slopes),
        (:flux, flux),
        (:validity, validity),
    )
        FilterGraphAlgorithms._validate_declared_array(
            FilterGraphAlgorithms.algorithm_format(prepared, port), array,
        )
    end
    _require_capture_array("Shack-Hartmann image input", image, device)
    _require_capture_array("Shack-Hartmann slope output", slopes, device)
    _require_capture_array("Shack-Hartmann flux output", flux, device)
    _require_capture_array("Shack-Hartmann validity output", validity, device)
    _require_capture_array("captured frame status", frame_status, device)
    node_index > 0 || throw(ArgumentError("captured node index must be positive"))
    (Base.mightalias(image, slopes) || Base.mightalias(image, flux) ||
        Base.mightalias(image, validity)) && throw(
        ArgumentError("captured Shack-Hartmann outputs must not alias the image"),
    )
    (Base.mightalias(slopes, flux) || Base.mightalias(slopes, validity) ||
        Base.mightalias(flux, validity)) && throw(
        ArgumentError("captured Shack-Hartmann outputs must not alias one another"),
    )
    (Base.mightalias(frame_status, image) || Base.mightalias(frame_status, slopes) ||
        Base.mightalias(frame_status, flux) || Base.mightalias(frame_status, validity)) &&
        throw(ArgumentError("captured frame status must not alias image or outputs"))

    origin_rows = _capture_array(
        device,
        [first(origin) for origin in plan.regions.origins],
    )
    origin_columns = _capture_array(
        device,
        [last(origin) for origin in plan.regions.origins],
    )
    reference_x = _capture_array(device, plan.reference_slopes[1:2:end])
    reference_y = _capture_array(device, plan.reference_slopes[2:2:end])
    backend = KernelAbstractions.get_backend(slopes)
    subaperture_rows, subaperture_columns = plan.regions.region_size
    return ShackHartmannImage(
        _shack_hartmann_kernel(device, backend),
        slopes,
        flux,
        validity,
        frame_status,
        image,
        origin_rows,
        origin_columns,
        _capture_array(device, plan.x_coordinates),
        _capture_array(device, plan.y_coordinates),
        reference_x,
        reference_y,
        _capture_array(device, plan.pixel_thresholds),
        _capture_array(device, plan.flux_thresholds),
        _capture_array(device, plan.active),
        _capture_shack_hartmann_weights(device, weights),
        subaperture_rows,
        subaperture_columns,
    )
end

preflight(::ShackHartmannImage) = nothing

function enqueue!(captured::ShackHartmannImage)
    captured.measure!(
        captured.slopes,
        captured.flux,
        captured.validity,
        captured.frame_status,
        captured.image,
        captured.origin_rows,
        captured.origin_columns,
        captured.x_coordinates,
        captured.y_coordinates,
        captured.reference_x,
        captured.reference_y,
        captured.pixel_thresholds,
        captured.flux_thresholds,
        captured.active,
        captured.weights,
        captured.subaperture_rows,
        captured.subaperture_columns;
        ndrange=(
            SHACK_HARTMANN_IMAGE_WORKGROUP_SIZE,
            length(captured.flux),
        ),
    )
    return nothing
end

complete!(::ShackHartmannImage, ::Int32) = nothing
reset!(captured::ShackHartmannImage) = captured
