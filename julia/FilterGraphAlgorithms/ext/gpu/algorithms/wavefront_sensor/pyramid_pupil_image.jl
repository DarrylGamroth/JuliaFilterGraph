# Pupil reconstruction uses four normalized pupil images, with the
# previous frame's gain. These ports are distinct from the modern I4Q signal.
const PYRAMID_PUPIL_LANES = 256

@inline function _pupil_mean_result!(mean_output, candidate_gain, frame_status,
        image, indices, mean_scale, sum, sumabs, rescan, input_failure, mean_failure, gain_failure)
    @inbounds begin
        # On exceptional frames, establish the CPU's first rejection in
        # pupil-major, row-then-column order. A later NaN must not replace
        # an earlier serial accumulation overflow (or vice versa).
        failure = Int32(0)
        # Near cancellation or reciprocal overflow, use the CPU order.
        # The term count bound also covers larger pupil geometries.
        cancellation = max(0.01f0, 2.0f0 * eps(Float32) * Float32(length(indices)))
        if !iszero(rescan) || abs(sum) <= cancellation * sumabs ||
                abs(sum) * mean_scale < floatmin(Float32)
            serial_sum = 0.0f0
            for entry in eachindex(indices)
                value = image[indices[entry]]
                if !isfinite(value)
                    failure = input_failure
                    break
                end
                serial_sum += value
                if !isfinite(serial_sum)
                    failure = mean_failure
                    break
                end
            end
            # A finite serial sum can survive a parallel tree overflow.
            # Its successful rescan is authoritative for this frame too.
            iszero(failure) && (sum = serial_sum)
        end
        mean = sum * mean_scale
        gain = mean > 0.0f0 ? inv(mean) : 0.0f0
        if iszero(failure)
            failure = !isfinite(mean) ? mean_failure :
                      !isfinite(gain) ? gain_failure : Int32(0)
        end
        if iszero(failure)
            mean_output[1] = mean
            candidate_gain[1] = gain
        else
            frame_status[1] = failure
        end
    end
    return nothing
end

@kernel function _pyramid_pupil_mean!(
        mean_output, candidate_gain, frame_status, image, indices, mean_scale,
        input_failure, mean_failure, gain_failure
)
    lane = @index(Local, Linear)
    sums = @localmem Float32 (PYRAMID_PUPIL_LANES,)
    magnitudes = @localmem Float32 (PYRAMID_PUPIL_LANES,)
    failures = @localmem Int32 (PYRAMID_PUPIL_LANES,)
    @uniform frame_valid = @inbounds iszero(frame_status[1])
    sum = 0.0f0
    sumabs = 0.0f0
    failure = Int32(0)
    index = Int(lane)
    while frame_valid && index <= length(indices)
        value = @inbounds image[indices[index]]
        if !isfinite(value)
            failure = Int32(1)
        else
            sum += value
            sumabs += abs(value)
            !isfinite(sum) && (failure = Int32(1))
            # If any serial prefix could overflow, at least one sample exceeds
            # this conservative bound. Rescan those extreme finite frames even
            # when the parallel tree cancels to a finite result.
            abs(value) >= floatmax(Float32) * mean_scale * 0.5f0 &&
                (failure = Int32(1))
        end
        index += PYRAMID_PUPIL_LANES
    end
    @inbounds begin
        sums[lane] = sum
        magnitudes[lane] = sumabs
        failures[lane] = failure
    end
    @synchronize
    for stride in (128, 64, 32, 16, 8, 4, 2, 1)
        if lane <= stride
            @inbounds begin
                sum = sums[lane] + sums[lane + stride]
                sums[lane] = sum
                magnitudes[lane] += magnitudes[lane + stride]
                failures[lane] = max(failures[lane], failures[lane + stride])
                !isfinite(sum) && (failures[lane] = Int32(1))
            end
        end
        @synchronize
    end
    if frame_valid && lane == 1
        @inbounds _pupil_mean_result!(mean_output, candidate_gain, frame_status,
            image, indices, mean_scale, sums[1], magnitudes[1], failures[1],
            input_failure, mean_failure, gain_failure)
    end
end

@kernel function _pyramid_pupil_pixels!(
        output, frame_status, image, selected_indices, previous_gain
)
    index = @index(Global, Linear)
    if index <= length(output) && @inbounds(iszero(frame_status[1]))
        # selected_indices follows the public (4, selected) matrix storage;
        # its selected axis retains the Plan's row-major mask ordering.
        @inbounds output[index] = previous_gain[1] * image[selected_indices[index]]
    end
end

@kernel function _reset_pyramid_pupil_gain!(previous_gain, candidate_gain)
    @inbounds begin
        previous_gain[1] = 0.0f0
        candidate_gain[1] = 0.0f0
    end
end

@kernel function _finalize_pyramid_pupil!(
        previous_gain, candidate_gain, frame_status, pixels, output_failure)
    lane = @index(Local, Linear)
    failures = @localmem Int32 (VALIDATION_WORKGROUP_SIZE,)
    @uniform valid = @inbounds iszero(frame_status[1])
    failure = Int32(0)
    index = Int(lane)
    while valid && index <= length(pixels)
        if !isfinite(@inbounds pixels[index])
            failure = Int32(1)
            break
        end
        index += VALIDATION_WORKGROUP_SIZE
    end
    @inbounds failures[lane] = failure
    @synchronize
    for stride in (128, 64, 32, 16, 8, 4, 2, 1)
        _reduce_validation!(failures, Int(lane), stride)
        @synchronize
    end
    # The preceding pixel kernel has finished every read of previous_gain.
    # Earlier mean/input/gain rejection keeps priority and suppresses adoption.
    if valid && lane == 1
        if @inbounds iszero(failures[1])
            @inbounds previous_gain[1] = candidate_gain[1]
        else
            @inbounds frame_status[1] = output_failure
        end
    end
end


import FilterGraphAlgorithms:
                              PyramidPupilImageF32,
                              PyramidPixelImagePlan,
                              PyramidPixelWorkspace,
                              PyramidPixelImageNonFiniteInput,
                              PyramidPixelImageNonFiniteMean,
                              PyramidPixelImageNonFiniteNormalizationGain,
                              PyramidPixelImageNonFiniteOutput,
                              PyramidPupilReconstructorF32,
                              DenseReconstructorPlan,
                              DenseReconstructorWorkspace,
                              DenseReconstructionNonFiniteInput,
                              DenseReconstructionNonFiniteOutput,
                              PreparedAlgorithm,
                              SupportsCapture,
                              capture_support,
                              prepare_capture,
                              preflight,
                              enqueue!,
                              complete!,
                              reset!

_require_pupil_array(name, array, device) = _require_capture_array(name, array, device)
_pupil_pixel_storage(output) = output
# Compare dense parents because Base.dataids does not propagate through the
# pupil-major CUDA wrapper (originally qualified on Julia 1.12).
_pupil_mightalias(left, right) =
    Base.mightalias(_pupil_pixel_storage(left), _pupil_pixel_storage(right))
_pupil_selected_indices(output, indices) = indices
_pupil_order_kernel(backend, input) =
    _order_pyramid_pupil_pixels!(backend, PROJECTION_WORKGROUP_SIZE)
_pupil_order_buffer(device, input) = _capture_zeros(device, Float32, length(input))
_prepare_pupil_mean(device, backend, indices) =
    _pyramid_pupil_mean!(backend, PYRAMID_PUPIL_LANES)

struct PyramidPupilImage{
    Mean, Pixels, Validate, Reset, O, M, S, I, Indices, G}
    mean!::Mean
    pixels!::Pixels
    validate!::Validate
    reset!::Reset
    output::O
    mean_output::M
    frame_status::S
    image::I
    pupil_indices::Indices
    selected_indices::Indices
    previous_gain::G
    candidate_gain::G
    mean_scale::Float32
    input_failure::Int32
    mean_failure::Int32
    gain_failure::Int32
    output_failure::Int32
end

function _prepare_capture(
        ::CompleteTransition,
        device,
        prepared::PreparedAlgorithm{
            PyramidPupilImageF32, <:PyramidPixelImagePlan, <:PyramidPixelWorkspace
        },
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32
)
    length(outputs) == 2 ||
        throw(ArgumentError("PyramidPupilImageF32 capture requires two outputs"))
    length(inputs) == 1 ||
        throw(ArgumentError("PyramidPupilImageF32 capture requires one input"))
    output, mean_output = outputs
    image = only(inputs)
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :image),
        image,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("reconstruction-pixels")),
        output,
    )
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, Symbol("mean-pupil-intensity")),
        mean_output,
    )
    _require_capture_array("pyramid detector image", image, device)
    _require_pupil_array("pyramid reconstruction pixels", output, device)
    _require_capture_array("pyramid mean intensity", mean_output, device)
    _require_capture_array("captured frame status", frame_status, device)
    (_pupil_mightalias(output, image) || _pupil_mightalias(mean_output, image) ||
     _pupil_mightalias(output, mean_output)) && throw(
        ArgumentError("captured pyramid pupil inputs and outputs must not overlap"),
    )

    regions = prepared.plan.pupil_regions
    normalization = prepared.plan.normalization
    rows, columns = regions.region_size
    selected_count = size(output, 2)
    pupil_indices = Int[]
    selected_indices = Matrix{Int}(undef, 4, selected_count)
    image_indices = LinearIndices(regions.image_size)
    for pupil in 1:4
        origin_row, origin_column = regions.origins[pupil]
        selected = 0
        for row in 1:rows, column in 1:columns

            index = image_indices[origin_row + row, origin_column + column]
            push!(pupil_indices, index)
            if normalization.mask[(row - 1) * columns + column]
                selected += 1
                selected_indices[pupil, selected] = index
            end
        end
    end
    _require_capture_workgroup(device, PYRAMID_PUPIL_LANES;
        shared_memory=PYRAMID_PUPIL_LANES * (sizeof(Float32) + sizeof(Int32)))
    backend = KernelAbstractions.get_backend(_pupil_pixel_storage(output))
    return PyramidPupilImage(
        _prepare_pupil_mean(device, backend, pupil_indices),
        _pyramid_pupil_pixels!(backend, PROJECTION_WORKGROUP_SIZE),
        _finalize_pyramid_pupil!(backend, VALIDATION_WORKGROUP_SIZE),
        _reset_pyramid_pupil_gain!(backend, 1),
        output, mean_output, frame_status, image,
        _capture_array(device, pupil_indices),
        _capture_array(device, _pupil_selected_indices(output, vec(selected_indices))),
        _capture_array(device, prepared.workspace.reciprocal_previous_frame_mean),
        _capture_zeros(device, Float32, 1), normalization.mean_scale,
        _failure_code(node_index, PyramidPixelImageNonFiniteInput),
        _failure_code(node_index, PyramidPixelImageNonFiniteMean),
        _failure_code(node_index, PyramidPixelImageNonFiniteNormalizationGain),
        _failure_code(node_index, PyramidPixelImageNonFiniteOutput)
    )
end

preflight(::PyramidPupilImage) = nothing

function enqueue!(captured::PyramidPupilImage)
    captured.mean!(
        captured.mean_output, captured.candidate_gain, captured.frame_status,
        captured.image, captured.pupil_indices, captured.mean_scale,
        captured.input_failure, captured.mean_failure, captured.gain_failure;
        ndrange = PYRAMID_PUPIL_LANES
    )
    captured.pixels!(
        _pupil_pixel_storage(captured.output), captured.frame_status, captured.image,
        captured.selected_indices, captured.previous_gain;
        ndrange = length(captured.output)
    )
    captured.validate!(
        captured.previous_gain, captured.candidate_gain, captured.frame_status,
        _pupil_pixel_storage(captured.output), captured.output_failure;
        ndrange = VALIDATION_WORKGROUP_SIZE
    )
    return nothing
end

function complete!(captured::PyramidPupilImage, status::Int32)
    status == captured.input_failure && return PyramidPixelImageNonFiniteInput
    status == captured.mean_failure && return PyramidPixelImageNonFiniteMean
    status == captured.gain_failure && return PyramidPixelImageNonFiniteNormalizationGain
    status == captured.output_failure && return PyramidPixelImageNonFiniteOutput
    return nothing
end

function reset!(captured::PyramidPupilImage)
    captured.reset!(captured.previous_gain, captured.candidate_gain; ndrange = 1)
    return captured
end

import FilterGraphAlgorithms: _supports_validation_reuse, preserves_inputs, finite_output

# Sharing this adapter does not qualify another backend for validation reuse.
# Qualified backends preserve supplied inputs in their ordered stream.

preserves_inputs(c::PyramidPupilImage) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

# Output finiteness retains the existing runtime validation.

finite_output(c::PyramidPupilImage, ::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))


@kernel function _adopt_pyramid_readout_gain!(committed_gain, candidate_gain, global_status)
    if iszero(@inbounds global_status[1])
        @inbounds committed_gain[1] = candidate_gain[1]
    end
end

function _pupil_bound(bounds::_PixelBounds, mean_scale::Float32, initial_gain::Float32)
    _supported_bound_arithmetic() || return nothing
    isfinite(initial_gain) || return nothing
    isfinite(mean_scale) && mean_scale > 0 || return nothing
    iszero(bounds.magnitude) && return _FiniteBound(0.0)
    isnothing(bounds.quantum_exponent) && return nothing
    minimum_mean = Float32(ldexp(Float64(mean_scale), bounds.quantum_exponent))
    # Do not depend on subnormal reciprocal/flush behavior. The positive mean
    # must have a normal lower bound before the finite-output proof is selected.
    isfinite(minimum_mean) && minimum_mean >= floatmin(Float32) || return nothing
    # Any positive successful mean is at least minimum_mean: sums remain on the
    # common dyadic lattice, including cancellation and different sum orders.
    # A factor-two reciprocal envelope is deliberately wider than IEEE rounding.
    reciprocal_bound = ldexp(1.0, 1 - exponent(Float64(minimum_mean)))
    gain_bound = max(reciprocal_bound, _power_bound(abs(Float64(initial_gain))))
    product = bounds.magnitude * gain_bound
    isfinite(product) || return nothing
    magnitude = _power_bound(max(product, Float64(floatmin(Float32))))
    magnitude < Float64(floatmax(Float32)) || return nothing
    return _FiniteBound(magnitude)
end

struct _BoundedPupil{K}
    finalize!::K
    bound::_FiniteBound
end

function (validation::_BoundedPupil)(
        previous_gain, candidate_gain, status, pixels, failure; ndrange)
    validation.finalize!(previous_gain, candidate_gain, status; ndrange=1)
    return nothing
end

_capture_output_bound(c::PyramidPupilImage, ::Val{1}) =
    _pupil_output_bound(c.validate!)
_pupil_output_bound(validation) = nothing
_pupil_output_bound(validation::_BoundedPupil) = validation.bound

function _prepare_bounded_validation(c::PyramidPupilImage,
        inputs::Tuple{_PixelBounds})
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status)) || return c
    # Preparation is outside capture. Include the actual retained initial gain;
    # reset is zero and subsequent successful gains obey the lattice bound.
    initial_gain = only(Array(c.previous_gain))
    bound = _pupil_bound(only(inputs), c.mean_scale, initial_gain)
    isnothing(bound) && return c
    backend = KernelAbstractions.get_backend(c.previous_gain)
    validation = _BoundedPupil(_adopt_pyramid_readout_gain!(backend, 1), bound)
    return PyramidPupilImage(
        c.mean!, c.pixels!, validation, c.reset!, c.output, c.mean_output,
        c.frame_status, c.image, c.pupil_indices, c.selected_indices,
        c.previous_gain, c.candidate_gain, c.mean_scale,
        c.input_failure, c.mean_failure, c.gain_failure, c.output_failure,
    )
end
