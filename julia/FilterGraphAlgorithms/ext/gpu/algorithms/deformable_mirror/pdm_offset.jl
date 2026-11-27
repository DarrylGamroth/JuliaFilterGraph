# Each lane retains two additions in the bounded path. The larger path rescans
# after the ranked failure reduction to preserve atomic vector publication.
const PDM_OFFSET_LANES = 256
const PDM_OFFSET_TILE = 2 * PDM_OFFSET_LANES

@kernel function _pdm_offset!(
        output, current_offset, input, offset, frame_status, input_failure, output_failure)
    lane = @index(Local, Linear)
    failures = @localmem Int32 (PDM_OFFSET_LANES,)
    @uniform valid = @inbounds iszero(frame_status[1])
    failure = typemax(Int32)
    index = Int(lane)
    while valid && index <= length(input)
        @inbounds value = input[index]
        if !isfinite(value)
            failure = Int32(2index - 1)
            break
        else
            @inbounds addition = current_offset === nothing ? value + offset[index] :
                                 FilterGraphAlgorithms._pdm_offset_sum(value, offset[index])
            if !isfinite(addition)
                failure = Int32(2index)
                break
            end
        end
        index += PDM_OFFSET_LANES
    end
    @inbounds failures[lane] = failure
    @synchronize
    for stride in (128, 64, 32, 16, 8, 4, 2, 1)
        if lane <= stride
            @inbounds failures[lane] = min(failures[lane], failures[lane + stride])
        end
        @synchronize
    end
    if valid && @inbounds(failures[1] != typemax(Int32))
        if lane == 1
            @inbounds frame_status[1] = isodd(failures[1]) ? input_failure : output_failure
        end
    elseif valid
        index = Int(lane)
        while index <= length(output)
            @inbounds output[index] = current_offset === nothing ?
                                      input[index] + offset[index] :
                                      FilterGraphAlgorithms._pdm_offset_sum(input[index], offset[index])
            if current_offset !== nothing
                @inbounds current_offset[index] = offset[index]
            end
            index += PDM_OFFSET_LANES
        end
    end
end

# One block retains at most two results per lane until the complete publication
# decision. The staged kernel above remains the fallback for larger extents.
@kernel function _pdm_offset_small!(
        output, current_offset, input, offset, frame_status, input_failure, output_failure)
    lane = @index(Local, Linear)
    failures = @localmem Int32 (PDM_OFFSET_LANES,)
    @uniform valid = @inbounds iszero(frame_status[1])
    failure = typemax(Int32)
    first_addition = @private Float32 (1,)
    first_addition[1] = 0.0f0
    second_addition = @private Float32 (1,)
    second_addition[1] = 0.0f0
    if valid && lane <= length(input)
        @inbounds value = input[lane]
        if !isfinite(value)
            failure = Int32(2 * Int(lane) - 1)
        else
            @inbounds first_addition[1] = current_offset === nothing ? value + offset[lane] :
                                      FilterGraphAlgorithms._pdm_offset_sum(value, offset[lane])
            isfinite(first_addition[1]) || (failure = Int32(2 * Int(lane)))
        end
    end
    second_index = Int(lane) + PDM_OFFSET_LANES
    if valid && second_index <= length(input) && failure == typemax(Int32)
        @inbounds value = input[second_index]
        if !isfinite(value)
            failure = Int32(2second_index - 1)
        else
            @inbounds second_addition[1] = current_offset === nothing ? value + offset[second_index] :
                                       FilterGraphAlgorithms._pdm_offset_sum(value, offset[second_index])
            isfinite(second_addition[1]) || (failure = Int32(2second_index))
        end
    end
    @inbounds failures[lane] = failure
    @synchronize
    for stride in (128, 64, 32, 16, 8, 4, 2, 1)
        if lane <= stride
            @inbounds failures[lane] = min(failures[lane], failures[lane + stride])
        end
        @synchronize
    end
    if valid && @inbounds(failures[1] != typemax(Int32))
        if lane == 1
            @inbounds frame_status[1] = isodd(failures[1]) ? input_failure : output_failure
        end
    elseif valid
        second_index = Int(lane) + PDM_OFFSET_LANES
        # Every input has been read before exact in-place publication begins.
        if lane <= length(output)
            @inbounds output[lane] = first_addition[1]
            if current_offset !== nothing
                @inbounds current_offset[lane] = offset[lane]
            end
        end
        if second_index <= length(output)
            @inbounds output[second_index] = second_addition[1]
            if current_offset !== nothing
                @inbounds current_offset[second_index] = offset[second_index]
            end
        end
    end
end

import FilterGraphAlgorithms:
                              PdmSystemFlatF32, PdmSystemFlatPlan,
                              PdmSystemFlatNonFiniteInput,
                              PdmSystemFlatNonFiniteOutput, PdmEngineeringOffsetF32,
                              PdmEngineeringOffsetPlan,
                              PdmEngineeringOffsetNonFiniteInput,
                              PdmEngineeringOffsetNonFiniteOutput

struct PdmOffset{K, O, C, I, P, S, F}
    add!::K
    output::O
    current_offset::C
    input::I
    offset::P
    frame_status::S
    input_failure::Int32
    output_failure::Int32
    failures::F
end


function _prepare_pdm_offset(device, prepared, outputs, inputs, frame_status,
        node_index, offset, failures, output_format_names::Tuple)
    length(inputs) == 1 || throw(ArgumentError("PDM offset capture requires one input"))
    length(outputs) in (1, 2) ||
        throw(ArgumentError("PDM offset capture requires one or two outputs"))
    input = only(inputs)
    output = first(outputs)
    length(input) <= (typemax(Int32) >> 1) ||
        throw(ArgumentError("PDM offset extent exceeds status indexing capacity"))
    FilterGraphAlgorithms._validate_declared_array(
        FilterGraphAlgorithms.algorithm_format(prepared, :requested), input,
    )
    _require_capture_array("PDM offset input/output", input, device)
    for (format_name, array) in zip(output_format_names, outputs)
        FilterGraphAlgorithms._validate_declared_array(
            FilterGraphAlgorithms.algorithm_format(prepared, format_name), array,
        )
        _require_capture_array("PDM offset input/output", array, device)
    end
    _require_capture_array("captured frame status", frame_status, device)
    length(outputs) == 2 && Base.mightalias(outputs...) &&
        throw(ArgumentError("PDM offset outputs must not alias"))
    for array in outputs
        array === input || !Base.mightalias(array, input) ||
            throw(ArgumentError("PDM offset outputs must not partially overlap input"))
    end
    current = length(outputs) == 2 ? last(outputs) : nothing
    current !== nothing && Base.mightalias(current, input) &&
        throw(ArgumentError("current-offset output must not alias input"))
    _require_capture_workgroup(device, PDM_OFFSET_LANES;
        shared_memory=PDM_OFFSET_LANES * sizeof(Int32))
    backend = KernelAbstractions.get_backend(input)
    add! = 1 <= length(input) <= PDM_OFFSET_TILE ?
           _pdm_offset_small!(backend, PDM_OFFSET_LANES) : _pdm_offset!(backend, PDM_OFFSET_LANES)
    return PdmOffset(
        add!, output,
        current, input, _capture_array(device, offset), frame_status,
        _failure_code(node_index, failures.input),
        _failure_code(node_index, failures.output), failures)
end

function _prepare_capture(::CompleteTransition, device,
        prepared::PreparedAlgorithm{PdmSystemFlatF32, <:PdmSystemFlatPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    length(outputs) == 1 || throw(ArgumentError("system-flat capture requires one output"))
    return _prepare_pdm_offset(device, prepared, outputs, inputs, frame_status,
        node_index, prepared.plan.system_flat,
        (input = PdmSystemFlatNonFiniteInput, output = PdmSystemFlatNonFiniteOutput),
        (Symbol("requested-with-system-flat"),))
end

function _prepare_capture(::CompleteTransition, device,
        prepared::PreparedAlgorithm{PdmEngineeringOffsetF32, <:PdmEngineeringOffsetPlan},
        outputs::Tuple, inputs::Tuple, frame_status, node_index::Int32)
    length(outputs) == 2 ||
        throw(ArgumentError("engineering-offset capture requires two outputs"))
    return _prepare_pdm_offset(device, prepared, outputs, inputs, frame_status,
        node_index, prepared.plan.engineering_offset,
        (input = PdmEngineeringOffsetNonFiniteInput, output = PdmEngineeringOffsetNonFiniteOutput),
        (Symbol("requested-with-offset"), Symbol("current-offset")))
end

preflight(::PdmOffset) = nothing

function enqueue!(captured::PdmOffset)
    captured.add!(captured.output, captured.current_offset, captured.input,
        captured.offset, captured.frame_status, captured.input_failure,
        captured.output_failure; ndrange = PDM_OFFSET_LANES)
    return nothing
end

function complete!(captured::PdmOffset, status::Int32)
    status == captured.input_failure && return captured.failures.input
    status == captured.output_failure && return captured.failures.output
    return nothing
end

reset!(captured::PdmOffset) = captured

import FilterGraphAlgorithms: _supports_validation_reuse, preserves_inputs, finite_output

# Sharing this adapter does not qualify another backend for validation reuse.
# Qualified backends preserve supplied inputs in their ordered stream.

preserves_inputs(c::PdmOffset) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))

# Output finiteness retains the existing runtime validation.

finite_output(c::PdmOffset, ::Val{1}) =
    _supports_validation_reuse(KernelAbstractions.get_backend(c.frame_status))
