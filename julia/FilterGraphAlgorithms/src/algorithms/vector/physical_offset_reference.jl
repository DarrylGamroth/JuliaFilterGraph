"""
Read-only adopted physical offset and its matched PWFS reference correction.
Prepare through [`PhysicalOffsetReferenceF32`](@ref). `physical_to_modes` and
`modes_to_reference` retain the calibrated factors, `reference_sign` retains the
explicit sign, and `offset` and `correction` belong to the same snapshot.
Cold admission may inspect these fields to verify calibration identity. Never
mutate their arrays; use declared parameter replacement between frame calls.
"""
struct PhysicalOffsetReferencePlan <: AbstractAlgorithmPlan
    combination::VectorLinearCombinationPlan
    physical_to_modes::Matrix{Float32}
    modes_to_reference::Matrix{Float32}
    offset::Vector{Float32}
    correction::Vector{Float32}
    reference_sign::Float32
end

function _physical_offset_reference_plan(combination, q, c, offset, reference_sign)
    Base.require_one_based_indexing(q, c, offset)
    size(q, 2) == length(offset) && size(c, 2) == size(q, 1) &&
        size(c, 1) == combination.extent ||
        throw(DimensionMismatch("physical offset reference calibration dimensions differ"))
    all(isfinite, q) && all(isfinite, c) && all(isfinite, offset) ||
        throw(ArgumentError("physical offset reference calibration and offset must be finite"))
    # Cold replacement computes only the two calibrated factors; never form C*Q.
    modes = q * offset
    all(isfinite, modes) || throw(ArgumentError("physical offset modal coefficients overflow"))
    correction = c * modes
    correction .*= reference_sign
    all(isfinite, correction) || throw(ArgumentError("physical offset reference correction overflows"))
    return PhysicalOffsetReferencePlan(combination, q, c, offset, correction, reference_sign)
end

function _replace_physical_offset_reference(plan, ::Val{:offset}, values)
    return _physical_offset_reference_plan(plan.combination, plan.physical_to_modes,
        plan.modes_to_reference, copy(values), plan.reference_sign)
end
function _replace_physical_offset_reference(plan, ::Val{:physical_to_modes}, values)
    q = DenseReconstructorPlan(values).matrix
    return _physical_offset_reference_plan(plan.combination, q,
        plan.modes_to_reference, plan.offset, plan.reference_sign)
end
function _replace_physical_offset_reference(plan, ::Val{:modes_to_reference}, values)
    c = DenseReconstructorPlan(values).matrix
    return _physical_offset_reference_plan(plan.combination, plan.physical_to_modes,
        c, plan.offset, plan.reference_sign)
end

function _physical_reference_output_alias(output, residual, plan, workspace)
    return Base.mightalias(output, residual) || Base.mightalias(output, plan.offset) ||
        Base.mightalias(output, plan.correction) || Base.mightalias(output, plan.physical_to_modes) ||
        Base.mightalias(output, plan.modes_to_reference) || Base.mightalias(output, workspace.candidate_output)
end

function _process_physical_offset_reference!(compensated, physical, workspace, plan, residual)
    Base.require_one_based_indexing(compensated, physical, residual)
    length(physical) == length(plan.offset) ||
        throw(DimensionMismatch("physical offset output extent differs from calibration"))
    # Reject output aliases before either component can publish. Scientific Plan
    # data and disposable scratch remain private to this prepared owner.
    Base.mightalias(compensated, physical) &&
        throw(ArgumentError("physical offset reference outputs must not alias"))
    (_physical_reference_output_alias(compensated, residual, plan, workspace) ||
     _physical_reference_output_alias(physical, residual, plan, workspace)) &&
        throw(ArgumentError("physical offset reference output aliases input, calibration or scratch"))
    status = process!(compensated, workspace, plan.combination, residual, plan.correction)
    status === nothing && copyto!(physical, plan.offset)
    return status
end

"""
Compensate a reconstructed residual for one adopted physical mirror offset.
With calibrated factors `Q = physical_to_modes` and `C = modes_to_reference`,
compute `compensated_residual = reconstructed_residual + reference_sign*C*(Q*offset)`.
The explicit sign is +1 or -1, determined by calibration. No mirror projection,
unit conversion or coordinate mapping is inferred. Factor rows/columns must
respect the declared residual, physical-actuator and common modal orders.

Cold parameter replacement validates and copies each factor or offset, computes
its finite correction, then adopts one Plan containing both correction and exact
physical offset. Replace parameters only between processing calls. Each success
publishes both outputs with residual metadata; rejection preserves both outputs.
Reset clears scratch and retains calibration and offset.
"""
@algorithm PhysicalOffsetReferenceF32 begin
    label = "physical-offset-reference-f32"
    failure = VectorLinearCombinationFailure

    """
    Set positive residual, physical-actuator and modal counts, their explicit
    schemas, and calibrated `reference_sign` (+1 or -1). Factors initially zero;
    bind both calibration parameters before operation. Matrix parameter shape
    declares its row/column order; schemas identify units, basis and calibration.
    """
    @configuration PhysicalOffsetReferenceF32Configuration begin
        residual_count::Int => graph_rebuild
        physical_count::Int => graph_rebuild
        mode_count::Int => graph_rebuild
        reference_sign::Float32 => construction
        residual_schema::String => graph_rebuild
        physical_schema::String => graph_rebuild
        physical_to_modes_schema::String => graph_rebuild
        modes_to_reference_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(:reconstructed_residual, Float32, (configuration.residual_count,),
            configuration.residual_schema; name="reconstructed-residual"),
        output(:compensated_residual, Float32, (configuration.residual_count,),
            configuration.residual_schema; name="compensated-residual", metadata_from=:reconstructed_residual),
        output(:physical_offset, Float32, (configuration.physical_count,),
            configuration.physical_schema; name="physical-offset", metadata_from=:reconstructed_residual),
        parameter(:offset, Float32, (configuration.physical_count,), configuration.physical_schema;
            replace=(plan, values) -> _replace_physical_offset_reference(plan, Val(:offset), values)),
        parameter(:physical_to_modes, Float32, (configuration.mode_count, configuration.physical_count),
            configuration.physical_to_modes_schema; name="physical-to-modes",
            replace=(plan, values) -> _replace_physical_offset_reference(plan, Val(:physical_to_modes), values)),
        parameter(:modes_to_reference, Float32, (configuration.residual_count, configuration.mode_count),
            configuration.modes_to_reference_schema; name="modes-to-reference",
            replace=(plan, values) -> _replace_physical_offset_reference(plan, Val(:modes_to_reference), values)),
    )

    prepare(configuration) = begin
        configuration.reference_sign in (-1f0, 1f0) ||
            throw(ArgumentError("physical offset reference sign must be +1 or -1"))
        combination = VectorLinearCombinationPlan(configuration.residual_count, 1f0, 1f0)
        _validate_dense_reconstructor_dimensions(configuration.mode_count, configuration.physical_count, Float32)
        _validate_dense_reconstructor_dimensions(configuration.residual_count, configuration.mode_count, Float32)
        plan = _physical_offset_reference_plan(combination,
            zeros(Float32, configuration.mode_count, configuration.physical_count),
            zeros(Float32, configuration.residual_count, configuration.mode_count),
            zeros(Float32, configuration.physical_count), configuration.reference_sign)
        (plan=plan, workspace=VectorLinearCombinationWorkspace(combination))
    end

    process!(plan, workspace) = _process_physical_offset_reference!(
        compensated_residual, physical_offset, workspace, plan, reconstructed_residual)
end
