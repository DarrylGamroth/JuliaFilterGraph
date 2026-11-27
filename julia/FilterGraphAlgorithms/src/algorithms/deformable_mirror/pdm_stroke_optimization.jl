"""
    PdmStrokeOptimizationPlan(
        preferred_command,
        command_to_null_modes,
        null_modes_to_command,
    )

Prepare affine removal of calibrated physical-command null modes. For
requested command `c`, preferred command `b`, command-to-null-mode map `H`,
and independently prepared null-mode-to-command map `J`, processing computes
`c - J * H * (c - b)`.

`H` has shape `(null_mode_count, actuator_count)` and `J` has shape
`(actuator_count, null_mode_count)`. The arrays must be finite and nonempty and are
copied into dense Julia storage. No transpose relationship is inferred between
the independently calibrated maps.
"""
struct PdmStrokeOptimizationPlan{T<:Union{Float32,Float64}} <:
       AbstractAlgorithmPlan
    actuator_count::Int
    null_mode_count::Int
    preferred_command::Vector{T}
    command_to_null_modes::Matrix{T}
    null_modes_to_command::Matrix{T}
end

function _validate_pdm_stroke_optimization_artifacts(
    preferred_command::AbstractVector{T},
    command_to_null_modes::AbstractMatrix{T},
    null_modes_to_command::AbstractMatrix{T},
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(
        preferred_command,
        command_to_null_modes,
        null_modes_to_command,
    )
    actuator_count = length(preferred_command)
    actuator_count > 0 || throw(ArgumentError("PDM actuator extent must be nonempty"))
    null_mode_count = size(command_to_null_modes, 1)
    null_mode_count > 0 || throw(
        ArgumentError("PDM stroke optimization requires at least one null mode"),
    )
    size(command_to_null_modes, 2) == actuator_count || throw(
        DimensionMismatch("command-to-null-mode map has an invalid shape"),
    )
    size(null_modes_to_command) == (actuator_count, null_mode_count) || throw(
        DimensionMismatch("null-mode-to-command map has an invalid shape"),
    )
    all(isfinite, preferred_command) || throw(
        ArgumentError("preferred PDM command must be finite"),
    )
    all(isfinite, command_to_null_modes) || throw(
        ArgumentError("command-to-null-mode map must be finite"),
    )
    all(isfinite, null_modes_to_command) || throw(
        ArgumentError("null-mode-to-command map must be finite"),
    )
    return actuator_count, null_mode_count
end

function _prepared_pdm_stroke_optimization_plan(
    preferred_command::Vector{T},
    command_to_null_modes::Matrix{T},
    null_modes_to_command::Matrix{T},
) where {T<:Union{Float32,Float64}}
    return PdmStrokeOptimizationPlan{T}(
        length(preferred_command),
        size(command_to_null_modes, 1),
        preferred_command,
        command_to_null_modes,
        null_modes_to_command,
    )
end

function PdmStrokeOptimizationPlan(
    preferred_command::AbstractVector{T},
    command_to_null_modes::AbstractMatrix{T},
    null_modes_to_command::AbstractMatrix{T},
) where {T<:Union{Float32,Float64}}
    _validate_pdm_stroke_optimization_artifacts(
        preferred_command,
        command_to_null_modes,
        null_modes_to_command,
    )
    return _prepared_pdm_stroke_optimization_plan(
        Vector(preferred_command),
        Matrix(command_to_null_modes),
        Matrix(null_modes_to_command),
    )
end

function _pdm_stroke_row_major_matrix(
    ::Type{T},
    rows::Int,
    columns::Int,
    values::AbstractVector,
    name::AbstractString,
) where {T}
    Base.require_one_based_indexing(values)
    extent = Base.Checked.checked_mul(rows, columns)
    length(values) == extent || throw(
        DimensionMismatch("$name extent does not match its configured shape"),
    )
    matrix = Matrix{T}(undef, rows, columns)
    for row in 1:rows, column in 1:columns
        matrix[row, column] = values[(row - 1) * columns + column]
    end
    return matrix
end

function _configured_pdm_stroke_optimization_plan(
    actuator_count::Int,
    null_mode_count::Int,
    preferred_command::AbstractVector{T},
    command_to_null_modes::AbstractVector{T},
    null_modes_to_command::AbstractVector{T},
) where {T<:Union{Float32,Float64}}
    actuator_count > 0 || throw(ArgumentError("actuator_count must be positive"))
    null_mode_count > 0 || throw(
        ArgumentError("null_mode_count must be positive"),
    )
    length(preferred_command) == actuator_count || throw(
        DimensionMismatch("preferred PDM command has an invalid extent"),
    )
    command_to_null = _pdm_stroke_row_major_matrix(
        T,
        null_mode_count,
        actuator_count,
        command_to_null_modes,
        "command-to-null-mode map",
    )
    null_to_command = _pdm_stroke_row_major_matrix(
        T,
        actuator_count,
        null_mode_count,
        null_modes_to_command,
        "null-mode-to-command map",
    )
    _validate_pdm_stroke_optimization_artifacts(
        preferred_command,
        command_to_null,
        null_to_command,
    )
    return _prepared_pdm_stroke_optimization_plan(
        Vector{T}(preferred_command),
        command_to_null,
        null_to_command,
    )
end

function _replace_preferred_pdm_command(
    plan::PdmStrokeOptimizationPlan{T},
    values::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(values)
    length(values) == plan.actuator_count || throw(
        DimensionMismatch("preferred PDM command has an invalid extent"),
    )
    all(isfinite, values) || throw(
        ArgumentError("preferred PDM command must be finite"),
    )
    return _prepared_pdm_stroke_optimization_plan(
        Vector(values),
        plan.command_to_null_modes,
        plan.null_modes_to_command,
    )
end

function _replace_pdm_command_to_null_modes(
    plan::PdmStrokeOptimizationPlan{T},
    values::AbstractMatrix{T},
) where {T}
    Base.require_one_based_indexing(values)
    size(values) == (plan.null_mode_count, plan.actuator_count) || throw(
        DimensionMismatch("command-to-null-mode map has an invalid shape"),
    )
    all(isfinite, values) || throw(
        ArgumentError("command-to-null-mode map must be finite"),
    )
    return _prepared_pdm_stroke_optimization_plan(
        plan.preferred_command,
        Matrix(values),
        plan.null_modes_to_command,
    )
end

function _replace_pdm_null_modes_to_command(
    plan::PdmStrokeOptimizationPlan{T},
    values::AbstractMatrix{T},
) where {T}
    Base.require_one_based_indexing(values)
    size(values) == (plan.actuator_count, plan.null_mode_count) || throw(
        DimensionMismatch("null-mode-to-command map has an invalid shape"),
    )
    all(isfinite, values) || throw(
        ArgumentError("null-mode-to-command map must be finite"),
    )
    return _prepared_pdm_stroke_optimization_plan(
        plan.preferred_command,
        plan.command_to_null_modes,
        Matrix(values),
    )
end

"""Disposable fixed scratch for one PDM stroke-optimization Node."""
struct PdmStrokeOptimizationWorkspace{T<:Union{Float32,Float64}}
    centered_command::Vector{T}
    null_mode_coefficients::Vector{T}
end

function PdmStrokeOptimizationWorkspace(
    plan::PdmStrokeOptimizationPlan{T},
) where {T}
    return PdmStrokeOptimizationWorkspace(
        zeros(T, plan.actuator_count),
        zeros(T, plan.null_mode_count),
    )
end

commit!(workspace::PdmStrokeOptimizationWorkspace) = workspace
discard!(workspace::PdmStrokeOptimizationWorkspace) = workspace

function reset!(workspace::PdmStrokeOptimizationWorkspace)
    fill!(workspace.centered_command, zero(eltype(workspace.centered_command)))
    fill!(
        workspace.null_mode_coefficients,
        zero(eltype(workspace.null_mode_coefficients)),
    )
    return workspace
end

"""Expected numerical rejection from PDM stroke optimization."""
@enum PdmStrokeOptimizationFailure::UInt8 begin
    PdmStrokeOptimizationNonFiniteInput = 1
    PdmStrokeOptimizationNonFiniteOutput = 2
end

@doc "The requested PDM command contains a non-finite actuator value."
PdmStrokeOptimizationNonFiniteInput
@doc "PDM stroke optimization produced a non-finite intermediate or output."
PdmStrokeOptimizationNonFiniteOutput

function _validate_pdm_stroke_optimization_storage(
    optimized::AbstractVector{T},
    workspace::PdmStrokeOptimizationWorkspace{T},
    plan::PdmStrokeOptimizationPlan{T},
    requested::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(optimized, requested)
    length(requested) == plan.actuator_count || throw(
        DimensionMismatch("PDM stroke-optimization input extent is invalid"),
    )
    length(optimized) == plan.actuator_count || throw(
        DimensionMismatch("PDM stroke-optimization output extent is invalid"),
    )
    length(workspace.centered_command) == plan.actuator_count || throw(
        DimensionMismatch("PDM stroke-optimization Workspace extent is invalid"),
    )
    length(workspace.null_mode_coefficients) == plan.null_mode_count || throw(
        DimensionMismatch("PDM stroke-optimization Workspace extent is invalid"),
    )
    centered = workspace.centered_command
    coefficients = workspace.null_mode_coefficients
    Base.mightalias(optimized, centered) && throw(
        ArgumentError("optimized output must not alias Workspace scratch"),
    )
    Base.mightalias(optimized, coefficients) && throw(
        ArgumentError("optimized output must not alias Workspace scratch"),
    )
    Base.mightalias(requested, centered) && throw(
        ArgumentError("requested command must not alias Workspace scratch"),
    )
    Base.mightalias(requested, coefficients) && throw(
        ArgumentError("requested command must not alias Workspace scratch"),
    )
    Base.mightalias(optimized, plan.preferred_command) && throw(
        ArgumentError("optimized output must not alias prepared artifacts"),
    )
    Base.mightalias(optimized, plan.command_to_null_modes) && throw(
        ArgumentError("optimized output must not alias prepared artifacts"),
    )
    Base.mightalias(optimized, plan.null_modes_to_command) && throw(
        ArgumentError("optimized output must not alias prepared artifacts"),
    )
    return nothing
end

function _process_pdm_stroke_optimization!(
    optimized::AbstractVector{T},
    workspace::PdmStrokeOptimizationWorkspace{T},
    plan::PdmStrokeOptimizationPlan{T},
    requested::AbstractVector{T},
) where {T}
    _validate_pdm_stroke_optimization_storage(
        optimized,
        workspace,
        plan,
        requested,
    )
    for value in requested
        isfinite(value) || return PdmStrokeOptimizationNonFiniteInput
    end

    centered = workspace.centered_command
    @inbounds for index in eachindex(centered, requested, plan.preferred_command)
        value = requested[index] - plan.preferred_command[index]
        centered[index] = value
    end
    for value in centered
        isfinite(value) || return PdmStrokeOptimizationNonFiniteOutput
    end
    coefficients = workspace.null_mode_coefficients
    mul!(coefficients, plan.command_to_null_modes, centered)
    for value in coefficients
        isfinite(value) || return PdmStrokeOptimizationNonFiniteOutput
    end
    mul!(centered, plan.null_modes_to_command, coefficients)
    @inbounds for index in eachindex(centered, requested)
        value = requested[index] - centered[index]
        centered[index] = value
    end
    for value in centered
        isfinite(value) || return PdmStrokeOptimizationNonFiniteOutput
    end
    copyto!(optimized, centered)
    return nothing
end

"""
Float32 calibrated null-mode removal for requested physical-DM commands. The
optimized output retains the requested-command schema for subsequent physical
constraint handling and command conditioning.
"""
@algorithm PdmStrokeOptimizationF32 begin
    label = "pdm-stroke-optimization-f32"
    failure = PdmStrokeOptimizationFailure

    """Configuration for [`PdmStrokeOptimizationF32`](@ref)."""
    @configuration PdmStrokeOptimizationF32Configuration begin
        actuator_count::Int => graph_rebuild
        null_mode_count::Int => graph_rebuild
        preferred_command::Vector{Float32} => construction
        command_to_null_modes::Vector{Float32} => construction
        null_modes_to_command::Vector{Float32} => construction
    end

    ports(configuration) = (
        input(
            :requested,
            schema_element_type(REQUESTED_PDM_COMMAND_V1),
            (configuration.actuator_count,),
            REQUESTED_PDM_COMMAND_V1,
        ),
        output(
            :optimized,
            schema_element_type(REQUESTED_PDM_COMMAND_V1),
            (configuration.actuator_count,),
            REQUESTED_PDM_COMMAND_V1;
            metadata_from=:requested,
        ),
        parameter(
            :preferred_command,
            schema_element_type(PREFERRED_PDM_COMMAND_V1),
            (configuration.actuator_count,),
            PREFERRED_PDM_COMMAND_V1;
            name="preferred-command",
            replace=(plan, values) ->
                _replace_preferred_pdm_command(plan, values),
        ),
        parameter(
            :command_to_null_modes,
            schema_element_type(PDM_COMMAND_TO_NULL_MODES_V1),
            (configuration.null_mode_count, configuration.actuator_count),
            PDM_COMMAND_TO_NULL_MODES_V1;
            name="command-to-null-modes",
            replace=(plan, values) ->
                _replace_pdm_command_to_null_modes(plan, values),
        ),
        parameter(
            :null_modes_to_command,
            schema_element_type(PDM_NULL_MODES_TO_COMMAND_V1),
            (configuration.actuator_count, configuration.null_mode_count),
            PDM_NULL_MODES_TO_COMMAND_V1;
            name="null-modes-to-command",
            replace=(plan, values) ->
                _replace_pdm_null_modes_to_command(plan, values),
        ),
    )

    prepare(configuration) = begin
        plan = _configured_pdm_stroke_optimization_plan(
            configuration.actuator_count,
            configuration.null_mode_count,
            configuration.preferred_command,
            configuration.command_to_null_modes,
            configuration.null_modes_to_command,
        )
        (plan=plan, workspace=PdmStrokeOptimizationWorkspace(plan))
    end

    process!(plan, workspace) = begin
        _process_pdm_stroke_optimization!(
            optimized,
            workspace,
            plan,
            requested,
        )
    end
end
