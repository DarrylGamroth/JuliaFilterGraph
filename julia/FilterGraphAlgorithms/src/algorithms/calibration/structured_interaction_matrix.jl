const _AOCInteractionMatrices = AdaptiveOpticsCalibration.InteractionMatrices

function _structured_checked_product(name::AbstractString, factors::Int...)
    value = 1
    try
        for factor in factors
            value = Base.Checked.checked_mul(value, factor)
        end
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("$name extent overflows Int"))
    end
    return value
end

function _structured_hadamard_order(corrector_coordinate_count::Int)
    target = try
        Base.Checked.checked_add(corrector_coordinate_count, 1)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("Hadamard order overflows Int"))
    end
    order = 1
    while order < target
        order = try
            Base.Checked.checked_mul(order, 2)
        catch error
            error isa OverflowError || rethrow()
            throw(ArgumentError("Hadamard order overflows Int"))
        end
    end
    return order
end

function _structured_float_type(amplitudes::AbstractVector{<:Real})
    T = float(eltype(amplitudes))
    T <: AbstractFloat ||
        throw(
            ArgumentError(
                "structured interaction-matrix preparation requires an AbstractFloat",
            ),
        )
    return T
end

"""
    ZonalInteractionMatrixPlan{T}(measurement_count, amplitudes)
    ZonalInteractionMatrixPlan(measurement_count, amplitudes)

Prepare FGA's graph-facing contract for direct estimation of one complete
zonal push-pull interaction matrix. The immutable `calibration_plan` owns the
numerical contract in AdaptiveOpticsCalibration; FGA retains the response and
output extents and graph declaration contract. The `amplitudes` property
returns an immutable tuple snapshot of the prepared calibration amplitudes.
"""
struct ZonalInteractionMatrixPlan{T<:AbstractFloat} <: AbstractAlgorithmPlan
    measurement_count::Int
    corrector_coordinate_count::Int
    response_count::Int
    calibration_plan::_AOCInteractionMatrices.ZonalPushPullPlan{T}
end

function ZonalInteractionMatrixPlan{T}(
    measurement_count::Integer,
    amplitudes::AbstractVector{<:Real},
) where {T<:AbstractFloat}
    measurements = Int(measurement_count)
    method = _AOCInteractionMatrices.ZonalPushPull(T.(amplitudes))
    calibration_plan = _AOCInteractionMatrices.prepare(
        method,
        _AOCInteractionMatrices.ResponseSpecification(measurements),
    )
    correctors = calibration_plan.corrector_coordinate_count
    response_count = _structured_checked_product(
        "zonal response",
        calibration_plan.response_row_count,
        measurements,
    )
    return ZonalInteractionMatrixPlan{T}(
        measurements,
        correctors,
        response_count,
        calibration_plan,
    )
end

function ZonalInteractionMatrixPlan(
    measurement_count::Integer,
    amplitudes::AbstractVector{<:Real},
)
    T = _structured_float_type(amplitudes)
    return ZonalInteractionMatrixPlan{T}(measurement_count, amplitudes)
end

"""
    HadamardInteractionMatrixPlan{T}(measurement_count, amplitudes)
    HadamardInteractionMatrixPlan(measurement_count, amplitudes)

Prepare FGA's graph-facing contract for complete balanced Sylvester-Hadamard
push-pull interaction-matrix estimation. The immutable `calibration_plan` owns
the numerical contract in AdaptiveOpticsCalibration; FGA retains its public
extents, Hadamard order, and graph declaration contract. The `amplitudes`
property returns an immutable tuple snapshot of the prepared calibration amplitudes.
"""
struct HadamardInteractionMatrixPlan{T<:AbstractFloat} <: AbstractAlgorithmPlan
    measurement_count::Int
    corrector_coordinate_count::Int
    response_count::Int
    order::Int
    calibration_plan::_AOCInteractionMatrices.HadamardPushPullPlan{T}
end

function HadamardInteractionMatrixPlan{T}(
    measurement_count::Integer,
    amplitudes::AbstractVector{<:Real},
) where {T<:AbstractFloat}
    measurements = Int(measurement_count)
    method = _AOCInteractionMatrices.HadamardPushPull(T.(amplitudes))
    calibration_plan = _AOCInteractionMatrices.prepare(
        method,
        _AOCInteractionMatrices.ResponseSpecification(measurements),
    )
    correctors = calibration_plan.corrector_coordinate_count
    response_count = _structured_checked_product(
        "Hadamard response",
        calibration_plan.response_row_count,
        measurements,
    )
    return HadamardInteractionMatrixPlan{T}(
        measurements,
        correctors,
        response_count,
        calibration_plan.order,
        calibration_plan,
    )
end

function HadamardInteractionMatrixPlan(
    measurement_count::Integer,
    amplitudes::AbstractVector{<:Real},
)
    T = _structured_float_type(amplitudes)
    return HadamardInteractionMatrixPlan{T}(measurement_count, amplitudes)
end

function Base.getproperty(
    plan::Union{ZonalInteractionMatrixPlan,HadamardInteractionMatrixPlan},
    name::Symbol,
)
    if name === :amplitudes
        return Tuple(getfield(plan, :calibration_plan).amplitudes)
    end
    return getfield(plan, name)
end

Base.propertynames(::ZonalInteractionMatrixPlan, ::Bool=false) =
    (:measurement_count, :corrector_coordinate_count, :response_count, :amplitudes, :calibration_plan)

Base.propertynames(::HadamardInteractionMatrixPlan, ::Bool=false) =
    (:measurement_count, :corrector_coordinate_count, :response_count, :order, :amplitudes, :calibration_plan)

"""
    StructuredInteractionMatrixWorkspace(plan)

Allocate FGA's transaction owner for one structured interaction-matrix
estimator. FGA owns the `calibration_workspace` instance; its concrete
numerical workspace type is defined by AdaptiveOpticsCalibration and protects
the caller-owned graph output from failed cycles.
"""
mutable struct StructuredInteractionMatrixWorkspace{T<:AbstractFloat,W}
    calibration_workspace::W
end

"""Expected numerical rejection from a structured interaction-matrix estimator."""
@enum StructuredInteractionMatrixFailure::UInt8 begin
    StructuredNonFiniteResponse = 1
    StructuredNonFiniteCalculation = 2
end

@doc "A complete structured response cycle contains a non-finite value."
StructuredNonFiniteResponse
@doc "Structured interaction-matrix estimation produced a non-finite value."
StructuredNonFiniteCalculation

function StructuredInteractionMatrixWorkspace(
    plan::Union{ZonalInteractionMatrixPlan{T},HadamardInteractionMatrixPlan{T}},
) where {T<:AbstractFloat}
    calibration_workspace =
        AdaptiveOpticsCalibration.allocate_workspace(plan.calibration_plan)
    return StructuredInteractionMatrixWorkspace{T,typeof(calibration_workspace)}(
        calibration_workspace,
    )
end

commit!(workspace::StructuredInteractionMatrixWorkspace) = workspace
discard!(workspace::StructuredInteractionMatrixWorkspace) = workspace

reset!(workspace::StructuredInteractionMatrixWorkspace) = workspace

@inline function _structured_result(
    status::_AOCInteractionMatrices.StructuredInteractionMatrixStatus,
)
    if status === _AOCInteractionMatrices.StructuredValid
        return nothing
    elseif status === _AOCInteractionMatrices.StructuredNonFiniteResponse
        return StructuredNonFiniteResponse
    elseif status === _AOCInteractionMatrices.StructuredNonFiniteCalculation
        return StructuredNonFiniteCalculation
    end
    throw(
        ArgumentError(
            "AdaptiveOpticsCalibration direct structured-estimation status is unsupported by the FGA integration contract: $status",
        ),
    )
end

@inline function _process_structured!(
    output::AbstractMatrix{T},
    workspace::StructuredInteractionMatrixWorkspace{T},
    plan::Union{ZonalInteractionMatrixPlan{T},HadamardInteractionMatrixPlan{T}},
    responses::AbstractMatrix{T},
) where {T<:AbstractFloat}
    status = AdaptiveOpticsCalibration.process!(
        output,
        workspace.calibration_workspace,
        plan.calibration_plan,
        responses,
    )
    return _structured_result(status)
end

"""
    process!(output, workspace, plan::ZonalInteractionMatrixPlan, responses)
    process!(output, workspace, plan::HadamardInteractionMatrixPlan, responses)

Delegate one complete structured response cycle to
AdaptiveOpticsCalibration. FGA maps the calibration typed status to
its graph failure result: a complete estimate returns `nothing`; expected
non-finite input or calculation returns
[`StructuredInteractionMatrixFailure`](@ref). Invalid shape, workspace, or
alias contracts throw. The numerical owner stages before publication, so every
rejected cycle preserves `output`.
"""
function process!(
    output::AbstractMatrix{T},
    workspace::StructuredInteractionMatrixWorkspace{T},
    plan::ZonalInteractionMatrixPlan{T},
    responses::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return _process_structured!(output, workspace, plan, responses)
end

function process!(
    output::AbstractMatrix{T},
    workspace::StructuredInteractionMatrixWorkspace{T},
    plan::HadamardInteractionMatrixPlan{T},
    responses::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return _process_structured!(output, workspace, plan, responses)
end

function _replace_zonal_estimator_amplitudes(
    plan::ZonalInteractionMatrixPlan{T},
    amplitudes::AbstractVector{T},
) where {T<:AbstractFloat}
    return ZonalInteractionMatrixPlan(plan.measurement_count, amplitudes)
end

function _replace_hadamard_estimator_amplitudes(
    plan::HadamardInteractionMatrixPlan{T},
    amplitudes::AbstractVector{T},
) where {T<:AbstractFloat}
    return HadamardInteractionMatrixPlan(plan.measurement_count, amplitudes)
end

"""Float32 estimator for one complete zonal push-pull response cycle."""
@algorithm ZonalInteractionMatrixF32 begin
    label = "zonal-interaction-matrix-f32"
    failure = StructuredInteractionMatrixFailure

    """Configuration for [`ZonalInteractionMatrixF32`](@ref)."""
    @configuration ZonalInteractionMatrixF32Configuration begin
        measurement_count::Int => graph_rebuild
        amplitudes::Vector{Float32} => graph_rebuild
        response_schema::String => graph_rebuild
        interaction_matrix_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(
            :responses,
            Float32,
            (2 * length(configuration.amplitudes), configuration.measurement_count),
            configuration.response_schema,
        ),
        output(
            :interaction_matrix,
            Float32,
            (configuration.measurement_count, length(configuration.amplitudes)),
            configuration.interaction_matrix_schema;
            name="interaction-matrix",
            metadata_from=:responses,
        ),
        parameter(
            :amplitudes,
            schema_element_type(CALIBRATION_PROBE_AMPLITUDES_V1),
            (length(configuration.amplitudes),),
            CALIBRATION_PROBE_AMPLITUDES_V1;
            replace=_replace_zonal_estimator_amplitudes,
        ),
    )

    prepare(configuration) = begin
        plan = ZonalInteractionMatrixPlan(
            configuration.measurement_count,
            configuration.amplitudes,
        )
        (plan=plan, workspace=StructuredInteractionMatrixWorkspace(plan))
    end

    process!(plan, workspace) = process!(
        interaction_matrix,
        workspace,
        plan,
        responses,
    )
end

"""Float32 estimator for one complete Hadamard push-pull response cycle."""
@algorithm HadamardInteractionMatrixF32 begin
    label = "hadamard-interaction-matrix-f32"
    failure = StructuredInteractionMatrixFailure

    """Configuration for [`HadamardInteractionMatrixF32`](@ref)."""
    @configuration HadamardInteractionMatrixF32Configuration begin
        measurement_count::Int => graph_rebuild
        amplitudes::Vector{Float32} => graph_rebuild
        response_schema::String => graph_rebuild
        interaction_matrix_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(
            :responses,
            Float32,
            (
                2 * _structured_hadamard_order(length(configuration.amplitudes)),
                configuration.measurement_count,
            ),
            configuration.response_schema,
        ),
        output(
            :interaction_matrix,
            Float32,
            (configuration.measurement_count, length(configuration.amplitudes)),
            configuration.interaction_matrix_schema;
            name="interaction-matrix",
            metadata_from=:responses,
        ),
        parameter(
            :amplitudes,
            schema_element_type(CALIBRATION_PROBE_AMPLITUDES_V1),
            (length(configuration.amplitudes),),
            CALIBRATION_PROBE_AMPLITUDES_V1;
            replace=_replace_hadamard_estimator_amplitudes,
        ),
    )

    prepare(configuration) = begin
        plan = HadamardInteractionMatrixPlan(
            configuration.measurement_count,
            configuration.amplitudes,
        )
        (plan=plan, workspace=StructuredInteractionMatrixWorkspace(plan))
    end

    process!(plan, workspace) = process!(
        interaction_matrix,
        workspace,
        plan,
        responses,
    )
end
