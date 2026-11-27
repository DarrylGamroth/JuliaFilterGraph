const _PDM_SPARSE_FEEDBACK_COLUMN_LIMIT = 64

struct SparseFeedbackProjection
    min_input::Float32
    max_input::Float32
end

_sparse_feedback_size(projection) =
    size(projection, 2) > _PDM_SPARSE_FEEDBACK_COLUMN_LIMIT && length(projection) >= 1 << 18

_prepare_sparse_feedback_projection(projection, active_execution, full_execution) = nothing

_prepare_sparse_feedback_projection(
    projection::Matrix{Float32}, ::DenseProjection, ::IdentityProjection,
) = _prepare_sparse_feedback_projection(projection)

function _prepare_sparse_feedback_projection(projection::Matrix{Float32})
    # Reserve this path for matrix sizes where avoiding GEMV's coefficient
    # traffic can pay for inspecting the input and the cancellation guard.
    _sparse_feedback_size(projection) || return nothing
    min_coefficient = Inf32
    for coefficient in projection
        bits = reinterpret(UInt32, coefficient)
        iszero(bits) && continue
        # Inspect the representation even if preparation occurs in DAZ mode.
        bits == 0x80000000 && return nothing
        (bits & 0x7fffffff) >= 0x00800000 || return nothing
        min_coefficient = min(min_coefficient, abs(coefficient))
    end
    isfinite(min_coefficient) || return nothing

    # Bound the absolute error between the two Float32 reduction orders and
    # leave headroom for every intermediate, using the complete row L1 bound.
    # The lower magnitude bound keeps every nonzero product normal.
    rounding_bound = (2.0 * size(projection, 2) + 8.0) * eps(Float32) / 2
    rounding_bound < 0.25 || return nothing
    gamma = rounding_bound / (1 - rounding_bound)
    row_bound = max(1.0, _cpu_projection_row_bound(projection))
    max_input = Float32(1e-8 / (4gamma * row_bound))
    min_input = max(floatmin(Float32), Float32(2.0 * floatmin(Float32) / min_coefficient))
    isfinite(max_input) && min_input <= max_input || return nothing
    return SparseFeedbackProjection(min_input, max_input)
end

"""
    PdmFeedbackToVdmPlan(full_to_active, pdm_to_vdm; sparse_feedback=false)

Prepare the physical-deformable-mirror (PDM) constraint-feedback projection
into full virtual-deformable-mirror (VDM) coordinates and the subsequent
full-to-active VDM projection. Matrix rows select destination coordinates and
columns select source coordinates. Both matrices are copied into dense Julia
storage and must be nonempty, compatible, and finite.

Set `sparse_feedback=true` to enable bounded sparse Float32 feedback on CPU
or in a CUDA graph with a retained cuBLAS workspace and pedantic math mode.
The sparse path requires an identity PDM-to-VDM map, a sufficiently large
dense full-to-active calibration, and ordinary floating-point mode. Feedback
outside its support, magnitude, or numerical error bounds uses dense GEMV.
The sparse reduction may round differently from dense GEMV.
"""
struct PdmFeedbackToVdmPlan{T<:Union{Float32,Float64}} <: AbstractAlgorithmPlan
    full_to_active::Matrix{T}
    pdm_to_vdm::Matrix{T}
    full_to_active_execution::CpuProjectionExecution
    pdm_to_vdm_execution::CpuProjectionExecution
    sparse_feedback::Bool
    sparse_execution::Union{Nothing,SparseFeedbackProjection}

    function PdmFeedbackToVdmPlan(
        full_to_active::AbstractMatrix{T},
        pdm_to_vdm::AbstractMatrix{T};
        sparse_feedback::Bool=false,
    ) where {T<:Union{Float32,Float64}}
        isempty(full_to_active) && throw(
            ArgumentError("full-to-active VDM projection must be nonempty"),
        )
        isempty(pdm_to_vdm) &&
            throw(ArgumentError("PDM-to-VDM projection must be nonempty"))
        size(full_to_active, 2) == size(pdm_to_vdm, 1) || throw(
            DimensionMismatch(
                "full-to-active input extent must match the full VDM extent",
            ),
        )
        all(isfinite, full_to_active) || throw(
            ArgumentError("full-to-active VDM projection must be finite"),
        )
        all(isfinite, pdm_to_vdm) ||
            throw(ArgumentError("PDM-to-VDM projection must be finite"))
        selection = Matrix(full_to_active)
        projection = Matrix(pdm_to_vdm)
        active_execution = _cpu_projection_execution(selection)
        full_execution = _cpu_projection_execution(projection)
        sparse_execution = sparse_feedback ?
            _prepare_sparse_feedback_projection(selection, active_execution, full_execution) : nothing
        return new{T}(
            selection,
            projection,
            active_execution,
            full_execution,
            sparse_feedback,
            sparse_execution,
        )
    end
end

struct SparseFeedbackWorkspace
    columns::Vector{Int}
    sum_abs::Vector{Float32}
end

"""Fixed projection scratch owned by one PDM-feedback Algorithm instance."""
struct PdmFeedbackToVdmWorkspace{T,V<:AbstractVector{T}}
    full_vdm_feedback::V
    sparse::Union{Nothing,SparseFeedbackWorkspace}
end

PdmFeedbackToVdmWorkspace{T,V}(full_vdm_feedback) where {T,V<:AbstractVector{T}} =
    PdmFeedbackToVdmWorkspace{T,V}(full_vdm_feedback, nothing)

PdmFeedbackToVdmWorkspace(full_vdm_feedback::AbstractVector{T}) where {T} =
    PdmFeedbackToVdmWorkspace(full_vdm_feedback, nothing)

function PdmFeedbackToVdmWorkspace(plan::PdmFeedbackToVdmPlan{T}) where {T}
    columns = size(plan.full_to_active, 2)
    return PdmFeedbackToVdmWorkspace{T,Vector{T}}(
        zeros(T, columns), _prepare_sparse_feedback_workspace(plan),
    )
end

_prepare_sparse_feedback_workspace(plan::PdmFeedbackToVdmPlan) = nothing

function _prepare_sparse_feedback_workspace(plan::PdmFeedbackToVdmPlan{Float32})
    # Replacement can install an eligible calibration into an opted-in
    # instance. Its workspace is fixed for the declared parameter extents.
    rows, columns = size(plan.full_to_active)
    plan.sparse_feedback && size(plan.pdm_to_vdm, 2) == columns &&
        _sparse_feedback_size(plan.full_to_active) || return nothing
    return SparseFeedbackWorkspace(
        Vector{Int}(undef, _PDM_SPARSE_FEEDBACK_COLUMN_LIMIT),
        Vector{Float32}(undef, rows),
    )
end

commit!(workspace::PdmFeedbackToVdmWorkspace) = workspace
discard!(workspace::PdmFeedbackToVdmWorkspace) = workspace

function reset!(workspace::PdmFeedbackToVdmWorkspace)
    fill!(
        workspace.full_vdm_feedback,
        zero(eltype(workspace.full_vdm_feedback)),
    )
    return workspace
end

"""Expected numerical rejection from PDM-to-VDM feedback projection."""
@enum PdmFeedbackToVdmFailure::UInt8 begin
    PdmFeedbackToVdmNonFiniteInput = 1
    PdmFeedbackToVdmNonFiniteOutput = 2
end

@doc "Physical-actuator constraint feedback contains a non-finite component." PdmFeedbackToVdmNonFiniteInput
@doc "Projection from physical-actuator feedback to VDM space produced a non-finite value." PdmFeedbackToVdmNonFiniteOutput

_try_sparse_feedback!(output, projection, full_feedback, pdm_feedback, workspace, execution) =
    false

_sparse_feedback_mightalias(scratch, output, projection, full_feedback, pdm_feedback) =
    Base.mightalias(scratch, output) || Base.mightalias(scratch, projection) ||
    Base.mightalias(scratch, full_feedback) || Base.mightalias(scratch, pdm_feedback)

function _try_sparse_feedback!(
    output::Vector{Float32},
    projection::Matrix{Float32},
    full_feedback::Vector{Float32},
    pdm_feedback::Vector{Float32},
    workspace::SparseFeedbackWorkspace,
    execution::SparseFeedbackProjection,
)
    # Snapshot the FP mode on the processing thread. Sparse FMA may round
    # differently from BLAS in a nondefault mode, even for zero feedback.
    (!get_zero_subnormals() && rounding(Float32) === RoundNearest) || return false
    columns = workspace.columns
    sum_abs = workspace.sum_abs
    length(columns) == _PDM_SPARSE_FEEDBACK_COLUMN_LIMIT &&
        length(sum_abs) == length(output) || return false
    # The original PDM input may alias the output because the full projection
    # completes before writing the output. Neither may alias sparse scratch.
    (_sparse_feedback_mightalias(columns, output, projection, full_feedback, pdm_feedback) ||
     _sparse_feedback_mightalias(sum_abs, output, projection, full_feedback, pdm_feedback) ||
     Base.mightalias(columns, sum_abs) ||
     Base.mightalias(full_feedback, projection)) && return false

    count = 0
    # The caller checked the full projection extents. Record only valid
    # indices and stop at the 65th nonzero or the first ineligible magnitude.
    @inbounds for column in eachindex(full_feedback)
        value = full_feedback[column]
        reinterpret(UInt32, value) == 0 && continue
        (count == length(columns) || !(execution.min_input <= abs(value) <= execution.max_input)) &&
            return false
        count += 1
        columns[count] = column
    end
    if count == 0
        fill!(output, 0.0f0)
        return true
    end
    if count == 1
        column = @inbounds columns[1]
        weight = @inbounds full_feedback[column]
        if weight > 0.0f0
            # Each row has only one rounded normal product or a positive
            # zero from the selected column. Adding omitted products of
            # either zero sign preserves that result in RoundNearest.
            @inbounds @simd for row in eachindex(output)
                output[row] = projection[row, column] * weight
            end
            return true
        end
    end
    fill!(output, 0.0f0)
    fill!(sum_abs, 0.0f0)
    @inbounds for index in 1:count
        column = columns[index]
        weight = full_feedback[column]
        @simd for row in eachindex(output, sum_abs)
            coefficient = projection[row, column]
            output[row] = fma(coefficient, weight, output[row])
            sum_abs[row] = fma(abs(coefficient), abs(weight), sum_abs[row])
        end
    end

    # Products from omitted +0 inputs make exact additions. A conservative
    # count for nonzero terms bounds each order, including the estimate of
    # the sum of magnitudes. The extra min-normal covers tiny partial sums.
    rounding_bound = (2 * count + 8) * (eps(Float32) / 2)
    error_scale = 4rounding_bound / (1 - rounding_bound)
    @inbounds for row in eachindex(output, sum_abs)
        error = muladd(error_scale, sum_abs[row], floatmin(Float32))
        magnitude = abs(output[row])
        # Keep the dense result's sign and normal range unambiguous, and
        # satisfy the calibrated comparison tolerance against that result.
        isfinite(magnitude) && magnitude > error + floatmin(Float32) &&
            error <= 1f-11 + 2f-4 * (magnitude - error) || return false
    end
    return true
end

function _process_pdm_feedback_to_vdm!(
    vdm_constraint_feedback::AbstractVector{T},
    workspace::PdmFeedbackToVdmWorkspace{T},
    plan::PdmFeedbackToVdmPlan{T},
    pdm_constraint_feedback::AbstractVector{T},
) where {T}
    Base.require_one_based_indexing(
        vdm_constraint_feedback,
        pdm_constraint_feedback,
    )
    full_to_active = plan.full_to_active
    pdm_to_vdm = plan.pdm_to_vdm
    full_vdm_feedback = workspace.full_vdm_feedback
    length(pdm_constraint_feedback) == size(pdm_to_vdm, 2) || throw(
        DimensionMismatch("PDM constraint feedback has an incompatible extent"),
    )
    length(full_vdm_feedback) == size(pdm_to_vdm, 1) || throw(
        DimensionMismatch("PDM-feedback Workspace has an incompatible extent"),
    )
    length(vdm_constraint_feedback) == size(full_to_active, 1) || throw(
        DimensionMismatch("VDM constraint feedback has an incompatible extent"),
    )
    Base.mightalias(full_vdm_feedback, pdm_constraint_feedback) && throw(
        ArgumentError("PDM input must not alias full-VDM scratch"),
    )
    Base.mightalias(vdm_constraint_feedback, full_vdm_feedback) && throw(
        ArgumentError("VDM output must not alias full-VDM scratch"),
    )
    Base.mightalias(vdm_constraint_feedback, full_to_active) && throw(
        ArgumentError("VDM output must not alias the full-to-active projection"),
    )
    Base.mightalias(vdm_constraint_feedback, pdm_to_vdm) && throw(
        ArgumentError("VDM output must not alias the PDM-to-VDM projection"),
    )

    for value in pdm_constraint_feedback
        isfinite(value) || return PdmFeedbackToVdmNonFiniteInput
    end
    _cpu_project!(
        full_vdm_feedback,
        pdm_to_vdm,
        pdm_constraint_feedback,
        plan.pdm_to_vdm_execution,
    )
    if plan.sparse_execution !== nothing && _try_sparse_feedback!(
        vdm_constraint_feedback,
        full_to_active,
        full_vdm_feedback,
        pdm_constraint_feedback,
        workspace.sparse,
        plan.sparse_execution,
    )
        return nothing
    end
    _cpu_project_intermediate!(
        vdm_constraint_feedback,
        full_to_active,
        full_vdm_feedback,
        plan.full_to_active_execution,
    )
    for value in vdm_constraint_feedback
        isfinite(value) || return PdmFeedbackToVdmNonFiniteOutput
    end
    return nothing
end

function _replace_full_to_active(
    plan::PdmFeedbackToVdmPlan,
    values::AbstractMatrix,
)
    return PdmFeedbackToVdmPlan(values, plan.pdm_to_vdm; sparse_feedback=plan.sparse_feedback)
end

function _replace_pdm_to_vdm(
    plan::PdmFeedbackToVdmPlan,
    values::AbstractMatrix,
)
    return PdmFeedbackToVdmPlan(plan.full_to_active, values; sparse_feedback=plan.sparse_feedback)
end

function _validate_pdm_feedback_to_vdm_dimensions(
    active_vdm_count::Int,
    full_vdm_count::Int,
    physical_actuator_count::Int,
)
    active_vdm_count > 0 ||
        throw(ArgumentError("active_vdm_count must be positive"))
    full_vdm_count > 0 || throw(ArgumentError("full_vdm_count must be positive"))
    physical_actuator_count > 0 ||
        throw(ArgumentError("physical_actuator_count must be positive"))
    max_elements = div(UInt64(typemax(UInt32)), UInt64(sizeof(Float32)))
    encoded_active_count = UInt64(active_vdm_count)
    encoded_full_count = UInt64(full_vdm_count)
    encoded_physical_count = UInt64(physical_actuator_count)
    encoded_active_count <= max_elements ||
        throw(ArgumentError("active VDM byte extent must fit UInt32"))
    encoded_full_count <= div(max_elements, encoded_active_count) || throw(
        ArgumentError("full-to-active projection byte extent must fit UInt32"),
    )
    encoded_physical_count <= div(max_elements, encoded_full_count) || throw(
        ArgumentError("PDM-to-VDM projection byte extent must fit UInt32"),
    )
    return nothing
end

"""
Project physical-actuator constraint feedback into active VDM coordinates
through the prepared PDM-to-VDM and full-to-active matrices. Matrix
coefficients determine any unit conversion between the two schemas.
"""
@algorithm PdmFeedbackToVdmF32 begin
    label = "pdm-feedback-to-vdm-f32"
    failure = PdmFeedbackToVdmFailure

    """
    Set active/full VDM vector lengths and physical-actuator extent. The matrix
    parameter ports map PDM feedback to full VDM space and then full to active
    VDM space. Set `sparse_feedback=true` to use bounded sparse Float32
    feedback on eligible CPU and captured CUDA projections.
    """
    @configuration PdmFeedbackToVdmF32Configuration begin
        active_vdm_count::Int => graph_rebuild
        full_vdm_count::Int => graph_rebuild
        physical_actuator_count::Int => graph_rebuild
        (sparse_feedback::Bool = false) => graph_rebuild
    end

    ports(configuration) = (
        input(
            :pdm_constraint_feedback,
            schema_element_type(PDM_CONSTRAINT_FEEDBACK_V1),
            (configuration.physical_actuator_count,),
            PDM_CONSTRAINT_FEEDBACK_V1;
            name="pdm-constraint-feedback",
        ),
        output(
            :vdm_constraint_feedback,
            schema_element_type(VDM_CONSTRAINT_FEEDBACK_V1),
            (configuration.active_vdm_count,),
            VDM_CONSTRAINT_FEEDBACK_V1;
            name="vdm-constraint-feedback",
            metadata_from=:pdm_constraint_feedback,
        ),
        parameter(
            :full_to_active,
            schema_element_type(FULL_TO_ACTIVE_VDM_PROJECTION_V1),
            (configuration.active_vdm_count, configuration.full_vdm_count),
            FULL_TO_ACTIVE_VDM_PROJECTION_V1;
            name="full-to-active",
            replace=(plan, values) -> _replace_full_to_active(plan, values),
        ),
        parameter(
            :pdm_to_vdm,
            schema_element_type(PDM_TO_VDM_PROJECTION_V1),
            (
                configuration.full_vdm_count,
                configuration.physical_actuator_count,
            ),
            PDM_TO_VDM_PROJECTION_V1;
            name="pdm-to-vdm",
            replace=(plan, values) -> _replace_pdm_to_vdm(plan, values),
        ),
    )

    prepare(configuration) = begin
        _validate_pdm_feedback_to_vdm_dimensions(
            configuration.active_vdm_count,
            configuration.full_vdm_count,
            configuration.physical_actuator_count,
        )
        plan = PdmFeedbackToVdmPlan(
            zeros(
                Float32,
                configuration.active_vdm_count,
                configuration.full_vdm_count,
            ),
            zeros(
                Float32,
                configuration.full_vdm_count,
                configuration.physical_actuator_count,
            );
            sparse_feedback=configuration.sparse_feedback,
        )
        (plan=plan, workspace=PdmFeedbackToVdmWorkspace(plan))
    end

    process!(plan, workspace) = begin
        _process_pdm_feedback_to_vdm!(
            vdm_constraint_feedback,
            workspace,
            plan,
            pdm_constraint_feedback,
        )
    end
end

function PdmFeedbackToVdmF32Configuration(active_vdm_count, full_vdm_count, physical_actuator_count)
    return PdmFeedbackToVdmF32Configuration(
        active_vdm_count, full_vdm_count, physical_actuator_count, false,
    )
end
