# Execution metadata is prepared from each plan's owned, logically read-only
# dense matrix. Keep the matrix for dense execution and GPU capture, and keep
# the plan type independent of which CPU execution method was selected.
# Both matrix and metadata are logically read-only: parameter replacement
# rebuilds both through the plan constructor. Direct mutation is unsupported.
struct DenseProjection end
struct IdentityProjection end
struct SelectionProjection
    columns::Vector{Int}
end

const CpuProjectionExecution =
    Union{DenseProjection,IdentityProjection,SelectionProjection}

# Immutable preparation metadata only. Mutable multiplication scratch belongs
# to each Workspace, so owners may safely share one logically read-only Plan.
struct CompletionProjection{T}
    controlled_rows::Vector{Int}
    extrapolated_rows::Vector{Int}
    coefficients::Matrix{T}
    safe_input_limit::T
end

const CpuCompletionExecution{T} =
    Union{DenseProjection,IdentityProjection,SelectionProjection,CompletionProjection{T}}

function _cpu_projection_execution(projection::Matrix{T}) where {T}
    rows, columns = size(projection)
    selected_columns = Vector{Int}(undef, rows)
    identity = rows == columns
    for row in 1:rows
        selected = 0
        for column in 1:columns
            coefficient = projection[row, column]
            if coefficient == one(T)
                iszero(selected) || return DenseProjection()
                selected = column
            elseif !iszero(coefficient)
                return DenseProjection()
            end
        end
        iszero(selected) && return DenseProjection()
        selected_columns[row] = selected
        identity &= selected == row
    end
    return identity ? IdentityProjection() : SelectionProjection(selected_columns)
end

function _cpu_completion_execution(projection::Matrix, physical::Matrix)
    return _cpu_completion_execution(projection, _cpu_projection_execution(projection), physical)
end

_cpu_completion_execution(projection, execution::Union{IdentityProjection,SelectionProjection}, physical) = execution

function _cpu_completion_execution(projection::Matrix{T}, ::DenseProjection, physical) where {T}
    rows, columns = size(projection)
    rows > columns || return DenseProjection()
    # Require a smaller packed coefficient/index payload than the dense map.
    # Saved pass-through coefficient bytes must exceed the row-index bytes.
    UInt128(columns) * columns * sizeof(T) > UInt128(rows) * sizeof(Int) || return DenseProjection()
    controlled_rows = zeros(Int, columns)
    for row in 1:rows
        selected = 0
        for column in 1:columns
            coefficient = projection[row, column]
            if coefficient == one(T)
                if !iszero(selected)
                    selected = 0
                    break
                end
                selected = column
            elseif !iszero(coefficient)
                selected = 0
                break
            end
        end
        if !iszero(selected) && iszero(controlled_rows[selected])
            controlled_rows[selected] = row
        end
    end
    # Borrow the original input in its native column order: every input column
    # must have an exact unit row. Maps requiring partial input gathers remain dense.
    all(>(0), controlled_rows) || return DenseProjection()
    controlled = falses(rows)
    controlled[controlled_rows] .= true
    extrapolated_rows = findall(!, controlled)
    return CompletionProjection(controlled_rows, extrapolated_rows,
        projection[extrapolated_rows, :], _cpu_completion_input_limit(projection, physical))
end


# Bound both stages so normal-range execution cannot introduce or suppress a
# legacy overflow rejection through a different BLAS reduction order. Extreme
# inputs execute the retained dense matrices. All bound work is preparation.
function _cpu_completion_input_limit(completion::Matrix{T}, physical::Matrix{T}) where {T}
    rounding = (size(completion, 2) + size(physical, 2)) * Float64(eps(T))
    rounding < 0.25 || return zero(T)
    completion_bound = _cpu_projection_row_bound(completion)
    physical_bound = max(1.0, _cpu_projection_row_bound(physical))
    # Eightfold headroom and a margin for both stages' rounded products/sums
    # also cover Float64 row-bound rounding and conversion back to T.
    limit = (Float64(floatmax(T)) / 8 / completion_bound / physical_bound) *
        (1 - rounding)^2
    return T(limit)
end

function _cpu_projection_row_bound(matrix)
    bound = 0.0
    for row in axes(matrix, 1)
        total = 0.0
        for column in axes(matrix, 2)
            total += abs(Float64(matrix[row, column]))
        end
        bound = max(bound, total)
    end
    return bound
end

_cpu_completion_mightalias(array, ::Union{DenseProjection,IdentityProjection,SelectionProjection}) = false
_cpu_completion_mightalias(array, execution::CompletionProjection) =
    Base.mightalias(array, execution.coefficients) ||
    Base.mightalias(array, execution.controlled_rows) ||
    Base.mightalias(array, execution.extrapolated_rows)

function _cpu_complete!(output, projection, input,
    execution::Union{DenseProjection,IdentityProjection,SelectionProjection}, scratch)
    _cpu_project!(output, projection, input, execution)
    return nothing
end

function _cpu_complete!(output, projection, input, execution::CompletionProjection, scratch)
    Base.require_one_based_indexing(output, input, scratch)
    length(scratch) == length(execution.extrapolated_rows) ||
        throw(DimensionMismatch("completion scratch has an incompatible extent"))
    (Base.mightalias(scratch, input) || Base.mightalias(scratch, output) ||
     Base.mightalias(scratch, projection) || _cpu_completion_mightalias(scratch, execution) ||
     Base.mightalias(output, projection) || _cpu_completion_mightalias(output, execution)) &&
        throw(ArgumentError("completion scratch and output must not alias input or preparation metadata"))
    for value in input
        if abs(value) > execution.safe_input_limit
            mul!(output, projection, input)
            return nothing
        end
    end
    mul!(scratch, execution.coefficients, input)
    # The caller validates full/input extents and one-based indexing; cold
    # preparation partitions all output rows and assigns one row per column.
    # Plan-owned row indices and coefficients are logically read-only.
    @inbounds for column in eachindex(execution.controlled_rows)
        output[execution.controlled_rows[column]] = input[column]
    end
    @inbounds for row in eachindex(execution.extrapolated_rows)
        output[execution.extrapolated_rows[row]] = scratch[row]
    end
    return nothing
end

# Callers validate dimensions, indexing, aliasing, and finite external input.
# Identity and selection copy the selected finite value, including its zero
# sign and subnormal representation; dense summation can change a zero sign.
function _cpu_project!(output, projection, input, ::DenseProjection)
    mul!(output, projection, input)
    return nothing
end

function _cpu_project!(output, projection, input, ::IdentityProjection)
    copyto!(output, input)
    return nothing
end

function _cpu_project!(output, projection, input, execution::SelectionProjection)
    for row in eachindex(execution.columns)
        output[row] = input[execution.columns[row]]
    end
    return nothing
end

function _cpu_project_intermediate!(output, projection, input, ::DenseProjection)
    mul!(output, projection, input)
    return nothing
end

function _cpu_project_intermediate!(
    output,
    projection,
    input,
    execution::Union{IdentityProjection,SelectionProjection},
)
    # Dense multiplication propagates nonfinite scratch even through zero
    # coefficients. A selection must not hide overflow in an unselected input.
    if all(isfinite, input)
        _cpu_project!(output, projection, input, execution)
    else
        mul!(output, projection, input)
    end
    return nothing
end
