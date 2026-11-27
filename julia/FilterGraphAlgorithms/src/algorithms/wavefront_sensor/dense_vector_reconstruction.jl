function _process_dense_vector_reconstruction!(
    output::AbstractVector{T},
    plan::DenseReconstructorPlan{T},
    input::AbstractVector{T},
    check_input::Val=Val(true),
) where {T<:Union{Float32,Float64}}
    Base.require_one_based_indexing(output, input)
    matrix = plan.matrix
    length(output) == size(matrix, 1) || throw(
        DimensionMismatch("dense reconstruction output has an incompatible extent"),
    )
    length(input) == size(matrix, 2) || throw(
        DimensionMismatch("dense reconstruction input has an incompatible extent"),
    )
    Base.mightalias(output, input) && throw(
        ArgumentError("dense reconstruction input and output storage must not alias"),
    )
    Base.mightalias(output, matrix) && throw(
        ArgumentError("dense reconstruction output must not alias the prepared matrix"),
    )
    return _multiply_dense_reconstructor!(output, plan, input, check_input)
end

"""
Float32 dense matrix reconstruction from one explicitly ordered input vector to
declared output coordinates. The matrix rows are output coordinates and columns
are the input vector order. It applies no slope ordering, scale, sign, or unit
conversion. It defaults to `NoCapture()`; a loaded CUDA or AMDGPU extension
may qualify captured execution for a selected device.

A direct [`process!`](@ref) call returns `DenseReconstructionNonFiniteInput`
without writing the caller output when an input value is nonfinite. It returns
`DenseReconstructionNonFiniteOutput` after an overflowing matrix product has
written the direct caller output. A Graph supplies private Node output storage,
so either failure withholds the previously published external output.
"""
@algorithm DenseVectorReconstructorF32 begin
    label = "dense-vector-reconstructor-f32"
    failure = DenseReconstructionFailure
    operation = DenseF32Operation()

    """
    Configuration for [`DenseVectorReconstructorF32`](@ref). `initial_reconstructor`
    is an optional row-major matrix whose rows are output coordinates and columns
    are the declared ordered input coordinates. `coefficient_storage` defaults
    to `"fp32"`. Set it to `"fp16"` for row-scaled Float16 coefficients with
    Float32 ports and accumulation.
    """
    @configuration DenseVectorReconstructorF32Configuration begin
        input_count::Int => graph_rebuild
        output_count::Int => graph_rebuild
        initial_reconstructor::Union{Nothing,Vector{Float32}} => construction
        input_schema::String => graph_rebuild
        output_schema::String => graph_rebuild
        reconstructor_schema::String => graph_rebuild
        (coefficient_storage::String = "fp32") => graph_rebuild
    end

    ports(configuration) = (
        input(
            :ordered_input,
            Float32,
            (configuration.input_count,),
            configuration.input_schema;
            name="ordered-input",
        ),
        output(
            :output,
            Float32,
            (configuration.output_count,),
            configuration.output_schema;
            metadata_from=:ordered_input,
        ),
        parameter(
            :reconstructor,
            Float32,
            (configuration.output_count, configuration.input_count),
            configuration.reconstructor_schema;
            replace=_replace_dense_reconstructor_storage,
        ),
    )

    prepare(configuration) = begin
        plan = DenseReconstructorPlan(
            Float32,
            configuration.output_count,
            configuration.input_count,
            configuration.initial_reconstructor,
        )
        _configure_dense_reconstructor_storage(plan, configuration.coefficient_storage)
    end

    process!(plan) = _process_dense_vector_reconstruction!(output, plan, ordered_input)
end

capture_support(::Type{DenseVectorReconstructorF32}, device) = NoCapture()

function _process_finite_input!(output, prepared::PreparedAlgorithm{DenseVectorReconstructorF32},
        input, _metadata)
    return _process_dense_vector_reconstruction!(output, prepared.plan, input, Val(false))
end
