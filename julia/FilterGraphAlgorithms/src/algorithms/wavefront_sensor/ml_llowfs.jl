"""Expected rejection from one ML LLOWFS inference."""
@enum MlLlowfsFailure::UInt8 begin
    MlLlowfsParametersMissing = 1
    MlLlowfsNonFiniteInput = 2
    MlLlowfsInvalidFlux = 3
    MlLlowfsInvalidStandardDeviation = 4
    MlLlowfsNonFiniteOutput = 5
end

@doc "No model parameters have been adopted for the ML LLOWFS Node."
MlLlowfsParametersMissing
@doc "The ML LLOWFS input region contains a non-finite pixel."
MlLlowfsNonFiniteInput
@doc "The ML LLOWFS input has nonpositive or non-finite total flux."
MlLlowfsInvalidFlux
@doc "The normalized ML LLOWFS input has invalid sample deviation."
MlLlowfsInvalidStandardDeviation
@doc "The ML LLOWFS model produced a non-finite modal coefficient."
MlLlowfsNonFiniteOutput

struct MlLlowfsPlan{M<:MlLlowfsModel} <: AbstractAlgorithmPlan
    model::M
    parameters::Vector{Float32}
    ready::Bool
end

function _replace_ml_llowfs_parameters(plan::MlLlowfsPlan, parameters::AbstractVector{Float32})
    Base.require_one_based_indexing(parameters)
    length(parameters) == plan.model.parameter_count ||
        throw(DimensionMismatch("ML LLOWFS parameter vector has the wrong length"))
    all(isfinite, parameters) ||
        throw(ArgumentError("ML LLOWFS parameters must be finite"))
    return MlLlowfsPlan(plan.model, collect(parameters), true)
end

struct MlLlowfsWorkspace
    input::Array{Float32,3}
end

MlLlowfsWorkspace(model::MlLlowfsModel) =
    MlLlowfsWorkspace(zeros(Float32, model.image_size..., 1))

commit!(workspace::MlLlowfsWorkspace) = workspace
discard!(workspace::MlLlowfsWorkspace) = workspace

function reset!(workspace::MlLlowfsWorkspace)
    fill!(workspace.input, 0.0f0)
    return workspace
end

function _check_ml_llowfs_storage(
    modes::AbstractVector{Float32},
    workspace::MlLlowfsWorkspace,
    plan::MlLlowfsPlan,
    region_pixels::AbstractArray{Float32,3},
)
    Base.require_one_based_indexing(modes, region_pixels)
    model = plan.model
    size(region_pixels) == (1, model.image_size...) ||
        throw(DimensionMismatch("ML LLOWFS input must match the model's region size"))
    length(modes) == model.mode_count ||
        throw(DimensionMismatch("ML LLOWFS output must match the model's mode count"))
    size(workspace.input) == (model.image_size..., 1) ||
        throw(DimensionMismatch("ML LLOWFS Workspace has the wrong shape"))
    length(plan.parameters) == model.parameter_count ||
        throw(DimensionMismatch("ML LLOWFS Plan has the wrong parameter extent"))
    (Base.mightalias(modes, region_pixels) || Base.mightalias(modes, workspace.input) ||
        Base.mightalias(modes, plan.parameters)) &&
        throw(ArgumentError("ML LLOWFS output must not alias inference storage"))
    Base.mightalias(workspace.input, region_pixels) &&
        throw(ArgumentError("ML LLOWFS Workspace must not alias its input"))
    return nothing
end

# Use the same reduction order on every backend. The signed, calibrated ROI can
# contain cancelling samples. Accumulate in Float64 and round to the model's
# Float32 format at normalization boundaries.
@inline function _normalize_llowfs!(input, pixels)
    total = correction = 0.0
    @inbounds for column in axes(input, 2), row in axes(input, 1)
        value = pixels[1, row, column]
        isfinite(value) || return MlLlowfsNonFiniteInput
        input[row, column, 1] = value
        total, correction = _compensated_add(total, correction, Float64(value))
    end
    flux = Float32(total + correction)
    isfinite(flux) && flux > 0f0 || return MlLlowfsInvalidFlux

    n = length(input)
    total = correction = 0.0
    @inbounds for i in eachindex(input)
        value = input[i] / flux
        input[i] = value
        total, correction = _compensated_add(total, correction, Float64(value))
    end
    μ = (total + correction) / n
    squares = 0.0
    @inbounds for i in eachindex(input)
        delta = Float64(input[i]) - μ
        squares = fma(delta, delta, squares)
        input[i] = Float32(delta)
    end
    σ = Float32(sqrt(squares / (n - 1)))
    isfinite(σ) && σ > 0f0 || return MlLlowfsInvalidStandardDeviation
    @inbounds for i in eachindex(input)
        input[i] /= σ
    end
    return nothing
end

"""
    normalize_llowfs!(input, region_pixels)

Write the inference preprocessing for one calibrated detector region from
`(1, rows, columns)` to `(rows, columns, 1)`. Use the same preprocessing before
training. The normalization uses unit flux, zero mean, and corrected sample
standard deviation. Return `nothing` on success, or a [`MlLlowfsFailure`](@ref)
for invalid input or normalization. The buffers must be disjoint.
"""
function normalize_llowfs!(input::AbstractArray{Float32,3},
        region_pixels::AbstractArray{Float32,3})
    Base.require_one_based_indexing(input, region_pixels)
    rows, columns, channels = size(input)
    channels == 1 && length(input) > 1 && size(region_pixels) == (1, rows, columns) ||
        throw(DimensionMismatch("expected one LLOWFS region and a matching model input"))
    Base.mightalias(input, region_pixels) &&
        throw(ArgumentError("LLOWFS preprocessing buffers must be disjoint"))
    return _normalize_llowfs!(input, region_pixels)
end

function _process_ml_llowfs!(
    modes::AbstractVector{Float32},
    workspace::MlLlowfsWorkspace,
    plan::MlLlowfsPlan,
    region_pixels::AbstractArray{Float32,3},
)
    _check_ml_llowfs_storage(modes, workspace, plan, region_pixels)
    plan.ready || return MlLlowfsParametersMissing

    failure = _normalize_llowfs!(workspace.input, region_pixels)
    isnothing(failure) || return failure

    inferred_modes = plan.model.chain(workspace.input, plan.parameters)
    for mode in inferred_modes
        isfinite(mode) || return MlLlowfsNonFiniteOutput
    end
    copyto!(modes, inferred_modes)
    return nothing
end

function _prepare_ml_llowfs(model::MlLlowfsModel)
    plan = MlLlowfsPlan(model, zeros(Float32, model.parameter_count), false)
    workspace = MlLlowfsWorkspace(model)
    inferred_modes = model.chain(workspace.input, plan.parameters)
    size(inferred_modes) == (model.mode_count,) ||
        error("ML LLOWFS model produced the wrong output extent during warmup")
    return (plan=plan, workspace=workspace)
end

"""
Infer a vector of low-order wavefront sensor modes from one detector region.
The model fixes the image and output extents during preparation. Supply model
parameters in the model's SimpleChains order through the `model-parameters`
Parameter Port. The caller declares the mode basis and parameter layout schemas.

The CPU and capture adapters use the same input normalization as
[`normalize_llowfs!`](@ref). GPU capture binds the model parameters, intermediate
tensors, and output on the selected device.
"""
@algorithm MlLlowfsF32 begin
    label = "ml-llowfs-f32"
    failure = MlLlowfsFailure

    """Construction configuration for [`MlLlowfsF32`](@ref)."""
    @configuration MlLlowfsF32Configuration begin
        model::MlLlowfsModel => graph_rebuild
        modes_schema::String => graph_rebuild
        parameters_schema::String => graph_rebuild
    end

    ports(configuration) = (
        input(
            :region_pixels,
            schema_element_type(REGION_PIXELS_V1),
            (1, configuration.model.image_size...),
            REGION_PIXELS_V1;
            name="region-pixels",
        ),
        output(
            :modes,
            Float32,
            (configuration.model.mode_count,),
            configuration.modes_schema;
            metadata_from=:region_pixels,
        ),
        parameter(
            :model_parameters,
            Float32,
            (configuration.model.parameter_count,),
            configuration.parameters_schema;
            name="model-parameters",
            replace=(plan, values) ->
                _replace_ml_llowfs_parameters(plan, values),
        ),
    )

    prepare(configuration) = _prepare_ml_llowfs(configuration.model)

    process!(plan, workspace) = begin
        _process_ml_llowfs!(modes, workspace, plan, region_pixels)
    end
end
