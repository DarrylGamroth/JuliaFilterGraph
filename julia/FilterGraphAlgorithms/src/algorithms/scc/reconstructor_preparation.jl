"""
    SccReconstructorPreparationPlan{T}(
        retained_rank,
        tikhonov_scale;
        validation_tolerance=sqrt(eps(T)),
    )
    SccReconstructorPreparationPlan(
        retained_rank,
        tikhonov_scale;
        validation_tolerance=nothing,
    )

Prepare the noise- and stroke-weighted SCC response-singular factorization.

`retained_rank` is the number of SCC response-singular controller coordinates
emitted in the online factors. `tikhonov_scale` is the nonnegative `lambda` in
`sigma / (sigma^2 + lambda^2)` and has the same units as the singular values
of the whitened, stroke-scaled response. The outer constructor selects the
floating element type from `tikhonov_scale`; use the parametric constructor
to select it explicitly.

Preparation preserves any `AbstractFloat` representation supported by its
arithmetic operations. Julia's `LinearAlgebra` supplies the native and
`Float16` factorization paths; `GenericLinearAlgebra` supplies generic
factorizations such as `BigFloat`. `validation_tolerance` bounds relative
symmetry error in the pixel-noise covariance and
candidate-mode stroke metric, and relative symmetry and idempotence error in
the LLOWFS authority projector. It is a preparation-time admission tolerance,
not a singular-value cutoff or a runtime validity limit.
"""
struct SccReconstructorPreparationPlan{T<:AbstractFloat} <:
       AbstractAlgorithmPlan
    retained_rank::Int
    tikhonov_scale::T
    validation_tolerance::T

    function SccReconstructorPreparationPlan{T}(
        retained_rank::Integer,
        tikhonov_scale::Real;
        validation_tolerance::Real=sqrt(eps(T)),
    ) where {T<:AbstractFloat}
        rank = Int(retained_rank)
        rank > 0 || throw(ArgumentError("retained_rank must be positive"))

        scale = T(tikhonov_scale)
        isfinite(scale) && scale >= zero(T) ||
            throw(
                ArgumentError(
                    "tikhonov_scale must be finite and nonnegative",
                ),
            )

        tolerance = T(validation_tolerance)
        isfinite(tolerance) && zero(T) <= tolerance < one(T) ||
            throw(
                ArgumentError(
                    "validation_tolerance must be finite and in [0, 1)",
                ),
            )

        return new{T}(rank, scale, tolerance)
    end
end

function SccReconstructorPreparationPlan(
    retained_rank::Integer,
    tikhonov_scale::Real;
    validation_tolerance::Union{Nothing,Real}=nothing,
)
    T = float(typeof(tikhonov_scale))
    T <: AbstractFloat ||
        throw(
            ArgumentError(
                "SCC reconstructor preparation requires an AbstractFloat",
            ),
        )
    tolerance = if isnothing(validation_tolerance)
        sqrt(eps(T))
    else
        validation_tolerance
    end
    return SccReconstructorPreparationPlan{T}(
        retained_rank,
        tikhonov_scale;
        validation_tolerance=tolerance,
    )
end

"""
    SccReconstructorCalibration(
        selected_pixel_response,
        candidate_mode_to_vdm,
        pixel_noise_covariance,
        candidate_mode_stroke_metric,
        llowfs_authority_projector,
    )

Bundle the calibrated matrices for SCC reconstructor preparation without
copying them.

For `p` selected pixels, `m` candidate SCC modes, and a VDM extent `v`:

- `selected_pixel_response` is `J`, with shape `p` by `m`;
- `candidate_mode_to_vdm` is `Bv`, with shape `v` by `m`, and maps candidate
  mode coefficients to canonical VDM values in micrometres of wavefront;
- `pixel_noise_covariance` is `Cn`, with shape `p` by `p`;
- `candidate_mode_stroke_metric` is `Wu`, with shape `m` by `m`; and
- `llowfs_authority_projector` is `Plo`, with shape `m` by `m`, and retains
  the candidate-mode authority assigned to SCC.

The covariance and stroke metric must be symmetric positive definite. `Plo`
must be an orthogonal projector within the plan's validation tolerance. All
matrices must be finite and use mutually coherent units. This numerical input
does not infer a unit conversion, basis identifier, or artifact provenance.
The caller must not mutate the matrices during `process!`.
"""
struct SccReconstructorCalibration{
    R<:AbstractMatrix{<:Real},
    B<:AbstractMatrix{<:Real},
    N<:AbstractMatrix{<:Real},
    W<:AbstractMatrix{<:Real},
    P<:AbstractMatrix{<:Real},
}
    selected_pixel_response::R
    candidate_mode_to_vdm::B
    pixel_noise_covariance::N
    candidate_mode_stroke_metric::W
    llowfs_authority_projector::P
end

"""
    SccReconstructorDiagnostics{T}

Report conditioning and approximation diagnostics for one SCC factorization.

`numerical_rank` uses the conventional dense-SVD threshold
`max(p, m) * eps(T) * maximum(singular_values)`. A rank-deficient response has
an infinite `response_condition_number`; selecting beyond the numerical rank
has an infinite `retained_condition_number`. Residuals and LLOWFS authority
leakage are relative Frobenius norms. `vdm_noise_amplification` is the
Frobenius norm of the retained controller-to-VDM factor after applying its
Tikhonov gains, for unit-covariance whitened pixel noise.
"""
struct SccReconstructorDiagnostics{T<:AbstractFloat}
    numerical_rank::Int
    retained_rank::Int
    response_condition_number::T
    retained_condition_number::T
    whitened_response_residual::T
    fused_reconstructor_residual::T
    vdm_noise_amplification::T
    llowfs_authority_leakage::T
    projector_symmetry_error::T
    projector_idempotence_error::T
end

function _empty_scc_reconstructor_diagnostics(
    ::Type{T},
    retained_rank::Int,
) where {T<:AbstractFloat}
    return SccReconstructorDiagnostics{T}(
        0,
        retained_rank,
        T(Inf),
        T(Inf),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
    )
end

"""
    SccReconstructorArtifacts(plan, calibration)

Allocate caller-owned numerical artifacts for one SCC preparation result.

The matrices use mathematical orientation independent of serialized storage:

- `pixel_to_controller` is the retained `r` by `p` matrix `E`;
- `controller_to_vdm` is the `v` by `r` matrix `P`;
- `fused_reconstructor` is the full-component Tikhonov `v` by `p` comparison
  oracle `K` (or the numerical-rank pseudoinverse when `lambda` is zero);
- `singular_values` and `tikhonov_gains` contain all `min(p, m)` response
  singular components; and
- `diagnostics` describes both the full oracle and retained factors.

The arrays are mutable preparation output. Artifact serialization, basis
identifiers, hashes, and activation remain caller-owned responsibilities.
"""
mutable struct SccReconstructorArtifacts{T<:AbstractFloat}
    pixel_to_controller::Matrix{T}
    controller_to_vdm::Matrix{T}
    fused_reconstructor::Matrix{T}
    singular_values::Vector{T}
    tikhonov_gains::Vector{T}
    diagnostics::SccReconstructorDiagnostics{T}
end

"""
    SccReconstructorPreparationWorkspace(plan, calibration)

Construct the zero-size workspace for SCC reconstructor preparation.

`process!` validates complete local solver products and copies them directly
into caller-owned [`SccReconstructorArtifacts`](@ref). The preparation path
may allocate `LinearAlgebra` solver storage; this workspace does not duplicate
the output artifacts and is not an operating-path or warmed zero-allocation
claim.
"""
struct SccReconstructorPreparationWorkspace{T<:AbstractFloat} end

function _scc_reconstructor_dimensions(
    calibration::SccReconstructorCalibration,
)
    pixel_count, candidate_mode_count =
        size(calibration.selected_pixel_response)
    vdm_extent, projected_mode_count =
        size(calibration.candidate_mode_to_vdm)

    pixel_count > 0 && candidate_mode_count > 0 && vdm_extent > 0 ||
        throw(
            DimensionMismatch(
                "SCC pixel, candidate-mode, and VDM extents must be nonempty",
            ),
        )
    projected_mode_count == candidate_mode_count ||
        throw(
            DimensionMismatch(
                "candidate_mode_to_vdm must have one column per candidate mode",
            ),
        )
    size(calibration.pixel_noise_covariance) ==
        (pixel_count, pixel_count) ||
        throw(
            DimensionMismatch(
                "pixel_noise_covariance must be square in selected-pixel order",
            ),
        )
    size(calibration.candidate_mode_stroke_metric) ==
        (candidate_mode_count, candidate_mode_count) ||
        throw(
            DimensionMismatch(
                "candidate_mode_stroke_metric must be square in candidate-mode order",
            ),
        )
    size(calibration.llowfs_authority_projector) ==
        (candidate_mode_count, candidate_mode_count) ||
        throw(
            DimensionMismatch(
                "llowfs_authority_projector must be square in candidate-mode order",
            ),
        )

    return (
        pixel_count=pixel_count,
        candidate_mode_count=candidate_mode_count,
        vdm_extent=vdm_extent,
        singular_count=min(pixel_count, candidate_mode_count),
    )
end

function _check_scc_retained_rank(
    plan::SccReconstructorPreparationPlan,
    dimensions,
)
    plan.retained_rank <= dimensions.singular_count ||
        throw(
            DimensionMismatch(
                "retained_rank exceeds the SCC response singular extent",
            ),
        )
    return nothing
end

function SccReconstructorArtifacts(
    plan::SccReconstructorPreparationPlan{T},
    calibration::SccReconstructorCalibration,
) where {T<:AbstractFloat}
    dimensions = _scc_reconstructor_dimensions(calibration)
    _check_scc_retained_rank(plan, dimensions)
    r = plan.retained_rank
    p = dimensions.pixel_count
    v = dimensions.vdm_extent
    q = dimensions.singular_count
    return SccReconstructorArtifacts{T}(
        zeros(T, r, p),
        zeros(T, v, r),
        zeros(T, v, p),
        zeros(T, q),
        zeros(T, q),
        _empty_scc_reconstructor_diagnostics(T, r),
    )
end

function SccReconstructorPreparationWorkspace(
    plan::SccReconstructorPreparationPlan{T},
    calibration::SccReconstructorCalibration,
) where {T<:AbstractFloat}
    dimensions = _scc_reconstructor_dimensions(calibration)
    _check_scc_retained_rank(plan, dimensions)
    return SccReconstructorPreparationWorkspace{T}()
end

function _check_scc_artifact_extents(
    artifacts::SccReconstructorArtifacts,
    plan::SccReconstructorPreparationPlan,
    dimensions,
)
    r = plan.retained_rank
    p = dimensions.pixel_count
    v = dimensions.vdm_extent
    q = dimensions.singular_count
    size(artifacts.pixel_to_controller) == (r, p) ||
        throw(
            DimensionMismatch(
                "pixel_to_controller has an incompatible extent",
            ),
        )
    size(artifacts.controller_to_vdm) == (v, r) ||
        throw(
            DimensionMismatch(
                "controller_to_vdm has an incompatible extent",
            ),
        )
    size(artifacts.fused_reconstructor) == (v, p) ||
        throw(
            DimensionMismatch(
                "fused_reconstructor has an incompatible extent",
            ),
        )
    length(artifacts.singular_values) == q ||
        throw(
            DimensionMismatch(
                "singular_values has an incompatible extent",
            ),
        )
    length(artifacts.tikhonov_gains) == q ||
        throw(
            DimensionMismatch(
                "tikhonov_gains has an incompatible extent",
            ),
        )
    return nothing
end

function _canonicalize_svd_signs!(
    left_singular_vectors::AbstractMatrix{T},
    right_singular_vectors::AbstractMatrix{T},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(
        left_singular_vectors,
        right_singular_vectors,
    )
    for component in axes(right_singular_vectors, 2)
        pivot = first(axes(right_singular_vectors, 1))
        pivot_magnitude = abs(right_singular_vectors[pivot, component])
        for candidate_mode in axes(right_singular_vectors, 1)
            magnitude =
                abs(right_singular_vectors[candidate_mode, component])
            if magnitude > pivot_magnitude
                pivot = candidate_mode
                pivot_magnitude = magnitude
            end
        end
        if signbit(right_singular_vectors[pivot, component])
            for pixel in axes(left_singular_vectors, 1)
                left_singular_vectors[pixel, component] =
                    -left_singular_vectors[pixel, component]
            end
            for candidate_mode in axes(right_singular_vectors, 1)
                right_singular_vectors[candidate_mode, component] =
                    -right_singular_vectors[candidate_mode, component]
            end
        end
    end
    return nothing
end

function _validated_scc_reconstructor_calibration(
    ::Type{T},
    calibration::SccReconstructorCalibration,
) where {T<:AbstractFloat}
    response = Matrix{T}(calibration.selected_pixel_response)
    candidate_mode_to_vdm = Matrix{T}(calibration.candidate_mode_to_vdm)
    pixel_noise_covariance = Matrix{T}(calibration.pixel_noise_covariance)
    candidate_mode_stroke_metric =
        Matrix{T}(calibration.candidate_mode_stroke_metric)
    llowfs_authority_projector =
        Matrix{T}(calibration.llowfs_authority_projector)

    for (name, matrix) in (
        ("selected_pixel_response", response),
        ("candidate_mode_to_vdm", candidate_mode_to_vdm),
        ("pixel_noise_covariance", pixel_noise_covariance),
        ("candidate_mode_stroke_metric", candidate_mode_stroke_metric),
        ("llowfs_authority_projector", llowfs_authority_projector),
    )
        all(isfinite, matrix) ||
            throw(ArgumentError("$name must contain only finite values"))
    end
    return (
        response=response,
        candidate_mode_to_vdm=candidate_mode_to_vdm,
        pixel_noise_covariance=pixel_noise_covariance,
        candidate_mode_stroke_metric=candidate_mode_stroke_metric,
        llowfs_authority_projector=llowfs_authority_projector,
    )
end

function _validated_scc_llowfs_authority_projector(
    plan::SccReconstructorPreparationPlan{T},
    llowfs_authority_projector::AbstractMatrix{T},
) where {T<:AbstractFloat}
    projector_symmetry_error = _reconstructor_relative_matrix_error(
        llowfs_authority_projector,
        transpose(llowfs_authority_projector),
    )
    projector_idempotence_error = _reconstructor_relative_matrix_error(
        llowfs_authority_projector,
        llowfs_authority_projector * llowfs_authority_projector,
    )
    projector_symmetry_error <= plan.validation_tolerance ||
        throw(
            ArgumentError(
                "llowfs_authority_projector must be symmetric within validation_tolerance",
            ),
        )
    projector_idempotence_error <= plan.validation_tolerance ||
        throw(
            ArgumentError(
                "llowfs_authority_projector must be idempotent within validation_tolerance",
            ),
        )

    projector =
        (
            llowfs_authority_projector +
            transpose(llowfs_authority_projector)
        ) / T(2)
    return (
        projector=projector,
        symmetry_error=projector_symmetry_error,
        idempotence_error=projector_idempotence_error,
    )
end

function _scc_weighted_response(
    calibration_matrices,
    projector_validation,
    weighting,
)
    authority_scaled_modes =
        projector_validation.projector * weighting.stroke_inverse_sqrt
    whitened_response =
        weighting.noise_whitener *
        calibration_matrices.response *
        authority_scaled_modes
    return (
        authority_scaled_modes=authority_scaled_modes,
        whitened_response=whitened_response,
    )
end

function _scc_reconstructor_products(
    plan::SccReconstructorPreparationPlan{T},
    calibration_matrices,
    weighting,
    weighted_response,
    factorization,
) where {T<:AbstractFloat}
    left_singular_vectors = factorization.left_singular_vectors
    singular_values = factorization.singular_values
    right_singular_vectors = factorization.right_singular_vectors
    tikhonov_gains = factorization.tikhonov_gains

    retained = 1:plan.retained_rank
    retained_left = @view left_singular_vectors[:, retained]
    retained_right = @view right_singular_vectors[:, retained]
    retained_singular_values = @view singular_values[retained]
    retained_gains = @view tikhonov_gains[retained]

    pixel_to_controller =
        Diagonal(retained_gains) *
        transpose(retained_left) *
        weighting.noise_whitener
    candidate_controller_modes =
        weighted_response.authority_scaled_modes * retained_right
    controller_to_vdm =
        calibration_matrices.candidate_mode_to_vdm * candidate_controller_modes

    full_candidate_modes =
        weighted_response.authority_scaled_modes * right_singular_vectors
    full_pixel_to_controller =
        Diagonal(tikhonov_gains) *
        transpose(left_singular_vectors) *
        weighting.noise_whitener
    fused_reconstructor =
        calibration_matrices.candidate_mode_to_vdm *
        full_candidate_modes *
        full_pixel_to_controller

    retained_response =
        retained_left *
        Diagonal(retained_singular_values) *
        transpose(retained_right)
    selected_fused_reconstructor =
        controller_to_vdm * pixel_to_controller
    return (
        pixel_to_controller=pixel_to_controller,
        controller_to_vdm=controller_to_vdm,
        fused_reconstructor=fused_reconstructor,
        retained_response=retained_response,
        selected_fused_reconstructor=selected_fused_reconstructor,
        candidate_controller_modes=candidate_controller_modes,
        retained_gains=retained_gains,
    )
end

function _scc_reconstructor_diagnostics(
    plan::SccReconstructorPreparationPlan{T},
    projector_validation,
    weighted_response,
    factorization,
    products,
) where {T<:AbstractFloat}
    projector = projector_validation.projector
    candidate_mode_count = size(projector, 1)
    candidate_mode_identity =
        Matrix{T}(I, candidate_mode_count, candidate_mode_count)
    rejected_authority =
        candidate_mode_identity - projector
    authority_leakage_numerator =
        norm(rejected_authority * products.candidate_controller_modes)
    authority_leakage_denominator =
        norm(products.candidate_controller_modes)
    llowfs_authority_leakage =
        iszero(authority_leakage_denominator) ? zero(T) :
        T(authority_leakage_numerator / authority_leakage_denominator)

    singular_values = factorization.singular_values
    largest_singular_value = factorization.largest_singular_value
    numerical_rank = factorization.numerical_rank
    response_condition_number =
        numerical_rank == length(singular_values) ?
        T(largest_singular_value / last(singular_values)) : T(Inf)
    retained_condition_number =
        plan.retained_rank <= numerical_rank ?
        T(largest_singular_value / singular_values[plan.retained_rank]) :
        T(Inf)
    vdm_noise_amplification =
        T(norm(products.controller_to_vdm * Diagonal(products.retained_gains)))
    return SccReconstructorDiagnostics{T}(
        numerical_rank,
        plan.retained_rank,
        response_condition_number,
        retained_condition_number,
        _reconstructor_relative_matrix_residual(
            weighted_response.whitened_response,
            products.retained_response,
        ),
        _reconstructor_relative_matrix_residual(
            products.fused_reconstructor,
            products.selected_fused_reconstructor,
        ),
        vdm_noise_amplification,
        llowfs_authority_leakage,
        projector_validation.symmetry_error,
        projector_validation.idempotence_error,
    )
end

function _validate_scc_reconstructor_results(
    products,
    factorization,
)
    for (name, values) in (
        ("pixel_to_controller", products.pixel_to_controller),
        ("controller_to_vdm", products.controller_to_vdm),
        ("fused_reconstructor", products.fused_reconstructor),
        ("tikhonov_gains", factorization.tikhonov_gains),
    )
        all(isfinite, values) ||
            throw(
                ArgumentError(
                    "SCC reconstructor preparation produced non-finite $name",
                ),
            )
    end
    return nothing
end

function _prepare_scc_reconstructor!(
    output::SccReconstructorArtifacts{T},
    plan::SccReconstructorPreparationPlan{T},
    calibration::SccReconstructorCalibration,
) where {T<:AbstractFloat}
    calibration_matrices =
        _validated_scc_reconstructor_calibration(T, calibration)
    projector_validation = _validated_scc_llowfs_authority_projector(
        plan,
        calibration_matrices.llowfs_authority_projector,
    )
    weighting = _reconstructor_weighting(
        calibration_matrices.pixel_noise_covariance,
        calibration_matrices.candidate_mode_stroke_metric,
        plan.validation_tolerance;
        noise_covariance_name="pixel_noise_covariance",
        stroke_metric_name="candidate_mode_stroke_metric",
    )
    weighted_response = _scc_weighted_response(
        calibration_matrices,
        projector_validation,
        weighting,
    )
    factorization = _reconstructor_weighted_svd(
        weighted_response.whitened_response,
        plan.tikhonov_scale,
        "whitened SCC response",
    )
    _canonicalize_svd_signs!(
        factorization.left_singular_vectors,
        factorization.right_singular_vectors,
    )
    products = _scc_reconstructor_products(
        plan,
        calibration_matrices,
        weighting,
        weighted_response,
        factorization,
    )
    diagnostics = _scc_reconstructor_diagnostics(
        plan,
        projector_validation,
        weighted_response,
        factorization,
        products,
    )
    _validate_scc_reconstructor_results(products, factorization)

    copyto!(output.pixel_to_controller, products.pixel_to_controller)
    copyto!(output.controller_to_vdm, products.controller_to_vdm)
    copyto!(output.fused_reconstructor, products.fused_reconstructor)
    copyto!(output.singular_values, factorization.singular_values)
    copyto!(output.tikhonov_gains, factorization.tikhonov_gains)
    output.diagnostics = diagnostics
    return output
end

"""
    process!(
        output::SccReconstructorArtifacts,
        workspace::SccReconstructorPreparationWorkspace,
        plan::SccReconstructorPreparationPlan,
        calibration::SccReconstructorCalibration,
    ) -> output

Prepare weighted SCC online factors and a full fused comparison oracle.

With `N = Cn^(-1/2)`, `S = Wu^(-1/2)`, `Q = Plo * S`,
`Bv = candidate_mode_to_vdm`, and
`N * J * Q = U * Diagonal(sigma) * V'`, the complete oracle is

```text
K = Bv * Q * V * Diagonal(gain) * U' * N
gain[i] = sigma[i] / (sigma[i]^2 + lambda^2)
```

For retained rank `r`, the online factors are

```text
E = Diagonal(gain[1:r]) * U[:, 1:r]' * N
P = Bv * Q * V[:, 1:r]
```

so selected-pixel residuals `y` produce SCC response-singular controller
values `z = E * y` and canonical VDM values `u = P * z`. With every singular
component retained, `P * E` equals `K` to solver precision. A smaller rank is
a measurable approximation recorded in `diagnostics`.

The method validates all dimensions, matrices, and complete local solver
products before copying them directly into `output`. It may allocate solver
temporaries and is a preparation-only operation unless separately qualified
for an operating path.

Each singular-vector pair is sign-canonicalized so the largest-magnitude
candidate-mode coordinate of the right singular vector is nonnegative.
Exactly repeated singular values still define a subspace rather than a unique
basis; artifact qualification must inspect the emitted singular values and
must not claim a stable coordinate identity across an unresolved repeated
subspace.
"""
function process!(
    output::SccReconstructorArtifacts{T},
    _workspace::SccReconstructorPreparationWorkspace{T},
    plan::SccReconstructorPreparationPlan{T},
    calibration::SccReconstructorCalibration,
) where {T<:AbstractFloat}
    dimensions = _scc_reconstructor_dimensions(calibration)
    _check_scc_retained_rank(plan, dimensions)
    _check_scc_artifact_extents(output, plan, dimensions)
    return _prepare_scc_reconstructor!(output, plan, calibration)
end
