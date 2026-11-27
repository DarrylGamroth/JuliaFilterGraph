# This file is generated from schemas/catalog.toml in JuliaFilterGraph.
# Run `julia scripts/generate_schemas.jl` with the consumer root options to regenerate.

"""
F32 dimensionless normalized temporal biquad coefficients in (coefficient, section,
coefficient-set) order. Coefficient order is (b0, b1, b2, a1, a2), with a0 = 1 and
denominator 1 + a1*z^-1 + a2*z^-2. Sections execute in ascending order. A single coefficient
set is shared across modes; otherwise sets follow input-mode order. Every coefficient is
finite and each section has poles strictly inside the unit circle.
"""
const BIQUAD_COEFFICIENTS_V1 = "pipewireao/biquad-coefficients-f32/1"

"""F32 reconstructed residual error in controller-coordinate order."""
const CONTROLLER_RESIDUAL_ERROR_V1 = "pipewireao/controller-residual-error-f32/1"

"""F32 corrector-to-measurement map for pseudo-open-loop reconstruction."""
const PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1 =
    "pipewireao/pseudo-open-loop-interaction-matrix-f32/1"

"""F32 measurement-to-residual-error pseudo-open-loop reconstructor."""
const PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1 = "pipewireao/pseudo-open-loop-reconstructor-f32/1"

"""F32 corrector-to-residual-error map for pseudo-open-loop reconstruction."""
const PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1 =
    "pipewireao/pseudo-open-loop-corrector-to-error-f32/1"

"""F32 correction command in controller-coordinate order."""
const CONTROLLER_COMMAND_V1 = "pipewireao/controller-command-f32/1"

"""F32 committed closed-loop state in controller-coordinate order."""
const CONTROLLER_STATE_V1 = "pipewireao/controller-state-f32/1"

"""F32 constraint feedback in concatenated controller-coordinate order."""
const CONTROLLER_CONSTRAINT_FEEDBACK_V1 = "pipewireao/controller-constraint-feedback-f32/1"

"""F32 commands in active virtual-deformable-mirror coordinate order."""
const VDM_COMMAND_V1 = "pipewireao/vdm-command-f32/1"

"""F32 controller-command to active-VDM projection."""
const CONTROLLER_TO_VDM_PROJECTION_V1 = "pipewireao/controller-to-vdm-projection-f32/1"

"""F32 active-VDM to controller-coordinate feedback projection."""
const VDM_TO_CONTROLLER_PROJECTION_V1 = "pipewireao/vdm-to-controller-projection-f32/1"

"""F32 requested command in concatenated physical-actuator order."""
const REQUESTED_PDM_COMMAND_V1 = "pipewireao/requested-pdm-command-f32/1"

"""F32 physical-DM system flat in concatenated actuator order."""
const PDM_SYSTEM_FLAT_V1 = "pipewireao/pdm-system-flat-f32/1"

"""F32 physical-DM engineering offset in concatenated actuator order, in µm wavefront."""
const PDM_ENGINEERING_OFFSET_V1 = "pipewireao/dm-offset-opd-um-f32/1"

"""F32 maximum differences for the configured physical-actuator edges."""
const INTER_ACTUATOR_STROKE_LIMIT_V1 = "pipewireao/inter-actuator-stroke-limit-f32/1"

"""F32 preferred physical-DM command in actuator order."""
const PREFERRED_PDM_COMMAND_V1 = "pipewireao/preferred-pdm-command-f32/1"

"""F32 map from physical-DM commands to calibrated null modes."""
const PDM_COMMAND_TO_NULL_MODES_V1 = "pipewireao/pdm-command-to-null-modes-f32/1"

"""F32 map from calibrated null modes to physical-DM commands."""
const PDM_NULL_MODES_TO_COMMAND_V1 = "pipewireao/pdm-null-modes-to-command-f32/1"

"""
F32 commanded physical-DM values in concatenated actuator order, in µm wavefront optical
path difference.
"""
const DEMANDED_PDM_COMMAND_V1 = "pipewireao/dm-command-opd-um-f32/1"

"""
F32 native device command values in physical-actuator order. The values have no displacement
unit.
"""
const NATIVE_PDM_COMMAND_V1 = "pipewireao/demanded-pdm-command-f32/1"

"""F32 requested-minus-commanded PDM feedback in actuator order."""
const PDM_CONSTRAINT_FEEDBACK_V1 = "pipewireao/pdm-constraint-feedback-f32/1"

"""F32 constraint feedback in active VDM coordinate order."""
const VDM_CONSTRAINT_FEEDBACK_V1 = "pipewireao/vdm-constraint-feedback-f32/1"

"""F32 active-to-full VDM projection."""
const ACTIVE_TO_FULL_VDM_PROJECTION_V1 = "pipewireao/active-to-full-vdm-projection-f32/1"

"""F32 full-to-active VDM projection."""
const FULL_TO_ACTIVE_VDM_PROJECTION_V1 = "pipewireao/full-to-active-vdm-projection-f32/1"

"""F32 full-VDM-to-physical-actuator projection."""
const VDM_TO_PDM_PROJECTION_V1 = "pipewireao/vdm-to-pdm-projection-f32/1"

"""F32 physical-actuator to full-VDM feedback projection."""
const PDM_TO_VDM_PROJECTION_V1 = "pipewireao/pdm-to-vdm-projection-f32/1"

"""F32 controller-state to hidden-mode projection."""
const SHAPE_TO_HIDDEN_PROJECTION_V1 = "pipewireao/shape-to-hidden-projection-f32/1"

"""F32 hidden-mode to controller-state projection."""
const HIDDEN_TO_SHAPE_PROJECTION_V1 = "pipewireao/hidden-to-shape-projection-f32/1"

"""F32 deterministic corrector excitation in corrector-coordinate order."""
const DOCRIME_EXCITATION_V1 = "pipewireao/docrime-excitation-f32/1"

"""F32 positive DO-CRIME amplitude in corrector-coordinate order."""
const DOCRIME_EXCITATION_AMPLITUDE_V1 = "pipewireao/docrime-excitation-amplitude-f32/1"

"""U64 number of samples committed into published DO-CRIME covariance sums."""
const DOCRIME_COVARIANCE_SAMPLE_COUNT_V1 =
    "pipewireao/docrime-covariance-sample-count-u64/1"

"""Boolean marker for a completed finite DO-CRIME covariance window."""
const DOCRIME_COVARIANCE_COMPLETE_V1 = "pipewireao/docrime-covariance-complete-bool/1"

"""U8 classification of one DO-CRIME interaction-matrix solve."""
const DOCRIME_INTERACTION_MATRIX_STATUS_V1 =
    "pipewireao/docrime-interaction-matrix-status-u8/1"

"""Zero-based U64 corrector-coordinate pair associated with a DO-CRIME status."""
const DOCRIME_INTERACTION_MATRIX_STATUS_COORDINATES_V1 =
    "pipewireao/docrime-interaction-matrix-status-coordinates-u64/1"

"""F32 minimum and maximum corrector sample variances."""
const DOCRIME_CORRECTOR_VARIANCE_RANGE_V1 =
    "pipewireao/docrime-corrector-variance-range-f32/1"

"""F32 maximum absolute off-diagonal corrector correlation."""
const DOCRIME_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_V1 =
    "pipewireao/docrime-maximum-absolute-corrector-correlation-f32/1"

"""F32 minimum Cholesky factorization pivot ratio."""
const DOCRIME_MINIMUM_FACTORIZATION_PIVOT_RATIO_V1 =
    "pipewireao/docrime-minimum-factorization-pivot-ratio-f32/1"

"""
U8 interaction-matrix stability classification: valid `0`, insufficient column norm `1`,
excessive relative column difference `2`, or excessive relative Frobenius difference `3`.
"""
const INTERACTION_MATRIX_STABILITY_STATUS_V1 =
    "pipewireao/interaction-matrix-stability-status-u8/1"

"""
Zero-based U64 corrector coordinate. A status coordinate is zero when the classification
does not identify one coordinate.
"""
const CORRECTOR_COORDINATE_V1 = "pipewireao/corrector-coordinate-u64/1"

"""F32 minimum column norms in first-window, second-window order."""
const INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1 =
    "pipewireao/interaction-matrix-stability-column-norms-f32/1"

"""F32 maximum symmetric relative interaction-matrix column difference."""
const INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1 =
    "pipewireao/interaction-matrix-stability-relative-column-difference-f32/1"

"""F32 symmetric relative interaction-matrix Frobenius difference."""
const INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1 =
    "pipewireao/interaction-matrix-stability-relative-frobenius-difference-f32/1"

"""
Boolean validity of matched first-difference outputs. `false` identifies the zero-valued
priming output; `true` identifies two valid differenced vectors.
"""
const MATCHED_CALIBRATION_FILTERED_V1 = "pipewireao/matched-calibration-filtered-bool/1"

"""Zero-based U64 structured calibration pattern coordinate."""
const CALIBRATION_PATTERN_INDEX_V1 = "pipewireao/calibration-pattern-index-u64/1"

"""U8 structured calibration polarity: positive `0` or negative `1`."""
const CALIBRATION_PROBE_POLARITY_V1 = "pipewireao/calibration-probe-polarity-u8/1"

"""Boolean marker for the final negative command in one probe cycle."""
const CALIBRATION_CYCLE_COMPLETE_V1 = "pipewireao/calibration-cycle-complete-bool/1"

"""F32 positive structured-probe amplitudes in corrector-coordinate order."""
const CALIBRATION_PROBE_AMPLITUDES_V1 = "pipewireao/calibration-probe-amplitudes-f32/1"

"""F32 modal positive commands in mode-by-corrector order."""
const MODAL_POSITIVE_COMMANDS_V1 = "pipewireao/modal-positive-commands-f32/1"

"""
U16 detector ADU values for one contiguous row block. SampleMetadata.offset is the
zero-based first detector row; axis 1 advances through contiguous detector rows and axis 2
through detector columns. Published blocks are complete and immutable.
"""
const RAW_PIXEL_ROW_BLOCK_V1 = "pipewireao/raw-pixel-row-block-u16/1"

"""Complete detector-plane U16 ADU values in detector (row, column) axis order."""
const RAW_DETECTOR_PIXELS_V1 = "pipewireao/raw-detector-pixels-u16/1"

"""
U16 detector ADU values for one acquisition-ordered readout block. SampleMetadata.offset is
the zero-based block ordinal; the prepared detector-readout mapping defines each value's
detector coordinate. Published blocks are complete and immutable.
"""
const RAW_PIXEL_READOUT_BLOCK_V1 = "pipewireao/raw-pixel-readout-block-u16/1"

"""
Calibrated F32 PDE values for one contiguous detector-row block. SampleMetadata.offset is
the zero-based first detector row; axis 1 advances through contiguous detector rows and axis
2 through detector columns. Published blocks are complete and immutable.
"""
const CALIBRATED_PIXEL_ROW_BLOCK_V1 = "pipewireao/calibrated-pixel-row-block-f32/1"

"""
Complete detector-plane calibrated F32 PDE values in detector (row, column) axis order.
Values are flat * (F32(raw) - background).
"""
const CALIBRATED_PIXELS_V1 = "pipewireao/calibrated-pixels-f32/1"

"""F32 reconstructed modal coefficients in controller mode order."""
const RECONSTRUCTED_MODES_V1 = "pipewireao/reconstructed-modes-f32/1"

"""
F32 LLOWFS modal coefficients in the calibrated shared LLOWFS mode order consumed by the
modes-to-controlled-VDM projection.
"""
const LLOWFS_MODES_V1 = "pipewireao/llowfs-modes-f32/1"

"""
Finite F32 parameters in the legacy LLOWFS model order, for downstream compatibility.
This is a sparse prepared artifact rather than a per-frame input.
"""
const ML_LLOWFS_PARAMETERS_V1 = "pipewireao/ml-llowfs-parameters-f32/1"

"""F32 physical wavefront-sensor response ratios in controller mode order."""
const PHYSICAL_OPTICAL_GAIN_V1 = "pipewireao/physical-optical-gain-f32/1"

"""F32 commanded optical-gain dither values in prepared mode order."""
const OPTICAL_GAIN_DITHER_COMMAND_V1 = "pipewireao/optical-gain-dither-command-f32/1"

"""U64 zero-based frame within the current optical-gain estimate window."""
const OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1 =
    "pipewireao/optical-gain-dither-window-frame-u64/1"

"""U64 zero-based sinusoid phase per prepared optical-gain dither mode."""
const OPTICAL_GAIN_DITHER_PHASE_INDICES_V1 =
    "pipewireao/optical-gain-dither-phase-indices-u64/1"

"""BOOL flag identifying the final frame in one optical-gain estimate window."""
const OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1 =
    "pipewireao/optical-gain-dither-window-complete-bool/1"

"""F32 raw reconstructed response in prepared optical-gain dither-mode order."""
const OPTICAL_GAIN_DITHER_RESPONSE_V1 = "pipewireao/optical-gain-dither-response-f32/1"

"""F32 calibration response per optical-gain dither mode."""
const OPTICAL_GAIN_CALIBRATION_RESPONSE_V1 =
    "pipewireao/optical-gain-calibration-response-f32/1"

"""F32 demodulated realized-command amplitude per dither mode."""
const OPTICAL_GAIN_COMMAND_AMPLITUDE_V1 = "pipewireao/optical-gain-command-amplitude-f32/1"

"""F32 demodulated reconstructed-response amplitude per dither mode."""
const OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1 =
    "pipewireao/optical-gain-response-amplitude-f32/1"

"""F32 physical optical-gain estimate in prepared dither-mode order."""
const OPTICAL_GAIN_DITHER_ESTIMATE_V1 = "pipewireao/optical-gain-dither-estimate-f32/1"

"""F32 response-minus-command phase lag in radians per dither mode."""
const OPTICAL_GAIN_PHASE_LAG_V1 = "pipewireao/optical-gain-phase-lag-f32/1"

"""BOOL validity of each optical-gain dither-mode estimate."""
const OPTICAL_GAIN_ESTIMATE_VALIDITY_V1 = "pipewireao/optical-gain-estimate-validity-bool/1"

"""BOOL flag identifying publication of one complete optical-gain estimate."""
const OPTICAL_GAIN_ESTIMATE_COMPLETE_V1 = "pipewireao/optical-gain-estimate-complete-bool/1"

"""BOOL acceptance status in prepared optical-gain dither-mode order."""
const OPTICAL_GAIN_ACCEPTED_DITHER_MODES_V1 =
    "pipewireao/optical-gain-accepted-dither-modes-bool/1"

"""BOOL update status in reconstructed controller-mode order."""
const OPTICAL_GAIN_UPDATED_RECONSTRUCTED_MODES_V1 =
    "pipewireao/optical-gain-updated-reconstructed-modes-bool/1"

"""BOOL validity for complete calibrated detector pixels."""
const CALIBRATED_PIXEL_VALIDITY_V1 = "pipewireao/calibrated-pixel-validity-bool/1"

"""F32 Thompson four-image SCC calibration response."""
const SCC_PUSH_PULL_RESPONSE_V1 = "pipewireao/scc-push-pull-response-f32/1"

"""F32 rolling fringed-minus-unfringed SCC chopped-image difference."""
const SCC_CHOPPED_IMAGE_DIFFERENCE_V1 = "pipewireao/scc-chopped-image-difference-f32/1"

"""BOOL per-pixel validity of the rolling SCC chopped-image difference."""
const SCC_CHOPPED_DIFFERENCE_VALIDITY_V1 =
    "pipewireao/scc-chopped-difference-validity-bool/1"

"""U8 chopped-image pair status: priming `0` or complete `1`."""
const SCC_CHOPPED_DIFFERENCE_STATUS_V1 = "pipewireao/scc-chopped-difference-status-u8/1"

"""F32 reference SCC chopped-image difference in complete detector-pixel order."""
const SCC_REFERENCE_IMAGE_V1 = "pipewireao/scc-reference-image-f32/1"

"""F32 reference-subtracted SCC residuals in prepared dark-hole pixel order."""
const SCC_SELECTED_PIXEL_RESIDUALS_V1 = "pipewireao/scc-selected-pixel-residuals-f32/1"

"""BOOL validity in prepared SCC dark-hole pixel order."""
const SCC_SELECTED_PIXEL_VALIDITY_V1 = "pipewireao/scc-selected-pixel-validity-bool/1"

"""F32 fraction of prepared SCC dark-hole pixels valid in the current chopped pair."""
const SCC_SELECTED_PIXEL_VALID_FRACTION_V1 =
    "pipewireao/scc-selected-pixel-valid-fraction-f32/1"

"""
F32 mask-aware SCC spatial-carrier measurements with all selected real components followed
by all selected imaginary components.
"""
const SCC_SPATIAL_CARRIER_MEASUREMENTS_V1 =
    "pipewireao/scc-spatial-carrier-measurements-f32/1"

"""Finite nonnegative F32 confidence weights aligned with an SCC measurement vector."""
const SCC_MEASUREMENT_WEIGHTS_V1 = "pipewireao/scc-measurement-weights-f32/1"

"""BOOL validity aligned with an SCC measurement vector."""
const SCC_MEASUREMENT_VALIDITY_V1 = "pipewireao/scc-measurement-validity-bool/1"

"""
F32 fraction of selected SCC spatial-carrier locations with adequate normalized-convolution
support.
"""
const SCC_SPATIAL_CARRIER_VALID_FRACTION_V1 =
    "pipewireao/scc-spatial-carrier-valid-fraction-f32/1"

"""
Finite F32 detector-space SCC carrier phase in radians and complete detector-pixel order.
"""
const SCC_CARRIER_PHASE_V1 = "pipewireao/scc-carrier-phase-f32/1"

"""Finite F32 normalized-convolution FIR coefficients in tap order."""
const SCC_SPATIAL_FIR_V1 = "pipewireao/scc-spatial-fir-f32/1"

"""
Finite F32 SCC interaction matrix with measurement rows and controller-coordinate columns.
"""
const SCC_INTERACTION_MATRIX_V1 = "pipewireao/scc-interaction-matrix-f32/1"

"""
Finite F32 SCC regularization operator with penalty rows and controller-coordinate columns.
"""
const SCC_REGULARIZATION_OPERATOR_V1 = "pipewireao/scc-regularization-operator-f32/1"

"""
F32 selected-pixel-residual to SCC-controller-error reconstructor, with rows in
controller-coordinate order and columns in prepared dark-hole pixel order.
"""
const SCC_RECONSTRUCTOR_V1 = "pipewireao/scc-reconstructor-f32/1"

"""
Calibrated F32 PDE values for one acquisition-ordered readout block. SampleMetadata.offset
is the zero-based block ordinal; the prepared detector-readout mapping defines each value's
detector coordinate. Published blocks are complete and immutable.
"""
const CALIBRATED_PIXEL_READOUT_BLOCK_V1 = "pipewireao/calibrated-pixel-readout-block-f32/1"

"""
Complete detector-plane finite F32 flat-calibration coefficients in detector (row, column)
axis order, measured in PDE/ADU. This is a sparse prepared artifact rather than a per-frame
input.
"""
const FLAT_CALIBRATION_V1 = "pipewireao/flat-calibration-f32/1"

"""
Complete detector-plane finite F32 background calibration in detector (row, column) axis
order, measured in detector ADU. This is a sparse prepared artifact rather than a per-frame
input.
"""
const BACKGROUND_CALIBRATION_V1 = "pipewireao/background-calibration-f32/1"

"""Logical detector-pixel selection mask in detector row-major order."""
const PIXEL_SELECTION_MASK_V1 = "pipewireao/pixel-selection-mask-bool/1"

"""Calibrated F32 pixels grouped by rectangular detector region."""
const REGION_PIXELS_V1 = "pipewireao/region-pixels-f32/1"

"""Calibrated F32 pixels for one center-of-gravity region."""
const CENTER_OF_GRAVITY_PIXELS_V1 = "pipewireao/center-of-gravity-pixels-f32/1"

"""Pair-interleaved F32 X/Y coordinates in selected-pixel order."""
const CENTER_OF_GRAVITY_COORDINATES_V1 = "pipewireao/center-of-gravity-coordinates-f32/1"

"""
Finite nonnegative F32 dimensionless per-pixel center-of-gravity weights in selected-pixel
order.
"""
const CENTER_OF_GRAVITY_WEIGHTS_V1 = "pipewireao/center-of-gravity-weights-f32/1"

"""F32 weighted center-of-gravity flux Σ(weight × calibrated pixel) in PDE."""
const WEIGHTED_CENTER_OF_GRAVITY_FLUX_V1 =
    "pipewireao/weighted-center-of-gravity-flux-f32/1"

"""Measured-minus-reference F32 center-of-gravity X/Y offset."""
const CENTER_OF_GRAVITY_OFFSET_V1 = "pipewireao/center-of-gravity-offset-f32/1"

"""Selected-region F32 center-of-gravity flux in PDE."""
const CENTER_OF_GRAVITY_FLUX_V1 = "pipewireao/center-of-gravity-flux-f32/1"

"""Boolean numerical and flux-threshold validity for one measurement."""
const CENTER_OF_GRAVITY_VALIDITY_V1 = "pipewireao/center-of-gravity-validity-bool/1"

"""F32 Shack-Hartmann slopes in subaperture-by-X/Y order."""
const SHACK_HARTMANN_SLOPES_V1 = "pipewireao/shack-hartmann-slopes-f32/1"

"""F32 Shack-Hartmann fluxes in subaperture order."""
const SHACK_HARTMANN_FLUX_V1 = "pipewireao/shack-hartmann-flux-f32/1"

"""F32 weighted Shack-Hartmann fluxes Σ(weight × calibrated pixel) in subaperture order."""
const WEIGHTED_SHACK_HARTMANN_FLUX_V1 = "pipewireao/weighted-shack-hartmann-flux-f32/1"

"""Boolean Shack-Hartmann measurement validity in subaperture order."""
const SHACK_HARTMANN_VALIDITY_V1 = "pipewireao/shack-hartmann-validity-bool/1"

"""
Eight F32 diagnostics for one complete Shack-Hartmann frame in zero-based order: 0 total
subaperture count, 1 valid count, 2 mean valid flux, 3 X slope RMS, 4 Y slope RMS, 5 hypot(X
RMS, Y RMS), 6 mean X slope, 7 mean Y slope. Subaperture count is in 1:2^24 so both counts
are exact Float32 integers. RMS is uncentered. Only validity-true rows contribute;
accumulation uses Float64. Zero valid rows or any nonfinite flux or slope in a valid row
yields NaN in fields 2 through 7 while counts retain their validity-derived values.
Processing still succeeds. Flux and slopes retain their input units, including detector
pixels for pixel-coordinate slopes; these are not calibrated physical tip, tilt, or focus
estimates. Inputs belong to the same measurement frame and metadata follows slopes.
"""
const SHACK_HARTMANN_SUMMARY_V1 = "pipewireao/shack-hartmann-summary-f32/1"

"""F32 detector-plane X/Y coordinates for one subaperture."""
const SHACK_HARTMANN_COORDINATES_V1 = "pipewireao/shack-hartmann-coordinates-f32/1"

"""F32 Shack-Hartmann reference slopes in subaperture-by-X/Y order."""
const SHACK_HARTMANN_REFERENCE_SLOPES_V1 =
    "pipewireao/shack-hartmann-reference-slopes-f32/1"

"""F32 pixel and flux thresholds in subaperture order."""
const SHACK_HARTMANN_THRESHOLDS_V1 = "pipewireao/shack-hartmann-thresholds-f32/1"

"""
Finite nonnegative F32 dimensionless per-subaperture weights in FGA row-major selected-pixel
order.
"""
const SHACK_HARTMANN_WEIGHTS_V1 = "pipewireao/shack-hartmann-weights-f32/1"

"""Boolean Shack-Hartmann active mask in subaperture order."""
const SHACK_HARTMANN_ACTIVE_V1 = "pipewireao/shack-hartmann-active-bool/1"

"""F32 matrix mapping pair-interleaved Shack-Hartmann slopes to a controller vector."""
const SHWFS_RECONSTRUCTOR_V1 = "pipewireao/shwfs-reconstructor-f32/1"

"""Zero-based U32 `(row, column)` origins for region extraction."""
const REGION_ORIGINS_V1 = "pipewireao/region-origins-u32/1"

"""Normalized F32 pyramid reconstruction pixels in pupil-major order."""
const PYRAMID_RECONSTRUCTION_PIXELS_V1 = "pipewireao/pyramid-reconstruction-pixels-f32/1"

"""
U64 calibration signature synchronized with one complete calibrated Bi-O-edge detector
frame.
"""
const BI_O_EDGE_FRAME_CALIBRATION_SIGNATURE_V1 =
    "pipewireao/bi-o-edge-frame-calibration-signature-u64/1"

"""U64 active calibration signature for complete-frame Bi-O-edge estimation."""
const BI_O_EDGE_CALIBRATION_SIGNATURE_V1 =
    "pipewireao/bi-o-edge-calibration-signature-u64/1"

"""F32 calibrated Bi-O-edge differential signal in selected-coordinate X/Y order."""
const BI_O_EDGE_SIGNAL_V1 = "pipewireao/bi-o-edge-signal-f32/1"

"""F32 positive current-frame divisor used to calibrate a Bi-O-edge differential signal."""
const BI_O_EDGE_NORMALIZATION_DIVISOR_V1 =
    "pipewireao/bi-o-edge-normalization-divisor-f32/1"

"""Boolean FGA-column-major valid support for one Bi-O-edge pupil region."""
const BI_O_EDGE_SUPPORT_V1 = "pipewireao/bi-o-edge-support-bool/1"

"""F32 Bi-O-edge reference differential signal in selected-coordinate X/Y order."""
const BI_O_EDGE_REFERENCE_SIGNAL_V1 = "pipewireao/bi-o-edge-reference-signal-f32/1"

"""F32 per-component optical gain for a Bi-O-edge differential signal."""
const BI_O_EDGE_OPTICAL_GAIN_V1 = "pipewireao/bi-o-edge-optical-gain-f32/1"

"""
U64 calibration signature synchronized with one complete calibrated Zernike pupil frame.
"""
const ZERNIKE_PUPIL_SIGNAL_FRAME_CALIBRATION_SIGNATURE_V1 =
    "pipewireao/zernike-pupil-signal-frame-calibration-signature-u64/1"

"""U64 active calibration signature for complete-frame Zernike pupil estimation."""
const ZERNIKE_PUPIL_SIGNAL_CALIBRATION_SIGNATURE_V1 =
    "pipewireao/zernike-pupil-signal-calibration-signature-u64/1"

"""F32 calibrated Zernike normalized pupil signal in selected-coordinate order."""
const ZERNIKE_PUPIL_SIGNAL_V1 = "pipewireao/zernike-pupil-signal-f32/1"

"""F32 current-frame divisor used to calibrate a Zernike pupil signal."""
const ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_DIVISOR_V1 =
    "pipewireao/zernike-pupil-signal-normalization-divisor-f32/1"

"""Boolean FGA-column-major valid support for a complete Zernike pupil image."""
const ZERNIKE_PUPIL_SIGNAL_SUPPORT_V1 = "pipewireao/zernike-pupil-signal-support-bool/1"

"""F32 complete-frame reference normalized pupil image for Zernike estimation."""
const ZERNIKE_PUPIL_SIGNAL_REFERENCE_PUPIL_IMAGE_V1 =
    "pipewireao/zernike-pupil-signal-reference-pupil-image-f32/1"

"""F32 accepted incidence-flux calibration divisor for Zernike pupil estimation."""
const ZERNIKE_PUPIL_SIGNAL_INCIDENCE_FLUX_V1 =
    "pipewireao/zernike-pupil-signal-incidence-flux-f32/1"

"""
F32 complete detector image with configured positive- and negative-defocus curvature
branches.
"""
const CURVATURE_PAIRED_IMAGE_V1 = "pipewireao/curvature-paired-image-f32/1"

"""
F32 complete branch-by-channel matrix for positive- and negative-defocus curvature
estimation.
"""
const CURVATURE_PAIRED_CHANNELS_V1 = "pipewireao/curvature-paired-channels-f32/1"

"""U64 calibration signature synchronized with one complete paired-curvature input."""
const CURVATURE_PAIRED_FRAME_CALIBRATION_SIGNATURE_V1 =
    "pipewireao/curvature-paired-frame-calibration-signature-u64/1"

"""U64 active calibration signature for paired-curvature estimation."""
const CURVATURE_PAIRED_CALIBRATION_SIGNATURE_V1 =
    "pipewireao/curvature-paired-calibration-signature-u64/1"

"""F32 reference-subtracted paired-curvature signal in full FGA column-major grid order."""
const CURVATURE_PAIRED_SIGNAL_V1 = "pipewireao/curvature-paired-signal-f32/1"

"""Boolean valid support for the full FGA column-major paired-curvature signal grid."""
const CURVATURE_PAIRED_SUPPORT_V1 = "pipewireao/curvature-paired-support-bool/1"

"""F32 paired-curvature reference signal in full FGA column-major grid order."""
const CURVATURE_PAIRED_REFERENCE_SIGNAL_V1 =
    "pipewireao/curvature-paired-reference-signal-f32/1"

"""
Two positive finite F32 paired-curvature branch scales in positive- and negative-defocus
order.
"""
const CURVATURE_PAIRED_BRANCH_SCALES_V1 = "pipewireao/curvature-paired-branch-scales-f32/1"

"""F32 calibrated pyramid I4Q differential signal in selected-coordinate X/Y order."""
const PYRAMID_I4Q_SIGNAL_V1 = "pipewireao/pyramid-i4q-signal-f32/1"

"""F32 positive current-frame divisor used to calibrate a pyramid I4Q signal."""
const PYRAMID_NORMALIZATION_DIVISOR_V1 = "pipewireao/pyramid-normalization-divisor-f32/1"

"""Boolean FGA-column-major valid support for one pyramid I4Q pupil region."""
const PYRAMID_I4Q_SUPPORT_V1 = "pipewireao/pyramid-i4q-support-bool/1"

"""F32 pyramid I4Q reference differential signal in selected-coordinate X/Y order."""
const PYRAMID_REFERENCE_SIGNAL_V1 = "pipewireao/pyramid-reference-signal-f32/1"

"""F32 per-component optical gain for a pyramid I4Q differential signal."""
const PYRAMID_OPTICAL_GAIN_V1 = "pipewireao/pyramid-optical-gain-f32/1"

"""F32 matrix mapping packed pyramid I4Q X-then-Y signals to a reconstructed vector."""
const PYRAMID_I4Q_RECONSTRUCTOR_V1 = "pipewireao/pyramid-i4q-reconstructor-f32/1"

"""Current-frame F32 mean intensity across all four pyramid pupils."""
const PYRAMID_MEAN_PUPIL_INTENSITY_V1 = "pipewireao/pyramid-mean-pupil-intensity-f32/1"

"""
F64 exact sums of raw U16 detector ADU values over the four configured pyramid-pupil
rectangles, in the configured pupil-origin order. Each sum is per exposure and remains in
ADC counts; no calibration, mask, normalization, or averaging is applied.
"""
const PYRAMID_PUPIL_FLUX_ADU_F64_V1 = "pipewireao/pyramid-pupil-flux-adu-f64/1"

"""Shared row-major Boolean pyramid pupil mask."""
const PYRAMID_PUPIL_MASK_V1 = "pipewireao/pyramid-pupil-mask-bool/1"

"""F32 matrix mapping pupil-major pyramid pixels to a reconstructed vector."""
const PWFS_RECONSTRUCTOR_V1 = "pipewireao/pwfs-reconstructor-f32/1"

"""
Physical correction provenance: U8 codes 0 none, 1 controlled, 2 extrapolated, 3 unknown.
"""
const PDM_CORRECTION_ROLE_V1 = "pipewireao/pdm-correction-role-u8/1"

"""Complete detector-plane F32 ADU values in detector (row, column) axis order."""
const DETECTOR_FRAME_V1 = "pipewireao/detector-frame-f32/1"

"""
Complete detector-plane F32 samples in electron-multiplication-register output-electron
units, in detector (row, column) axis order. The declared acquisition includes electron
multiplication and output-referred readout noise; the electron-multiplication gain has not
been divided out. Signed read-noise excursions are retained unless the acquisition
explicitly declares clipping. This quantity does not imply ADC conversion, quantization, or
gain normalization.
"""
const DETECTOR_EM_OUTPUT_ELECTRONS_V1 = "pipewireao/detector-em-output-electrons-f32/1"

"""
Complete detector-plane F32 samples in input-equivalent electrons, in detector (row, column)
axis order. Each electron-multiplication-register output-electron sample, including
output-referred readout noise, is divided by the declared positive electron-multiplication
gain. Signed read-noise excursions are retained unless the producer explicitly declares
clipping. Gain normalization preserves multiplication noise and does not imply ADC
conversion or quantization.
"""
const DETECTOR_INPUT_EQUIVALENT_ELECTRONS_V1 =
    "pipewireao/detector-input-equivalent-electrons-f32/1"

"""F32 atmosphere optical path difference in metres on the sampled pupil grid."""
const ATMOSPHERE_OPD_M_V1 = "pipewireao/atmosphere-opd-m-f32/1"

"""
F32 actuator commands in metres of wavefront optical path difference, in physical-actuator
order.
"""
const ACTUATOR_COMMAND_OPD_M_V1 = "pipewireao/actuator-command-opd-m-f32/1"

"""
Two F32 coefficients in metres of wavefront optical path difference, ordered (X ramp, Y
ramp), for the dimensionless normalized-pupil-xy basis. On the supported pupil, the additive
OPD is coefficient[1]*X + coefficient[2]*Y. A positive coefficient adds OPD increasing with
its corresponding pupil coordinate. These are normalized-pupil ramp amplitudes, not angular
pointing values or mirror displacements.
"""
const TIP_TILT_OPD_COEFFICIENTS_M_V1 = "pipewireao/tip-tilt-opd-coefficients-m-f32/1"

"""
F32 deformable-mirror wavefront optical path difference in metres on the sampled pupil grid.
"""
const DM_SURFACE_OPD_M_V1 = "pipewireao/dm-surface-opd-m-f32/1"

"""
One-based I32 indices into the column-major square actuator grid, in physical-actuator
order. Indices are unique and address the declared grid.
"""
const ACTUATOR_GRID_INDICES_V1 = "pipewireao/actuator-grid-indices-i32/1"

"""F32 normalized pupil X/Y actuator coordinates in (coordinate, actuator) axis order."""
const NORMALIZED_PUPIL_ACTUATOR_COORDINATES_V1 =
    "pipewireao/normalized-pupil-actuator-coordinates-f32/1"

"""
F32 dimensionless Cartesian pupil ramps in (pupil_x, pupil_y, component) axis order with two
components: component 1 is X and component 2 is Y. For one-based indices i and j on an
Nx-by-Ny raster, X=(i-(Nx+1)/2)/(Nx/2) and Y=(j-(Ny+1)/2)/(Ny/2). The sample centers lie
within [-1, 1], with each coordinate increasing along its corresponding array axis. The
first two axes use column-major logical order; no pupil support mask is applied to the basis
itself.
"""
const NORMALIZED_PUPIL_XY_V1 = "pipewireao/normalized-pupil-xy-f32/1"

"""
Boolean sampled annular pupil support in (pupil_x, pupil_y) axis order matching the first
two normalized-pupil-xy axes, in column-major logical order. True marks the transmitting
annulus and false marks the central obstruction or exterior. On the normalized cell-centered
pupil grid, support includes samples satisfying obstruction_ratio^2 <= X^2 + Y^2 <= 1, with
the producer declaring 0 <= obstruction_ratio < 1.
"""
const ANNULAR_PUPIL_SUPPORT_V1 = "pipewireao/annular-pupil-support-bool/1"

"""
F32 deformable-mirror influence functions in (sampled pupil pixel, actuator) axis order,
using column-major pupil pixel order and physical-actuator order.
"""
const INFLUENCE_FUNCTIONS_V1 = "pipewireao/influence-functions-f32/1"

"""F32 pupil optical path difference in metres on the sampled pupil grid."""
const PUPIL_OPD_M_V1 = "pipewireao/pupil-opd-m-f32/1"

"""F32 cell-integrated photon rates in photons per second on the configured sensor grid."""
const PHOTON_RATE_V1 = "pipewireao/photon-rate-f32/1"

"""
F32 cell-integrated focal-plane photon rates in photons per second on the configured focal
grid.
"""
const FOCAL_PHOTON_RATE_V1 = "pipewireao/focal-photon-rate-f32/1"

"""F32 residual pupil optical path difference in nanometres on the sampled pupil grid."""
const RESIDUAL_PUPIL_OPD_NM_V1 = "pipewireao/residual-pupil-opd-nm-f32/1"

"""
F32 pupil optical path difference RMS in nanometres, using the configured pupil support.
"""
const PUPIL_OPD_RMS_NM_V1 = "pipewireao/pupil-opd-rms-nm-f32/1"

"""
F32 applied physical-DM commands used by a completed model step, in actuator order and in
the declared command units.
"""
const APPLIED_PDM_COMMAND_V1 = "pipewireao/applied-pdm-command-f32/1"

"""
F32 applied physical-DM command in µm wavefront optical path difference, in concatenated
actuator order.
"""
const APPLIED_DM_COMMAND_OPD_UM_V1 = "pipewireao/applied-dm-command-opd-um-f32/1"

"""
F32 dimensionless matrix mapping controller metre-OPD coordinates to normalized-pupil
tip/tilt metre-OPD coefficients. Rows are the declared two stage axes.
"""
const CONTROLLER_TO_TIP_TILT_V1 = "pipewireao/controller-to-tip-tilt-f32/1"

"""
F32 dimensionless matrix mapping normalized-pupil tip/tilt metre-OPD coefficients to mirror
actuator metre-OPD commands in declared command order.
"""
const TIP_TILT_TO_ACTUATORS_V1 = "pipewireao/tip-tilt-to-actuators-f32/1"

const _SCHEMA_ELEMENT_TYPES = Dict{String,DataType}(
    BIQUAD_COEFFICIENTS_V1 => Float32,
    CONTROLLER_RESIDUAL_ERROR_V1 => Float32,
    PSEUDO_OPEN_LOOP_INTERACTION_MATRIX_V1 => Float32,
    PSEUDO_OPEN_LOOP_RECONSTRUCTOR_V1 => Float32,
    PSEUDO_OPEN_LOOP_CORRECTOR_TO_ERROR_V1 => Float32,
    CONTROLLER_COMMAND_V1 => Float32,
    CONTROLLER_STATE_V1 => Float32,
    CONTROLLER_CONSTRAINT_FEEDBACK_V1 => Float32,
    VDM_COMMAND_V1 => Float32,
    CONTROLLER_TO_VDM_PROJECTION_V1 => Float32,
    VDM_TO_CONTROLLER_PROJECTION_V1 => Float32,
    REQUESTED_PDM_COMMAND_V1 => Float32,
    PDM_SYSTEM_FLAT_V1 => Float32,
    PDM_ENGINEERING_OFFSET_V1 => Float32,
    INTER_ACTUATOR_STROKE_LIMIT_V1 => Float32,
    PREFERRED_PDM_COMMAND_V1 => Float32,
    PDM_COMMAND_TO_NULL_MODES_V1 => Float32,
    PDM_NULL_MODES_TO_COMMAND_V1 => Float32,
    DEMANDED_PDM_COMMAND_V1 => Float32,
    NATIVE_PDM_COMMAND_V1 => Float32,
    PDM_CONSTRAINT_FEEDBACK_V1 => Float32,
    VDM_CONSTRAINT_FEEDBACK_V1 => Float32,
    ACTIVE_TO_FULL_VDM_PROJECTION_V1 => Float32,
    FULL_TO_ACTIVE_VDM_PROJECTION_V1 => Float32,
    VDM_TO_PDM_PROJECTION_V1 => Float32,
    PDM_TO_VDM_PROJECTION_V1 => Float32,
    SHAPE_TO_HIDDEN_PROJECTION_V1 => Float32,
    HIDDEN_TO_SHAPE_PROJECTION_V1 => Float32,
    DOCRIME_EXCITATION_V1 => Float32,
    DOCRIME_EXCITATION_AMPLITUDE_V1 => Float32,
    DOCRIME_COVARIANCE_SAMPLE_COUNT_V1 => UInt64,
    DOCRIME_COVARIANCE_COMPLETE_V1 => Bool,
    DOCRIME_INTERACTION_MATRIX_STATUS_V1 => UInt8,
    DOCRIME_INTERACTION_MATRIX_STATUS_COORDINATES_V1 => UInt64,
    DOCRIME_CORRECTOR_VARIANCE_RANGE_V1 => Float32,
    DOCRIME_MAXIMUM_ABSOLUTE_CORRECTOR_CORRELATION_V1 => Float32,
    DOCRIME_MINIMUM_FACTORIZATION_PIVOT_RATIO_V1 => Float32,
    INTERACTION_MATRIX_STABILITY_STATUS_V1 => UInt8,
    CORRECTOR_COORDINATE_V1 => UInt64,
    INTERACTION_MATRIX_STABILITY_COLUMN_NORMS_V1 => Float32,
    INTERACTION_MATRIX_STABILITY_RELATIVE_COLUMN_V1 => Float32,
    INTERACTION_MATRIX_STABILITY_RELATIVE_FROBENIUS_V1 => Float32,
    MATCHED_CALIBRATION_FILTERED_V1 => Bool,
    CALIBRATION_PATTERN_INDEX_V1 => UInt64,
    CALIBRATION_PROBE_POLARITY_V1 => UInt8,
    CALIBRATION_CYCLE_COMPLETE_V1 => Bool,
    CALIBRATION_PROBE_AMPLITUDES_V1 => Float32,
    MODAL_POSITIVE_COMMANDS_V1 => Float32,
    RAW_PIXEL_ROW_BLOCK_V1 => UInt16,
    RAW_DETECTOR_PIXELS_V1 => UInt16,
    RAW_PIXEL_READOUT_BLOCK_V1 => UInt16,
    CALIBRATED_PIXEL_ROW_BLOCK_V1 => Float32,
    CALIBRATED_PIXELS_V1 => Float32,
    RECONSTRUCTED_MODES_V1 => Float32,
    LLOWFS_MODES_V1 => Float32,
    ML_LLOWFS_PARAMETERS_V1 => Float32,
    PHYSICAL_OPTICAL_GAIN_V1 => Float32,
    OPTICAL_GAIN_DITHER_COMMAND_V1 => Float32,
    OPTICAL_GAIN_DITHER_WINDOW_FRAME_V1 => UInt64,
    OPTICAL_GAIN_DITHER_PHASE_INDICES_V1 => UInt64,
    OPTICAL_GAIN_DITHER_WINDOW_COMPLETE_V1 => Bool,
    OPTICAL_GAIN_DITHER_RESPONSE_V1 => Float32,
    OPTICAL_GAIN_CALIBRATION_RESPONSE_V1 => Float32,
    OPTICAL_GAIN_COMMAND_AMPLITUDE_V1 => Float32,
    OPTICAL_GAIN_RESPONSE_AMPLITUDE_V1 => Float32,
    OPTICAL_GAIN_DITHER_ESTIMATE_V1 => Float32,
    OPTICAL_GAIN_PHASE_LAG_V1 => Float32,
    OPTICAL_GAIN_ESTIMATE_VALIDITY_V1 => Bool,
    OPTICAL_GAIN_ESTIMATE_COMPLETE_V1 => Bool,
    OPTICAL_GAIN_ACCEPTED_DITHER_MODES_V1 => Bool,
    OPTICAL_GAIN_UPDATED_RECONSTRUCTED_MODES_V1 => Bool,
    CALIBRATED_PIXEL_VALIDITY_V1 => Bool,
    SCC_PUSH_PULL_RESPONSE_V1 => Float32,
    SCC_CHOPPED_IMAGE_DIFFERENCE_V1 => Float32,
    SCC_CHOPPED_DIFFERENCE_VALIDITY_V1 => Bool,
    SCC_CHOPPED_DIFFERENCE_STATUS_V1 => UInt8,
    SCC_REFERENCE_IMAGE_V1 => Float32,
    SCC_SELECTED_PIXEL_RESIDUALS_V1 => Float32,
    SCC_SELECTED_PIXEL_VALIDITY_V1 => Bool,
    SCC_SELECTED_PIXEL_VALID_FRACTION_V1 => Float32,
    SCC_SPATIAL_CARRIER_MEASUREMENTS_V1 => Float32,
    SCC_MEASUREMENT_WEIGHTS_V1 => Float32,
    SCC_MEASUREMENT_VALIDITY_V1 => Bool,
    SCC_SPATIAL_CARRIER_VALID_FRACTION_V1 => Float32,
    SCC_CARRIER_PHASE_V1 => Float32,
    SCC_SPATIAL_FIR_V1 => Float32,
    SCC_INTERACTION_MATRIX_V1 => Float32,
    SCC_REGULARIZATION_OPERATOR_V1 => Float32,
    SCC_RECONSTRUCTOR_V1 => Float32,
    CALIBRATED_PIXEL_READOUT_BLOCK_V1 => Float32,
    FLAT_CALIBRATION_V1 => Float32,
    BACKGROUND_CALIBRATION_V1 => Float32,
    PIXEL_SELECTION_MASK_V1 => Bool,
    REGION_PIXELS_V1 => Float32,
    CENTER_OF_GRAVITY_PIXELS_V1 => Float32,
    CENTER_OF_GRAVITY_COORDINATES_V1 => Float32,
    CENTER_OF_GRAVITY_WEIGHTS_V1 => Float32,
    WEIGHTED_CENTER_OF_GRAVITY_FLUX_V1 => Float32,
    CENTER_OF_GRAVITY_OFFSET_V1 => Float32,
    CENTER_OF_GRAVITY_FLUX_V1 => Float32,
    CENTER_OF_GRAVITY_VALIDITY_V1 => Bool,
    SHACK_HARTMANN_SLOPES_V1 => Float32,
    SHACK_HARTMANN_FLUX_V1 => Float32,
    WEIGHTED_SHACK_HARTMANN_FLUX_V1 => Float32,
    SHACK_HARTMANN_VALIDITY_V1 => Bool,
    SHACK_HARTMANN_SUMMARY_V1 => Float32,
    SHACK_HARTMANN_COORDINATES_V1 => Float32,
    SHACK_HARTMANN_REFERENCE_SLOPES_V1 => Float32,
    SHACK_HARTMANN_THRESHOLDS_V1 => Float32,
    SHACK_HARTMANN_WEIGHTS_V1 => Float32,
    SHACK_HARTMANN_ACTIVE_V1 => Bool,
    SHWFS_RECONSTRUCTOR_V1 => Float32,
    REGION_ORIGINS_V1 => UInt32,
    PYRAMID_RECONSTRUCTION_PIXELS_V1 => Float32,
    BI_O_EDGE_FRAME_CALIBRATION_SIGNATURE_V1 => UInt64,
    BI_O_EDGE_CALIBRATION_SIGNATURE_V1 => UInt64,
    BI_O_EDGE_SIGNAL_V1 => Float32,
    BI_O_EDGE_NORMALIZATION_DIVISOR_V1 => Float32,
    BI_O_EDGE_SUPPORT_V1 => Bool,
    BI_O_EDGE_REFERENCE_SIGNAL_V1 => Float32,
    BI_O_EDGE_OPTICAL_GAIN_V1 => Float32,
    ZERNIKE_PUPIL_SIGNAL_FRAME_CALIBRATION_SIGNATURE_V1 => UInt64,
    ZERNIKE_PUPIL_SIGNAL_CALIBRATION_SIGNATURE_V1 => UInt64,
    ZERNIKE_PUPIL_SIGNAL_V1 => Float32,
    ZERNIKE_PUPIL_SIGNAL_NORMALIZATION_DIVISOR_V1 => Float32,
    ZERNIKE_PUPIL_SIGNAL_SUPPORT_V1 => Bool,
    ZERNIKE_PUPIL_SIGNAL_REFERENCE_PUPIL_IMAGE_V1 => Float32,
    ZERNIKE_PUPIL_SIGNAL_INCIDENCE_FLUX_V1 => Float32,
    CURVATURE_PAIRED_IMAGE_V1 => Float32,
    CURVATURE_PAIRED_CHANNELS_V1 => Float32,
    CURVATURE_PAIRED_FRAME_CALIBRATION_SIGNATURE_V1 => UInt64,
    CURVATURE_PAIRED_CALIBRATION_SIGNATURE_V1 => UInt64,
    CURVATURE_PAIRED_SIGNAL_V1 => Float32,
    CURVATURE_PAIRED_SUPPORT_V1 => Bool,
    CURVATURE_PAIRED_REFERENCE_SIGNAL_V1 => Float32,
    CURVATURE_PAIRED_BRANCH_SCALES_V1 => Float32,
    PYRAMID_I4Q_SIGNAL_V1 => Float32,
    PYRAMID_NORMALIZATION_DIVISOR_V1 => Float32,
    PYRAMID_I4Q_SUPPORT_V1 => Bool,
    PYRAMID_REFERENCE_SIGNAL_V1 => Float32,
    PYRAMID_OPTICAL_GAIN_V1 => Float32,
    PYRAMID_I4Q_RECONSTRUCTOR_V1 => Float32,
    PYRAMID_MEAN_PUPIL_INTENSITY_V1 => Float32,
    PYRAMID_PUPIL_FLUX_ADU_F64_V1 => Float64,
    PYRAMID_PUPIL_MASK_V1 => Bool,
    PWFS_RECONSTRUCTOR_V1 => Float32,
    PDM_CORRECTION_ROLE_V1 => UInt8,
    DETECTOR_FRAME_V1 => Float32,
    DETECTOR_EM_OUTPUT_ELECTRONS_V1 => Float32,
    DETECTOR_INPUT_EQUIVALENT_ELECTRONS_V1 => Float32,
    ATMOSPHERE_OPD_M_V1 => Float32,
    ACTUATOR_COMMAND_OPD_M_V1 => Float32,
    TIP_TILT_OPD_COEFFICIENTS_M_V1 => Float32,
    DM_SURFACE_OPD_M_V1 => Float32,
    ACTUATOR_GRID_INDICES_V1 => Int32,
    NORMALIZED_PUPIL_ACTUATOR_COORDINATES_V1 => Float32,
    NORMALIZED_PUPIL_XY_V1 => Float32,
    ANNULAR_PUPIL_SUPPORT_V1 => Bool,
    INFLUENCE_FUNCTIONS_V1 => Float32,
    PUPIL_OPD_M_V1 => Float32,
    PHOTON_RATE_V1 => Float32,
    FOCAL_PHOTON_RATE_V1 => Float32,
    RESIDUAL_PUPIL_OPD_NM_V1 => Float32,
    PUPIL_OPD_RMS_NM_V1 => Float32,
    APPLIED_PDM_COMMAND_V1 => Float32,
    APPLIED_DM_COMMAND_OPD_UM_V1 => Float32,
    CONTROLLER_TO_TIP_TILT_V1 => Float32,
    TIP_TILT_TO_ACTUATORS_V1 => Float32,
)

"""
    schema_element_type(schema::AbstractString)

Return the declared Julia element type for a catalogued scientific schema.
Return `nothing` for an external schema; its owner defines the element type.
Identifiers are matched exactly, without interpreting their spelling.
"""
schema_element_type(schema::AbstractString) = get(_SCHEMA_ELEMENT_TYPES, schema, nothing)

const _SCHEMA_CHANNELS = Dict{String,Tuple}(
    SHACK_HARTMANN_SUMMARY_V1 => (
        (name = "subaperture_count", index = 0, title = "Subaperture count", units = "count"),
        (name = "valid_count", index = 1, title = "Valid count", units = "count"),
        (name = "mean_flux", index = 2, title = "Mean valid flux", units = nothing),
        (name = "slope_rms_x", index = 3, title = "X slope RMS", units = nothing),
        (name = "slope_rms_y", index = 4, title = "Y slope RMS", units = nothing),
        (name = "slope_rms", index = 5, title = "Slope RMS", units = nothing),
        (name = "mean_x_slope", index = 6, title = "Mean X slope", units = nothing),
        (name = "mean_y_slope", index = 7, title = "Mean Y slope", units = nothing),
    ),
    PYRAMID_PUPIL_FLUX_ADU_F64_V1 => (
        (name = "pupil_1", index = 0, title = "Pupil 1", units = "ADC counts per exposure"),
        (name = "pupil_2", index = 1, title = "Pupil 2", units = "ADC counts per exposure"),
        (name = "pupil_3", index = 2, title = "Pupil 3", units = "ADC counts per exposure"),
        (name = "pupil_4", index = 3, title = "Pupil 4", units = "ADC counts per exposure"),
    ),
)

const _SCHEMA_CHANNEL_SHAPES = Dict{String,Tuple}(
    SHACK_HARTMANN_SUMMARY_V1 => (8,),
    PYRAMID_PUPIL_FLUX_ADU_F64_V1 => (4,),
)

"Return immutable scalar channel declarations, or () when none are declared."
schema_channels(schema::AbstractString) = get(_SCHEMA_CHANNELS, schema, ())

"Return the exact logical shape for declared channels, or nothing."
schema_channel_shape(schema::AbstractString) = get(_SCHEMA_CHANNEL_SHAPES, schema, nothing)
