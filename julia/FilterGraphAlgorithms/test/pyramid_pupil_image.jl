module PyramidPupilImageTests

using FilterGraphAlgorithms
using LinearAlgebra: transpose
using Test

import FilterGraphAlgorithms: commit!, discard!, process!, reset!

function pyramid_pixel_image_prepared()
    return prepare_algorithm(
        PyramidPupilImageF32,
        PyramidPupilImageF32Configuration(
            image_size=(4, 4),
            pupil_size=(2, 2),
            pupil_origins=[[0, 0], [0, 2], [2, 0], [2, 2]],
            pupil_mask=[true, false, true, true],
        ),
    )
end

pyramid_pixel_image_outputs() = (zeros(Float32, 4, 3), zeros(Float32, 1))

function pyramid_pixel_image_allocation_bytes(outputs, prepared, image)
    return @allocated process!(outputs..., prepared, image)
end

function pyramid_pixel_image_admitted_allocation_bytes(outputs, prepared, image)
    return @allocated FilterGraphAlgorithms._process_admitted!(
        outputs...,
        prepared,
        image,
        FilterGraphAlgorithms._NO_SAMPLE_METADATA,
    )
end

@testset "image-backed pyramid declaration preserves the frozen FGN contract" begin
    prepared = pyramid_pixel_image_prepared()
    @test algorithm_label(PyramidPupilImageF32) == "pyramid-pupil-image-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(PyramidPupilImageF32) ===
          PyramidPixelImageFailure
    @test map(field -> field.name, algorithm_configuration(PyramidPupilImageF32)) == (
        :image_size,
        :pupil_size,
        :pupil_origins,
        :pupil_mask,
    )
    @test map(port -> port.name, algorithm_ports(PyramidPupilImageF32)) == (
        :image,
        Symbol("reconstruction-pixels"),
        Symbol("mean-pupil-intensity"),
        Symbol("pupil-origins"),
        Symbol("pupil-mask"),
    )
    formats = algorithm_formats(prepared)
    @test getproperty.(formats, :shape) ==
          ((4, 4), (4, 3), (1,), (4, 2), (2, 2))
    @test getproperty.(formats, :schema) == (
        CALIBRATED_PIXELS_V1,
        PYRAMID_RECONSTRUCTION_PIXELS_V1,
        PYRAMID_MEAN_PUPIL_INTENSITY_V1,
        REGION_ORIGINS_V1,
        PYRAMID_PUPIL_MASK_V1,
    )
    @test prepared.plan isa PyramidPixelImagePlan
    @test prepared.workspace isa PyramidPixelWorkspace
end

@testset "image-backed pyramid reads detector pupils without a slab" begin
    prepared = pyramid_pixel_image_prepared()
    outputs = pyramid_pixel_image_outputs()
    image = permutedims(reshape(Float32.(1:16), 4, 4))

    @test process!(outputs..., prepared, image) === nothing
    @test outputs[1] == zeros(Float32, 4, 3)
    @test outputs[2] == Float32[8.5]
    @test pyramid_normalization_gain(prepared.workspace) == 1.0f0 / 8.5f0

    @test process!(outputs..., prepared, image) === nothing
    expected = Float32[
        1 5 6
        3 7 8
        9 13 14
        11 15 16
    ] ./ 8.5f0
    @test outputs[1] == expected
    @test outputs[2] == Float32[8.5]
end

@testset "image-backed pyramid parameters replace failure-atomically" begin
    prepared = pyramid_pixel_image_prepared()
    image = permutedims(reshape(Float32.(1:16), 4, 4))
    outputs = pyramid_pixel_image_outputs()
    process!(outputs..., prepared, image)

    replacement_mask = Bool[false true; true true]
    @test replace_parameter!(prepared, "pupil-mask", replacement_mask) === prepared
    process!(outputs..., prepared, image)
    @test outputs[1][1, :] == Float32[2, 5, 6] ./ 8.5f0

    original_plan = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        "pupil-mask",
        Bool[true false; false true],
    )
    @test prepared.plan === original_plan

    replacement_origins = UInt32[0 2; 0 0; 2 0; 2 2]
    @test replace_parameter!(prepared, "pupil-origins", replacement_origins) ===
          prepared
    process!(outputs..., prepared, image)
    @test outputs[1][1, :] == Float32[4, 7, 8] ./ 8.5f0
end

@testset "image-backed pyramid rejection preserves output and history" begin
    prepared = pyramid_pixel_image_prepared()
    image = permutedims(reshape(Float32.(1:16), 4, 4))
    outputs = pyramid_pixel_image_outputs()
    process!(outputs..., prepared, image)
    process!(outputs..., prepared, image)
    retained = map(copy, outputs)
    committed = pyramid_normalization_gain(prepared.workspace)
    nonfinite = copy(image)
    nonfinite[2, 3] = NaN32

    @test process!(outputs..., prepared, nonfinite) ===
          PyramidPixelImageNonFiniteInput
    @test outputs == retained
    @test pyramid_normalization_gain(prepared.workspace) == committed

    aliased_output = reshape(prepared.workspace.staged_pixels, 4, 3)
    @test_throws ArgumentError process!(
        aliased_output,
        outputs[2],
        prepared,
        image,
    )
end

@testset "image-backed pyramid admitted paths allocate no heap" begin
    prepared = pyramid_pixel_image_prepared()
    outputs = pyramid_pixel_image_outputs()
    image = permutedims(reshape(Float32.(1:16), 4, 4))
    nonfinite = copy(image)
    nonfinite[2, 3] = NaN32
    process!(outputs..., prepared, image)

    @test pyramid_pixel_image_allocation_bytes(outputs, prepared, image) == 0
    discard!(prepared.workspace)
    @test pyramid_pixel_image_admitted_allocation_bytes(
        outputs,
        prepared,
        image,
    ) == 0
    discard!(prepared.workspace)
    @test pyramid_pixel_image_allocation_bytes(outputs, prepared, nonfinite) == 0
    @test pyramid_pixel_image_admitted_allocation_bytes(
        outputs,
        prepared,
        nonfinite,
    ) == 0
    @test @allocated(commit!(prepared.workspace)) == 0
    @test @allocated(discard!(prepared.workspace)) == 0
    @test @allocated(reset!(prepared.workspace)) == 0
end

@testset "selected pupil lookup preserves rectangular and strided image semantics" begin
    prepared = prepare_algorithm(PyramidPupilImageF32,
        PyramidPupilImageF32Configuration(image_size=(4, 6),
            pupil_size=(2, 3),
            pupil_origins=[[0, 0], [0, 3], [2, 0], [2, 3]],
            pupil_mask=[true, false, true, false, true, false]))
    storage = zeros(Float32, 8, 12)
    image = @view storage[1:2:8, 1:2:12]
    image .= reshape(Float32.(1:24), 4, 6)
    image[1, 1] = -0.0f0
    image[1, 3] = reinterpret(Float32, UInt32(1))
    outputs = (zeros(Float32, 4, 3), zeros(Float32, 1))
    @test process!(outputs..., prepared, image) === nothing
    gain = pyramid_normalization_gain(prepared.workspace)
    expected = gain .* Float32[
        image[1, 1] image[1, 3] image[2, 2];
        image[1, 4] image[1, 6] image[2, 5];
        image[3, 1] image[3, 3] image[4, 2];
        image[3, 4] image[3, 6] image[4, 5]
    ]
    @test process!(outputs..., prepared, image) === nothing
    @test reinterpret(UInt32, vec(outputs[1])) == reinterpret(UInt32, vec(expected))
    @test pyramid_pixel_image_allocation_bytes(outputs, prepared, image) == 0
end

@testset "selected pupil candidates reject overflow before publication" begin
    prepared = pyramid_pixel_image_prepared()
    image = fill(1.0f0, 4, 4)
    outputs = pyramid_pixel_image_outputs()
    process!(outputs..., prepared, image)
    retained = map(copy, outputs)
    prepared.workspace.reciprocal_previous_frame_mean[1] = floatmax(Float32)
    image[2, 2] = 2.0f0
    @test process!(outputs..., prepared, image) === PyramidPixelImageNonFiniteOutput
    @test outputs == retained
    @test pyramid_normalization_gain(prepared.workspace) == floatmax(Float32)
    @test !prepared.workspace.candidate_ready
    indices = copy(prepared.plan.selected_indices)
    storage = reinterpret(Float32, vec(prepared.plan.selected_indices))
    aliased = reshape(@view(storage[1:12]), 4, 3)
    @test_throws ArgumentError process!(aliased, outputs[2], prepared, image)
    @test prepared.plan.selected_indices == indices
    reset!(prepared.workspace)
    @test process!(outputs..., prepared, fill(1.0f0, 4, 4)) === nothing
end

function check_selected_storage(plan, image)
    transposed = transpose(permutedims(image))
    regions = plan.pupil_regions
    rows, columns = regions.region_size
    mask = plan.normalization.mask
    for pupil in eachindex(regions.origins)
        first_row, first_column = regions.origins[pupil]
        region = image[first_row+1:first_row+rows, first_column+1:first_column+columns]
        expected = vec(permutedims(region))[mask]
        actual = parent(transposed)[plan.transposed_indices[:, pupil]]
        @test reinterpret(UInt32, actual) == reinterpret(UInt32, expected)
        @test image[plan.selected_indices[:, pupil]] == expected
    end
end

@testset "transposed selected lookup preserves mask and pupil coordinate order" begin
    prepared = prepare_algorithm(PyramidPupilImageF32,
        PyramidPupilImageF32Configuration(image_size=(5, 7), pupil_size=(2, 3),
            pupil_origins=[[0, 0], [2, 0], [2, 3], [0, 3]],
            pupil_mask=[true, false, true, false, true, false]))
    image = reshape(Float32.(1:35), 5, 7)
    image[1, 1] = -0.0f0
    image[1, 3] = nextfloat(0.0f0)
    check_selected_storage(prepared.plan, image)
    old_indices = prepared.plan.transposed_indices
    old_runs = prepared.plan.transposed_run_starts
    replace_parameter!(prepared, "pupil-mask", Bool[false true false; true false true])
    @test prepared.plan.transposed_indices !== old_indices
    @test prepared.plan.transposed_run_starts !== old_runs
    check_selected_storage(prepared.plan, image)
    old_indices = prepared.plan.transposed_indices
    old_runs = prepared.plan.transposed_run_starts
    replace_parameter!(prepared, "pupil-origins", UInt32[0 3; 2 3; 2 0; 0 0])
    @test prepared.plan.transposed_indices !== old_indices
    @test prepared.plan.transposed_run_starts !== old_runs
    check_selected_storage(prepared.plan, image)
    starts = prepared.plan.transposed_run_starts
    @test first(starts) == 1 && last(starts) == 4
    @test all(all(diff(@view prepared.plan.transposed_indices[starts[run]:starts[run+1]-1, pupil]) .== 1)
        for run in 1:length(starts)-1, pupil in 1:4)
    retained = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared, "pupil-mask", Bool[true false false; false true false],
    )
    @test prepared.plan === retained

    # A transpose of strided storage retains the generic logical-index path.
    storage = zeros(Float32, 14, 10)
    parent_view = @view storage[1:2:14, 1:2:10]
    parent_view .= permutedims(image)
    images = (image, transpose(permutedims(image)), transpose(parent_view))
    preparations = map(_ -> deepcopy(prepared), images)
    outputs = map(_ -> (zeros(Float32, 4, 3), zeros(Float32, 1)), images)
    for _ in 1:3
        for (output, source, values) in zip(outputs, preparations, images)
            @test process!(output..., source, values) === nothing
        end
        for output in outputs[2:end]
            @test reinterpret(UInt32, vec(output[1])) == reinterpret(UInt32, vec(outputs[1][1]))
            @test output[2] == outputs[1][2]
        end
    end
    for (output, source, values) in zip(outputs, preparations, images)
        @test pyramid_pixel_image_allocation_bytes(output, source, values) == 0
    end
end

@testset "transposed selected lookup cannot alias writable outputs" begin
    prepared = pyramid_pixel_image_prepared()
    image = transpose(permutedims(reshape(Float32.(1:16), 4, 4)))
    outputs = pyramid_pixel_image_outputs()
    indices = copy(prepared.plan.transposed_indices)
    storage = reinterpret(Float32, vec(prepared.plan.transposed_indices))
    aliased = reshape(@view(storage[1:12]), 4, 3)
    @test_throws ArgumentError process!(aliased, outputs[2], prepared, image)
    @test_throws ArgumentError process!(outputs[1], @view(storage[1:1]), prepared, image)
    @test prepared.plan.transposed_indices == indices
    runs = copy(prepared.plan.transposed_run_starts)
    run_storage = reinterpret(Float32, prepared.plan.transposed_run_starts)
    @test_throws ArgumentError process!(outputs[1], @view(run_storage[1:1]), prepared, image)
    @test prepared.plan.transposed_run_starts == runs
    # The same protection applies when the input uses the generic Matrix path.
    @test_throws ArgumentError process!(aliased, outputs[2], prepared, Matrix(image))
    @test prepared.plan.transposed_indices == indices

    # The new lookup also remains read-only when a caller supplies custom
    # workspace vectors projected over its bytes.
    values() = @view reinterpret(Float32, zeros(Int, 1))[1:1]
    staged = @view storage[1:12]
    workspace = FilterGraphAlgorithms._prepared_pyramid_pixel_workspace(
        Float32, values(), staged, values(), values(), zeros(Int32, 1), zeros(Int32, 1),
    )
    conflicting = FilterGraphAlgorithms.PreparedAlgorithm(
        PyramidPupilImageF32, prepared.plan, workspace, prepared.formats,
    )
    @test_throws ArgumentError process!(outputs..., conflicting, image)
    @test prepared.plan.transposed_indices == indices
end

@testset "native transpose run eligibility preserves indexed fallback" begin
    origins = [[0, 0], [60, 0], [60, 60], [0, 60]]
    image = transpose(permutedims(reshape(Float32.(1:120*120), 120, 120)))
    masks = (
        (vcat(trues(1023), falses(3600 - 1023)), false),
        (vcat(trues(1024), falses(3600 - 1024)), true),
    )
    for (mask, expected) in masks
        plan = FilterGraphAlgorithms.PyramidPixelImagePlan(
            (120, 120), (60, 60), origins, mask)
        @test FilterGraphAlgorithms._pyramid_image_use_transposed_runs(plan) === expected
        indexed = FilterGraphAlgorithms.PyramidPixelWorkspace(plan.normalization)
        selected = FilterGraphAlgorithms.PyramidPixelWorkspace(plan.normalization)
        @test FilterGraphAlgorithms._stage_pyramid_selected_pixels_indexed!(
            indexed, plan, image, 0.001f0) ===
              FilterGraphAlgorithms._stage_pyramid_selected_pixels!(
                selected, plan, image, 0.001f0)
        @test reinterpret(UInt32, indexed.staged_pixels) ==
              reinterpret(UInt32, selected.staged_pixels)
        @test @allocated(FilterGraphAlgorithms._stage_pyramid_selected_pixels!(
            selected, plan, image, 0.001f0)) == 0
    end
    for (run_width, expected) in ((15, false), (16, true))
        mask = falses(3600)
        for row in 0:59
            mask[60row+1:60row+run_width] .= true
            mask[60row+61-run_width:60row+60] .= true
        end
        plan = FilterGraphAlgorithms.PyramidPixelImagePlan(
            (120, 120), (60, 60), origins, mask)
        @test count(mask) ÷ (length(plan.transposed_run_starts) - 1) == run_width
        @test FilterGraphAlgorithms._pyramid_image_use_transposed_runs(plan) === expected
        indexed = FilterGraphAlgorithms.PyramidPixelWorkspace(plan.normalization)
        selected = FilterGraphAlgorithms.PyramidPixelWorkspace(plan.normalization)
        @test FilterGraphAlgorithms._stage_pyramid_selected_pixels_indexed!(
            indexed, plan, image, 0.001f0) ===
              FilterGraphAlgorithms._stage_pyramid_selected_pixels!(
                selected, plan, image, 0.001f0)
        @test reinterpret(UInt32, indexed.staged_pixels) ==
              reinterpret(UInt32, selected.staged_pixels)
        @test @allocated(FilterGraphAlgorithms._stage_pyramid_selected_pixels!(
            selected, plan, image, 0.001f0)) == 0
    end
end

@testset "dense pupil publication preserves storage layouts, tails and history" begin
    for count in (1, 3, 4, 7, 8, 15, 16, 17, 63, 64, 65, 2776), transposed in (false, true)
        config = PyramidPupilImageF32Configuration(
            image_size=(2, 2count), pupil_size=(1, count),
            pupil_origins=[[0, 0], [0, count], [1, 0], [1, count]],
            pupil_mask=fill(true, count),
        )
        dense = prepare_algorithm(PyramidPupilImageF32, config)
        generic = deepcopy(dense)
        output = (zeros(Float32, 4, count), zeros(Float32, 1))
        backing = zeros(Float32, 8, 2count)
        reference = (@view(backing[1:2:8, 1:2:2count]), zeros(Float32, 1))
        image = reshape(Float32[1 + mod(7919i, 127) / 16 for i in 1:4count], 2, 2count)
        image[1, 1] = -0f0
        input = transposed ? transpose(permutedims(image)) : image
        for _ in 1:3
            gain = dense.workspace.reciprocal_previous_frame_mean[1]
            expected = [gain * input[pupil <= 2 ? 1 : 2, (isodd(pupil) ? 0 : count) + selected]
                        for pupil in 1:4, selected in 1:count]
            @test process!(output..., dense, input) === nothing
            @test process!(reference..., generic, input) === nothing
            @test reinterpret(UInt32, vec(output[1])) == reinterpret(UInt32, vec(expected))
            @test reinterpret(UInt32, vec(output[1])) == reinterpret(UInt32, vec(collect(reference[1])))
            @test output[2] == reference[2]
            for name in fieldnames(typeof(dense.workspace))
                @test isequal(getfield(dense.workspace, name), getfield(generic.workspace, name))
            end
        end
        @test pyramid_pixel_image_allocation_bytes(output, dense, input) == 0
        @test pyramid_pixel_image_admitted_allocation_bytes(output, dense, input) == 0
        # Staging permits a complete caller input/output overlap. SIMD only
        # promises that output is disjoint from owned workspace, not input.
        alias_input = copy(image)
        aliased = (reshape(alias_input, 4, count), zeros(Float32, 1))
        separate = deepcopy(aliased)
        a, b = prepare_algorithm(PyramidPupilImageF32, config), prepare_algorithm(PyramidPupilImageF32, config)
        @test process!(aliased..., a, alias_input) === nothing
        @test process!(separate..., b, image) === nothing
        @test reinterpret(UInt32, vec(aliased[1])) == reinterpret(UInt32, vec(separate[1]))
    end
end

@testset "pupil publication retains custom workspace and mean views" begin
    dense = pyramid_pixel_image_prepared()
    source = dense.workspace
    values = map((source.reciprocal_previous_frame_mean, source.staged_pixels,
                  source.staged_mean_pupil_intensity, source.staged_reciprocal_current_frame_mean)) do value
        view(copy(value), :)
    end
    workspace = FilterGraphAlgorithms._prepared_pyramid_pixel_workspace(
        Float32, values..., copy(source.status), copy(source.completion_status),
    )
    generic = FilterGraphAlgorithms.PreparedAlgorithm(
        PyramidPupilImageF32, dense.plan, workspace, dense.formats,
    )
    image = reshape(Float32.(1:16), 4, 4)
    dense_mean, generic_mean = zeros(Float32, 3), zeros(Float32, 3)
    a = (zeros(Float32, 4, 3), @view(dense_mean[2:2]))
    b = (zeros(Float32, 4, 3), @view(generic_mean[2:2]))
    for _ in 1:3
        @test process!(a..., dense, image) === nothing
        @test process!(b..., generic, image) === nothing
        @test reinterpret(UInt32, vec(a[1])) == reinterpret(UInt32, vec(b[1]))
        @test dense_mean == generic_mean
    end
    @test pyramid_pixel_image_allocation_bytes(a, dense, image) == 0
    @test pyramid_pixel_image_allocation_bytes(b, generic, image) == 0
end

end
