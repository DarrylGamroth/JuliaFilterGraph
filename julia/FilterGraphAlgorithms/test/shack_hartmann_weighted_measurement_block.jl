module ShackHartmannWeightedMeasurementBlockTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: commit!, discard!, process!, replace_parameter!, reset!

const IMAGE = Float32[
    1 1.1 1.2 1.3 1.4 1.5
    1.2 1.3 1.4 1.5 1.6 1.7
    1.4 1.5 1.6 1.7 1.8 1.9
    1.6 1.7 1.8 1.9 2.0 2.1
    1.8 1.9 2.0 2.1 2.2 2.3
    2.0 2.1 2.2 2.3 2.4 2.5
]
const ORIGINS = [[0, 0], [0, 3], [3, 0], [3, 3]]
const WEIGHTS = [
    Float32[1, 0.5, 2, 1.5],
    Float32[0.25, 3, 0.75, 1],
    Float32[2, 1, 0, 0.5],
    Float32[1.25, 0.5, 1.5, 2],
]
const MEASUREMENTS_SCHEMA = "org.example.ao.weighted-shack-hartmann-measurement-block/1"
const COLUMNS_SCHEMA = "org.example.ao.reconstructor-column-indices/1"
const COUNT_SCHEMA = "org.example.ao.measurement-block-count/1"

function weighted_block_prepared(
    ;
    row_block_rows=1,
    origins=ORIGINS,
    weights=WEIGHTS,
)
    return prepare_algorithm(
        ShackHartmannWeightedMeasurementBlockF32,
        ShackHartmannWeightedMeasurementBlockF32Configuration(
            image_size=(6, 6),
            row_block_rows=row_block_rows,
            subaperture_size=(2, 2),
            subaperture_count=4,
            initial_subaperture_origins=origins,
            coordinate_scale=1.0f0,
            pixel_threshold=0.0f0,
            flux_threshold=0.0f0,
            reference_slopes=Float32[0.1, -0.2, 0.2, 0.1, -0.1, 0.3, 0.0, -0.1],
            active=Bool[true, true, true, true],
            initial_weights=weights,
            measurements_schema=MEASUREMENTS_SCHEMA,
            column_indices_schema=COLUMNS_SCHEMA,
            measurement_count_schema=COUNT_SCHEMA,
        ),
    )
end

function weighted_complete_prepared(; origins=ORIGINS, weights=WEIGHTS)
    return prepare_algorithm(
        ShackHartmannWeightedCentroidF32,
        ShackHartmannWeightedCentroidF32Configuration(
            image_size=(6, 6),
            image_schema=CALIBRATED_PIXELS_V1,
            subaperture_size=(2, 2),
            subaperture_count=4,
            initial_subaperture_origins=origins,
            coordinate_scale=1.0f0,
            pixel_threshold=0.0f0,
            flux_threshold=0.0f0,
            reference_slopes=Float32[0.1, -0.2, 0.2, 0.1, -0.1, 0.3, 0.0, -0.1],
            active=Bool[true, true, true, true],
            initial_weights=weights,
        ),
    )
end

function unweighted_block_prepared(; row_block_rows=1, origins=ORIGINS)
    return prepare_algorithm(
        ShackHartmannMeasurementBlockF32,
        ShackHartmannMeasurementBlockF32Configuration(
            image_size=(6, 6),
            row_block_rows=row_block_rows,
            subaperture_size=(2, 2),
            subaperture_count=4,
            initial_subaperture_origins=origins,
            coordinate_scale=1.0f0,
            pixel_threshold=0.0f0,
            flux_threshold=0.0f0,
            reference_slopes=Float32[0.1, -0.2, 0.2, 0.1, -0.1, 0.3, 0.0, -0.1],
            active=Bool[true, true, true, true],
            measurements_schema=MEASUREMENTS_SCHEMA,
            column_indices_schema=COLUMNS_SCHEMA,
            measurement_count_schema=COUNT_SCHEMA,
        ),
    )
end

function complete_measurements(prepared, image)
    slopes = zeros(Float32, 4, 2)
    flux = zeros(Float32, 4)
    validity = falses(4)
    process!(slopes, flux, validity, prepared, image)
    measurements = Vector{Float32}(undef, 8)
    for subaperture in 1:4
        measurements[2 * subaperture - 1] = slopes[subaperture, 1]
        measurements[2 * subaperture] = slopes[subaperture, 2]
    end
    return measurements
end

function collect_measurements!(
    prepared,
    image;
    sequence=1,
    first_discontinuity=false,
)
    measurements = fill(-1.0f0, 8)
    columns = fill(typemax(UInt32), 8)
    count = fill(typemax(UInt32), 1)
    collected = zeros(Float32, 8)
    seen = falses(8)
    block_rows = prepared.plan.block_rows
    for first_row in 0:block_rows:(size(image, 1) - block_rows)
        block = @view image[(first_row + 1):(first_row + block_rows), :]
        sample = SampleMetadata(
            sequence,
            first_row;
            terminal=first_row + block_rows == size(image, 1),
        )
        expected = SampleMetadata(
            sample;
            discontinuity=first_discontinuity && iszero(first_row),
        )
        @test process!(
            measurements,
            columns,
            count,
            prepared,
            block,
            (row_block=sample,),
        ) == AlgorithmProcessComplete(expected)
        for index in 1:Int(count[1])
            column = Int(columns[index]) + 1
            @test !seen[column]
            seen[column] = true
            collected[column] = measurements[index]
        end
    end
    return collected, seen
end

function process_allocation_bytes(outputs, prepared, block, metadata)
    return @allocated process!(outputs..., prepared, block, metadata)
end

@testset "weighted Shack-Hartmann measurement declaration and storage" begin
    prepared = weighted_block_prepared()
    @test algorithm_label(ShackHartmannWeightedMeasurementBlockF32) ==
          "shack-hartmann-weighted-measurement-block-f32"
    @test FilterGraphAlgorithms.algorithm_failure_type(
        ShackHartmannWeightedMeasurementBlockF32,
    ) === ShackHartmannMeasurementBlockFailure
    @test map(port -> port.name, algorithm_ports(ShackHartmannWeightedMeasurementBlockF32)) == (
        Symbol("row-block"),
        :measurements,
        :columns,
        :count,
        Symbol("subaperture-origins"),
        :coordinates,
        Symbol("reference-slopes"),
        :thresholds,
        :active,
        :weights,
    )
    @test isconcretetype(typeof(prepared.plan))
    @test isconcretetype(typeof(prepared.workspace))
    @test all(isconcretetype, fieldtypes(typeof(prepared.plan)))
    @test all(isconcretetype, fieldtypes(typeof(prepared.workspace)))
    @test prepared.plan.sensing isa ShackHartmannWeightedCentroidPlan
    @test prepared.plan.sensing.weights isa Matrix{Float32}
end

@testset "weighted measurement blocks match complete-frame sensing" begin
    for block_rows in (1, 2, 3)
        block = weighted_block_prepared(row_block_rows=block_rows)
        complete = weighted_complete_prepared()
        collected, seen = collect_measurements!(block, IMAGE; sequence=block_rows)
        @test all(seen)
        @test collected == complete_measurements(complete, IMAGE)
        @test !block.workspace.frame.active
    end

    shuffled_origins = [ORIGINS[4], ORIGINS[2], ORIGINS[3], ORIGINS[1]]
    block = weighted_block_prepared(origins=shuffled_origins)
    complete = weighted_complete_prepared(origins=shuffled_origins)
    collected, seen = collect_measurements!(block, IMAGE; sequence=11)
    @test all(seen)
    @test collected == complete_measurements(complete, IMAGE)
end

@testset "unit weighted blocks equal regular measurement blocks" begin
    weighted = weighted_block_prepared(weights=[ones(Float32, 4) for _ in 1:4])
    regular = unweighted_block_prepared()
    weighted_measurements, weighted_seen = collect_measurements!(weighted, IMAGE; sequence=12)
    regular_measurements, regular_seen = collect_measurements!(regular, IMAGE; sequence=12)
    @test weighted_seen == regular_seen
    @test weighted_seen == trues(8)
    @test weighted_measurements == regular_measurements
end

@testset "weighted measurement block replacement is cold and failure-atomic" begin
    prepared = weighted_block_prepared()
    original = prepared.plan
    @test_throws DimensionMismatch replace_parameter!(
        prepared,
        :weights,
        ones(Float32, 4, 3),
    )
    @test prepared.plan === original
    @test !prepared.workspace.frame.active
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :weights,
        Float32[1 1 1 1; 1 1 NaN 1; 1 1 1 1; 1 1 1 1],
    )
    @test prepared.plan === original

    replacement = ones(Float32, 4, 4)
    @test replace_parameter!(prepared, :weights, replacement) === prepared
    @test prepared.plan !== original
    replacement .= 0.0f0
    @test all(==(1.0f0), prepared.plan.sensing.weights)
end

@testset "weighted measurement replacement preserves or abandons an active frame" begin
    prepared = weighted_block_prepared()
    outputs = (zeros(Float32, 8), zeros(UInt32, 8), zeros(UInt32, 1))
    first = @view IMAGE[1:1, :]
    second = @view IMAGE[2:2, :]
    first_sample = SampleMetadata(31, 0)
    @test process!(outputs..., prepared, first, (row_block=first_sample,)) ==
          AlgorithmProcessComplete(first_sample)
    original = prepared.plan
    @test_throws ArgumentError replace_parameter!(
        prepared,
        :weights,
        Float32[1 1 1 1; 1 1 NaN 1; 1 1 1 1; 1 1 1 1],
    )
    @test prepared.plan === original
    @test prepared.workspace.frame.active
    second_sample = SampleMetadata(31, 1)
    @test process!(outputs..., prepared, second, (row_block=second_sample,)) ==
          AlgorithmProcessComplete(second_sample)
    @test outputs[3] == UInt32[4]
    @test outputs[1][1:4] ==
          complete_measurements(weighted_complete_prepared(), IMAGE)[1:4]

    @test replace_parameter!(prepared, :weights, ones(Float32, 4, 4)) === prepared
    @test !prepared.workspace.frame.active
    @test process!(
        outputs...,
        prepared,
        @view(IMAGE[3:3, :]),
        (row_block=SampleMetadata(31, 2),),
    ) == AlgorithmProcessDeferred()
    collected, seen = collect_measurements!(
        prepared,
        IMAGE;
        sequence=32,
        first_discontinuity=true,
    )
    @test all(seen)
    @test collected == complete_measurements(
        weighted_complete_prepared(weights=[ones(Float32, 4) for _ in 1:4]),
        IMAGE,
    )
end

@testset "weighted blocks preserve invalid flux and active-mask semantics" begin
    image = copy(IMAGE)
    image[1, 1] = NaN32
    prepared = weighted_block_prepared(weights=[zeros(Float32, 4) for _ in 1:4])
    collected, seen = collect_measurements!(prepared, image; sequence=33)
    @test all(seen)
    @test all(iszero, collected)
    @test isnan(prepared.workspace.flux[1])
    @test !prepared.workspace.validity[1]

    prepared = weighted_block_prepared()
    @test replace_parameter!(prepared, :active, Bool[false, true, true, true]) ===
          prepared
    collected, seen = collect_measurements!(
        prepared,
        IMAGE;
        sequence=34,
        first_discontinuity=true,
    )
    @test all(seen)
    @test collected[1:2] == zeros(Float32, 2)
    @test prepared.workspace.flux[1] > 0.0f0
    @test prepared.workspace.validity[1]
end

@testset "weighted measurement block interruption, lifecycle, and hot path" begin
    prepared = weighted_block_prepared()
    outputs = (zeros(Float32, 8), zeros(UInt32, 8), zeros(UInt32, 1))
    first = @view IMAGE[1:1, :]
    second = @view IMAGE[2:2, :]
    first_metadata = (row_block=SampleMetadata(20, 0),)
    second_metadata = (row_block=SampleMetadata(20, 1),)
    @test @inferred Union{AlgorithmProcessResult,ShackHartmannMeasurementBlockFailure} process!(
        outputs...,
        prepared,
        first,
        first_metadata,
    ) == AlgorithmProcessComplete(first_metadata.row_block)
    @test process_allocation_bytes(outputs, prepared, second, second_metadata) == 0
    discard!(prepared.workspace)
    @test !prepared.workspace.frame.active

    process!(outputs..., prepared, first, (row_block=SampleMetadata(21, 0),))
    terminal = @view IMAGE[6:6, :]
    @test process!(
        outputs...,
        prepared,
        terminal,
        (row_block=SampleMetadata(21, 5; terminal=true),),
    ) == AlgorithmProcessDeferred()
    @test !prepared.workspace.frame.active

    process!(outputs..., prepared, first, (row_block=SampleMetadata(23, 0),))
    @test process_allocation_bytes(
        outputs,
        prepared,
        terminal,
        (row_block=SampleMetadata(23, 5; terminal=true),),
    ) == 0
    @test !prepared.workspace.frame.active
    @test prepared.workspace.frame.pending_discontinuity

    for first_row in 0:4
        block = @view IMAGE[(first_row + 1):(first_row + 1), :]
        sample = SampleMetadata(24, first_row)
        expected = SampleMetadata(sample; discontinuity=iszero(first_row))
        @test process!(outputs..., prepared, block, (row_block=sample,)) ==
              AlgorithmProcessComplete(expected)
    end
    @test process_allocation_bytes(
        outputs,
        prepared,
        terminal,
        (row_block=SampleMetadata(24, 5; terminal=true),),
    ) == 0
    @test !prepared.workspace.frame.active
    @test !prepared.workspace.frame.pending_discontinuity
    @test outputs[3] == UInt32[0]

    reset!(prepared.workspace)
    process!(outputs..., prepared, first, (row_block=SampleMetadata(22, 0),))
    @test prepared.workspace.frame.active
    @test @allocated(discard!(prepared.workspace)) == 0
    @test @allocated(reset!(prepared.workspace)) == 0
    @test @allocated(commit!(prepared.workspace)) == 0
end

@testset "weighted measurement blocks compose with incremental reconstruction" begin
    sensing = weighted_block_prepared()
    matrix = Float32[
        1 2 3 4 5 6 7 8
        -1 0.5 1.5 -2 0.25 2 -0.5 1
    ]
    reconstruction = prepare_algorithm(
        IncrementalDenseReconstructorF32,
        IncrementalDenseReconstructorF32Configuration(
            measurement_count=8,
            measurement_block_capacity=8,
            blocks_per_sample=6,
            reconstructed_count=2,
            initial_reconstructor=vec(permutedims(matrix)),
            measurements_schema=MEASUREMENTS_SCHEMA,
            column_indices_schema=COLUMNS_SCHEMA,
            measurement_count_schema=COUNT_SCHEMA,
            reconstructed_schema="org.example.ao.weighted-reconstructed/1",
            reconstructor_schema="org.example.ao.weighted-reconstructor/1",
        ),
    )
    measurements = zeros(Float32, 8)
    columns = zeros(UInt32, 8)
    count = zeros(UInt32, 1)
    reconstructed = zeros(Float32, 2)
    for first_row in 0:5
        sample = SampleMetadata(30, first_row; terminal=first_row == 5)
        block = @view IMAGE[(first_row + 1):(first_row + 1), :]
        @test process!(
            measurements,
            columns,
            count,
            sensing,
            block,
            (row_block=sample,),
        ) == AlgorithmProcessComplete(sample)
        result = process!(
            reconstructed,
            reconstruction,
            measurements,
            columns,
            count,
            (measurements=sample, columns=nothing, count=nothing),
        )
        if sample.terminal
            @test result == AlgorithmProcessComplete(SampleMetadata(sample; offset=0))
        else
            @test result == AlgorithmProcessDeferred()
        end
    end
    @test isapprox(
        reconstructed,
        matrix * complete_measurements(weighted_complete_prepared(), IMAGE);
        atol=1.0f-6,
        rtol=1.0f-5,
    )
end


@testset "packed weighted blocks retain scientific publication and aliases" begin
    common=(;image_size=(64,64),subaperture_size=(8,8),subaperture_count=64,
        initial_subaperture_origins=[[8i,8j] for i in 0:7 for j in 0:7],
        coordinate_scale=.75f0,pixel_threshold=.5f0,flux_threshold=1f0,
        reference_slopes=fill(.125f0,128),active=[i%3!=0 for i in 1:64],
        initial_weights=[[Float32(mod(i+3j,13))/7f0 for j in 1:64] for i in 1:64])
    full=prepare_algorithm(ShackHartmannWeightedCentroidF32,
        ShackHartmannWeightedCentroidF32Configuration(;common...,image_schema=CALIBRATED_PIXELS_V1))
    block=prepare_algorithm(ShackHartmannWeightedMeasurementBlockF32,
        ShackHartmannWeightedMeasurementBlockF32Configuration(;common...,row_block_rows=8,
            measurements_schema=MEASUREMENTS_SCHEMA,column_indices_schema=COLUMNS_SCHEMA,
            measurement_count_schema=COUNT_SCHEMA))
    image=[Float32(mod(7i+11j,31)-3)/8f0 for i in 1:64,j in 1:64]
    standalone = ShackHartmannWeightedMeasurementBlockPlan(full.plan, 8)
    @test standalone.sensing.packed_weights == permutedims(full.plan.weights)
    @test isnothing(full.plan.packed_weights)
    slopes,flux,valid=zeros(Float32,64,2),zeros(Float32,64),falses(64)
    process!(slopes,flux,valid,full,image)
    expected=vec(permutedims(slopes))
    outputs=(zeros(Float32,128),zeros(UInt32,128),zeros(UInt32,1))
    collected=zeros(Float32,128);seen=falses(128)
    for first_row in 0:8:56
        sample=SampleMetadata(1,first_row;terminal=first_row==56)
        process!(outputs...,block,@view(image[first_row+1:first_row+8,:]),(row_block=sample,))
        for i in 1:Int(outputs[3][1])
            column=Int(outputs[2][i])+1
            @test !seen[column]
            seen[column]=true;collected[column]=outputs[1][i]
        end
    end
    @test all(seen)
    @test reinterpret(UInt32,collected)==reinterpret(UInt32,expected)
    @test block.plan.sensing.packed_weights == permutedims(block.plan.sensing.weights)
    cache=block.plan.sensing.packed_weights
    metadata=(row_block=SampleMetadata(2,0;terminal=false),)
    input=@view image[1:8,:]
    @test_throws ArgumentError process!(@view(vec(cache)[1:128]),outputs[2],outputs[3],block,input,metadata)
    @test_throws ArgumentError process!(outputs[1],@view(reinterpret(UInt32,vec(cache))[1:128]),outputs[3],block,input,metadata)
    @test_throws ArgumentError process!(outputs[1],outputs[2],@view(reinterpret(UInt32,vec(cache))[1:1]),block,input,metadata)
    process!(outputs...,block,input,metadata)
    reset!(block.workspace)
    @test process_allocation_bytes(outputs,block,input,metadata)==0
    replace_parameter!(block,:weights,fill(.5f0,64,64))
    @test block.plan.sensing.packed_weights == fill(.5f0,64,64)
end

end
