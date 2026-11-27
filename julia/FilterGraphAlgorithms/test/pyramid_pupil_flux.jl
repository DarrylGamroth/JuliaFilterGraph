module PyramidPupilFluxTests

using FilterGraphAlgorithms
using Test

import FilterGraphAlgorithms: process!

struct ShiftedDetector{A<:AbstractMatrix{UInt16}} <: AbstractMatrix{UInt16}
    storage::A
end
Base.size(image::ShiftedDetector) = size(image.storage)
Base.axes(image::ShiftedDetector) = (
    Base.IdentityUnitRange(-2:(size(image, 1) - 3)),
    Base.IdentityUnitRange(5:(size(image, 2) + 4)),
)
Base.getindex(image::ShiftedDetector, row::Int, column::Int) =
    image.storage[row + 3, column - 4]

flux_allocation_bytes(output, prepared, image) = @allocated process!(output, prepared, image)

# Independent integer oracle; include every pupil sample even when regions overlap.
function exact_flux(image, pupil_size, origins)
    row_start, column_start = first.(axes(image))
    return map(origins) do origin
        total = UInt64(0)
        for row in 1:pupil_size[1], column in 1:pupil_size[2]
            total += UInt64(image[
                row_start + origin[1] + row - 1,
                column_start + origin[2] + column - 1,
            ])
        end
        Float64(total)
    end
end

function prepared_flux(
    ;
    image_size=(4, 6),
    pupil_size=(2, 3),
    origins=[[0, 0], [0, 3], [2, 0], [2, 3]],
)
    return prepare_algorithm(
        PyramidPupilFluxU16F64,
        PyramidPupilFluxU16F64Configuration(
            image_size=image_size,
            pupil_size=pupil_size,
            pupil_origins=origins,
        ),
    )
end

@testset "raw pyramid pupil flux declaration and exact sums" begin
    prepared = prepared_flux()
    @test algorithm_label(PyramidPupilFluxU16F64) == "pyramid-pupil-flux-u16-f64"
    @test getproperty.(algorithm_formats(prepared), :schema) ==
          (RAW_DETECTOR_PIXELS_V1, PYRAMID_PUPIL_FLUX_ADU_F64_V1)
    @test getproperty.(algorithm_formats(prepared), :shape) == ((4, 6), (4,))
    @test getproperty(algorithm_ports(PyramidPupilFluxU16F64)[2], :metadata_source) === :raw

    image = reshape(UInt16.(1:24), 4, 6)
    output = zeros(Float64, 4)
    @test process!(output, prepared, image) === nothing
    @test output == Float64[33, 105, 45, 117]

    fill!(image, typemax(UInt16))
    @test process!(output, prepared, image) === nothing
    @test output == fill(6 * typemax(UInt16), 4)
end

@testset "raw pyramid pupil flux handles strided detector views" begin
    prepared = prepared_flux()
    storage = reshape(UInt16.(1:96), 8, 12)
    image = @view storage[1:2:8, 1:2:12]
    output = zeros(Float64, 4)
    @test process!(output, prepared, image) === nothing
    @test output == Float64[
        sum(image[1:2, 1:3]), sum(image[1:2, 4:6]),
        sum(image[3:4, 1:3]), sum(image[3:4, 4:6]),
    ]
end

@testset "raw pyramid pupil flux exact traversal and allocation" begin
    for pupil_size in ((1, 1), (1, 17), (17, 1), (3, 5), (17, 19), (64, 37))
        rows, columns = pupil_size
        image_size = (2rows + 5, 2columns + 7)
        origins = [[1, 2], [rows + 3, 1], [2, columns + 4], [rows + 2, columns + 3]]
        prepared = prepared_flux(; image_size, pupil_size, origins)
        storage = [UInt16(mod(104729row + 65521column, 65536))
                   for row in 1:image_size[1], column in 1:image_size[2]]
        strided_storage = zeros(UInt16, 2image_size[1], 2image_size[2])
        strided = @view strided_storage[1:2:end, 1:2:end]
        copyto!(strided, storage)
        padded_storage = zeros(UInt16, image_size[1] + 2, image_size[2] + 3)
        padded = @view padded_storage[2:(end - 1), 2:(end - 2)]
        copyto!(padded, storage)
        # The transposed parent presents row-contiguous storage at the same shape.
        row_contiguous = transpose(copy(transpose(storage)))
        generic = PermutedDimsArray(storage, (1, 2))
        @test !(generic isa StridedMatrix)
        for image in (storage, padded, strided, row_contiguous, generic)
            output_storage = fill(-1.0, 8)
            output = @view output_storage[1:2:8]
            @test process!(output, prepared, image) === nothing
            @test output == exact_flux(image, pupil_size, origins)
            @test output_storage[2:2:8] == fill(-1.0, 4)
            flux_allocation_bytes(output, prepared, image)
            @test flux_allocation_bytes(output, prepared, image) == 0
        end
        # The declared Port contract rejects offset axes before computation.
        output = fill(-1.0, 4)
        @test_throws ArgumentError process!(output, prepared, ShiftedDetector(storage))
        @test output == fill(-1.0, 4)
        fill!(storage, typemax(UInt16))
        output = zeros(4)
        @test process!(output, prepared, storage) === nothing
        @test output == fill(Float64(UInt64(rows) * UInt64(columns) * typemax(UInt16)), 4)
        fill!(storage, 0)
        @test process!(output, prepared, storage) === nothing
        @test reinterpret(UInt64, output) == zeros(UInt64, 4)
    end

    # Overlapping pupils retain independent full-region sums.
    origins = [[1, 1], [1, 1], [2, 2], [2, 2]]
    prepared = prepared_flux(image_size=(7, 11), pupil_size=(3, 5), origins=origins)
    image = reshape(UInt16.(1:77), 7, 11)
    output = zeros(4)
    @test process!(output, prepared, image) === nothing
    @test output == exact_flux(image, (3, 5), origins)
end

@testset "raw pyramid pupil flux rejects unsafe geometry during preparation" begin
    maximum_pixels = Int(fld(UInt64(1) << 53, UInt64(typemax(UInt16))))
    origins = ntuple(_ -> (0, 0), 4)
    # Exercise the sum bound directly; declared Port byte extents must fit UInt32.
    @test FilterGraphAlgorithms.PyramidPupilFluxPlan((maximum_pixels, 1),
        (maximum_pixels, 1), origins) isa FilterGraphAlgorithms.PyramidPupilFluxPlan
    @test_throws ArgumentError FilterGraphAlgorithms.PyramidPupilFluxPlan(
        (maximum_pixels + 1, 1), (maximum_pixels + 1, 1), origins)
    @test_throws ArgumentError prepared_flux(image_size=(maximum_pixels, 1),
        pupil_size=(maximum_pixels, 1), origins=fill([0, 0], 4))
    @test_throws DimensionMismatch prepared_flux(origins=[[0, 0], [0, 3], [2, 0]])
    @test_throws ArgumentError prepare_algorithm(
        PyramidPupilFluxU16F64,
        PyramidPupilFluxU16F64Configuration(
            image_size=(500_000, 500_000),
            pupil_size=(500_000, 500_000),
            pupil_origins=[[0, 0], [0, 0], [0, 0], [0, 0]],
        ),
    )
    @test_throws OverflowError prepare_algorithm(
        PyramidPupilFluxU16F64,
        PyramidPupilFluxU16F64Configuration(
            image_size=(typemax(Int), 2),
            pupil_size=(typemax(Int), 2),
            pupil_origins=[[0, 0], [0, 0], [0, 0], [0, 0]],
        ),
    )
end

end
