# CUDA retains its pupil-major storage layout and native warp reduction.
# Detector indexing, certificates and state adoption are shared.
include("pyramid_pupil_mean.jl")

# Qualified internal representation: logical (4,N), dense parent (N,4).
# Only the unexported pupil→reconstructor edge opts into this layout.
const _PupilMajorPixels = PermutedDimsArray{Float32,2,(2,1),(2,1),<:CUDA.CuArray}

function _require_pupil_array(name, array::_PupilMajorPixels, device::CUDA.CuDevice)
    return _require_capture_array(name, parent(array), device)
end

_pupil_pixel_storage(output::_PupilMajorPixels) = parent(output)

_pupil_selected_indices(output::_PupilMajorPixels, indices) =
    vec(permutedims(reshape(indices, size(output))))

struct _PupilMajorOrder end
(::_PupilMajorOrder)(output, status, input; ndrange) = nothing

_pupil_order_kernel(backend, input::_PupilMajorPixels) = _PupilMajorOrder()

_pupil_order_buffer(device, input::_PupilMajorPixels) = reshape(parent(input), length(input))
