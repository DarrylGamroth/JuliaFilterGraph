@kernel function _select_and_chop_pixels!(difference, validity, pair_status, history,
        history_validity, pixels, pixel_validity, image, indices, flags, status)
    i = @index(Global, Linear)
    @inbounds if iszero(status[1])
        current = image[indices[i]]
        valid = isfinite(current)
        pixels[i] = current
        pixel_validity[i] = valid
        _store_chopped_pixel!(i, difference, validity, pair_status, history,
            history_validity, current, valid, flags)
    end
end

# The fused chop writes both earlier nodes' outputs. Their logical positions
# still participate in graph metadata, diagnostics, and publication.
struct FusedSccPixelPreparation end
enqueue!(::FusedSccPixelPreparation) = nothing
preflight(::FusedSccPixelPreparation) = nothing
complete!(::FusedSccPixelPreparation, ::Int32) = nothing
reset!(adapter::FusedSccPixelPreparation) = adapter

function _fuse_scc_pixel_preparation(selection::CapturedKernel, validity::CapturedKernel,
        chop::CapturedChoppedImage)
    pixels, image, indices, selection_status = selection.arguments
    pixel_validity, validity_input, validity_status = validity.arguments
    difference, valid_pairs, pair_status, history, history_validity,
        chop_input, chop_validity, flags, status = chop.kernel.arguments
    pixels === validity_input === chop_input && pixel_validity === chop_validity &&
        selection_status === validity_status === status || return nothing
    selection.extent == validity.extent == chop.kernel.extent || return nothing

    # Each lane may read and write in a different order after fusion. Keep all
    # written buffers disjoint from each other and from the image gather.
    written = (pixels, pixel_validity, difference, valid_pairs, pair_status, history, history_validity)
    read_only = (image, indices, flags, status)
    for (i, output) in pairs(written)
        any(input -> _capture_arrays_overlap(output, input), read_only) && return nothing
        any(j -> _capture_arrays_overlap(output, written[j]), 1:(i-1)) && return nothing
    end

    backend = KernelAbstractions.get_backend(pixels)
    fused = CapturedKernel(_select_and_chop_pixels!(backend, 256),
        (difference, valid_pairs, pair_status, history, history_validity,
            pixels, pixel_validity, image, indices, flags, status), selection.extent)
    return CapturedChoppedImage(fused, chop.flags, chop.plan, chop.workspace)
end
