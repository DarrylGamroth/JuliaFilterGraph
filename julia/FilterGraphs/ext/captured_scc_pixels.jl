function _fuse_captured_scc_pixels(nodes, routes, indices=eachindex(nodes))
    extension = Base.get_extension(FGA, algorithm_extension)
    extension !== nothing && isdefined(extension, :_fuse_scc_pixel_preparation) || return nodes

    for i in 1:(length(nodes)-2)
        selection, validity, chop = nodes[i], nodes[i+1], nodes[i+2]
        selection isa FG.CapturedNode{FGA.PixelSelectionF32} &&
            validity isa FG.CapturedNode{FGA.PixelValidityF32} &&
            chop isa FG.CapturedNode{FGA.ChoppedImageDifferenceF32} || continue
        selection_index, validity_index = indices[i], indices[i+1]
        selected = FG.NodeOutput{selection_index,1}()
        valid = FG.NodeOutput{validity_index,1}()
        routes[i+1] === (selected,) && routes[i+2] === (selected, valid) || continue
        selection.adapter isa extension.CapturedKernel &&
            validity.adapter isa extension.CapturedKernel &&
            chop.adapter isa extension.CapturedChoppedImage || continue

        fused = extension._fuse_scc_pixel_preparation(selection.adapter, validity.adapter, chop.adapter)
        fused === nothing && continue
        skipped = extension.FusedSccPixelPreparation()
        result = Base.setindex(nodes, FG._captured_node_with_adapter(selection, skipped), i)
        result = Base.setindex(result, FG._captured_node_with_adapter(validity, skipped), i+1)
        # Each captured stage admits only one camera metadata consumer.
        return Base.setindex(result, FG._captured_node_with_adapter(chop, fused), i+2)
    end
    return nodes
end
