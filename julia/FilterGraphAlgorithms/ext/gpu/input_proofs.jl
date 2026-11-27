# Expert-only cold refinement. The existing public Bool proof API is unchanged.
_capture_output_bound(adapter, ::Val) = nothing
_prepare_bounded_validation(adapter, inputs::Tuple) = adapter
