module FilterGraphPipeWireForwardDiffExt

using FilterGraphPipeWire: BorrowedStorageArray
using ForwardDiff: Partials

# Partials stores its values in an immutable tuple, with no borrowed storage.
Base.mightalias(::BorrowedStorageArray, ::Partials) = false
Base.mightalias(::Partials, ::BorrowedStorageArray) = false

end
