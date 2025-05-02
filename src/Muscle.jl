module Muscle

import EinExprs: inds
using Compat: @compat

include("Utils/Utils.jl")

include("Index.jl")
export Index

include("Tensor.jl")
export Tensor

include("Architecture.jl")

include("Operations/unary_einsum.jl")
export unary_einsum, unary_einsum!

include("Operations/binary_einsum.jl")
export binary_einsum, binary_einsum!

include("Operations/tensor_qr.jl")
export tensor_qr_thin, tensor_qr_thin!

include("Operations/tensor_svd.jl")
export tensor_svd_thin, tensor_svd_thin!

include("Operations/simple_update.jl")
export simple_update, simple_update!

# rexports from EinExprs
export inds

end
