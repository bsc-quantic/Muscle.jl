module Muscle

import EinExprs: inds
using Compat: @compat

include("Utils/Utils.jl")

include("Index.jl")
export Index

include("Architecture.jl")

include("Tensor.jl")
export Tensor

include("Einsum.jl")
export unary_einsum, unary_einsum!, binary_einsum, binary_einsum!

include("Operations/Operations.jl")

# rexports from EinExprs
export inds

end
