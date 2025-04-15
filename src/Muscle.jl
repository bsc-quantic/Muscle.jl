module Muscle

import EinExprs: inds
using Compat: @compat

include("Utils/Utils.jl")

include("Index.jl")
export Index

include("Tensor.jl")
export Tensor

include("Architecture.jl")

# rexports from EinExprs
export inds

end
