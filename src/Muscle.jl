module Muscle

import EinExprs: inds
using Compat: @compat

include("Utils/Utils.jl")

include("Index.jl")
export Index

include("Tensor.jl")
export Tensor

include("Architecture.jl")

include("Operations/Operations.jl")

# rexports from EinExprs
export inds

end
