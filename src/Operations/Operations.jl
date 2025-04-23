module Operations

using ..Muscle: Tensor, Index, inds
using ..Muscle: arch, CPU, GPU

include("BinaryEinsum.jl")
include("SVD.jl")
include("SimpleUpdate.jl")

end
