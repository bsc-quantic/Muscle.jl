module Operations

using ..Muscle: Muscle, Tensor, Index, inds
using ..Muscle: arch, CPU, GPU

include("binary_einsum.jl")
include("QR.jl")
include("SVD.jl")
include("SimpleUpdate.jl")

end
