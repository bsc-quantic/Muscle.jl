module Operations

using ..Muscle: Muscle, Tensor, Index, inds
using ..Muscle: arch, CPU, GPU

include("binary_einsum.jl")
include("tensor_qr.jl")
include("tensor_svd.jl")
include("simple_update.jl")

end
