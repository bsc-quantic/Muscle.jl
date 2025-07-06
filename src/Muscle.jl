module Muscle

import EinExprs: inds

include("Utils/Utils.jl")

include("Index.jl")
export Index

include("Tensor.jl")
export Tensor

include("Domain.jl")
include("Backend.jl")

include("Operations/hadamard.jl")
export hadamard, hadamard!

include("Operations/unary_einsum.jl")
export unary_einsum, unary_einsum!

include("Operations/binary_einsum.jl")
export binary_einsum, binary_einsum!

include("Operations/tensor_qr.jl")
export tensor_qr_thin, tensor_qr_thin!

include("Operations/tensor_svd.jl")
export tensor_svd_thin, tensor_svd_thin!

include("Operations/tensor_eigen.jl")
export tensor_eigen_thin, tensor_eigen_thin!
export tensor_bieigen_thin, tensor_bieigen_thin!

include("Operations/simple_update.jl")
export simple_update, simple_update!

# rexports from EinExprs
export inds

# precompilation
using PrecompileTools

@setup_workload begin
    @compile_workload begin
        # `binary_einsum` instances
        for (Ta, Tb) in [(Float64, Float64), (ComplexF64, ComplexF64), (Float64, ComplexF64), (ComplexF64, Float64)]
            a = Tensor(ones(Ta, 2, 2), [:i, :j])
            b = Tensor(ones(Tb, 2, 2), [:j, :k])
            Muscle.binary_einsum(a, b)

            a = Tensor(ones(Ta, 2, 2, 2), [:i, :j, :k])
            b = Tensor(ones(Tb, 2, 2, 2), [:j, :k, :l])
            Muscle.binary_einsum(a, b)
        end
    end
end

end
