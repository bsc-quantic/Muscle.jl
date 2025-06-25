module MuscleCUDAExt

using CUDA
using cuTENSOR
using Muscle
import Muscle: binary_einsum, binary_einsum!, BackendCuTENSOR

memory_space(::Type{<:CuArray}) = CUDAMemorySpace()

choose_backend_rule(::typeof(binary_einsum), ::Type{<:CuArray}, ::Type{<:CuArray}) = BackendCuTENSOR()
choose_backend_rule(::typeof(binary_einsum!), ::Type{<:CuArray}, ::Type{<:CuArray}) = BackendCuTENSOR()
function choose_backend_rule(::typeof(binary_einsum!), ::Type{<:CuArray}, ::Type{<:CuArray}, ::Type{<:CuArray})
    BackendCuTENSOR()
end

choose_backend_rule(::typeof(unary_einsum), ::Type{<:CuArray}) = BackendOMEinsum()
choose_backend_rule(::typeof(unary_einsum!), ::Type{<:CuArray}, ::Type{<:CuArray}) = BackendOMEinsum()

choose_backend_rule(::typeof(tensor_qr_thin), ::Type{<:CuArray}) = BackendCuTensorNet()
function choose_backend_rule(::typeof(tensor_qr_thin!), ::Type{<:CuArray}, ::Type{<:CuArray}, ::Type{<:CuArray})
    BackendCuTensorNet()
end

function choose_backend_rule(::typeof(simple_update), ::Type{<:CuArray}, ::Type{<:CuArray}, ::Type{<:CuArray})
    BackendCuTensorNet()
end

choose_backend_rule(::typeof(tensor_svd_thin), ::Type{<:CuArray}) = BackendCuTensorNet()
function choose_backend_rule(
    ::typeof(tensor_svd_thin!), ::Type{<:CuArray}, ::Type{<:CuArray}, ::Type{<:CuArray}, ::Type{<:CuArray}
)
    BackendCuTensorNet()
end




## `CUDA` (uses cuTENSOR)
function binary_einsum(::BackendCuTENSOR, inds_c, a, inds_a, b, inds_b; kwargs...)
    size_dict = Dict{Index,Int}()
    for (ind, ind_size) in Iterators.flatten([inds_a .=> size(a), inds_b .=> size(b)])
        size_dict[ind] = ind_size
    end

    T = Base.promote_eltype(a, b)
    c = similar(a, T, Tuple(size_dict[i] for i in inds_c))
    binary_einsum!(BackendCuTENSOR(), c, inds_c, a, inds_a, b, inds_b; kwargs...)
    return c
end

function binary_einsum!(::BackendCuTENSOR, c, inds_c, a, inds_a, b, inds_b; kwargs...)
    # translate indices to mode numbers
    indmap = Dict{Index,Int}(ind => i for (i, ind) in enumerate(unique(inds_a ∪ inds_b)))
    inds_a = [indmap[ind] for ind in inds_a]
    inds_b = [indmap[ind] for ind in inds_b]
    inds_c = [indmap[ind] for ind in inds_c]

    # call cuTENSOR: op_out(C) := α * opA(A) * opB(B) + β * opC(C)
    α = 1
    β = 0
    op_a = cuTENSOR.OP_IDENTITY
    op_b = cuTENSOR.OP_IDENTITY
    op_c = cuTENSOR.OP_IDENTITY
    op_out = cuTENSOR.OP_IDENTITY
    cuTENSOR.contract!(
        α, parent(a), inds_a, op_a, parent(b), inds_b, op_b, β, parent(c), inds_c, op_c, op_out; kwargs...
    )

    return c
end

end
