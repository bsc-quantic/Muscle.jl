module MuscleCUDAExt

using Muscle
using CUDA
using cuTENSOR

Muscle.Domain(::Type{<:CuArray}) = Muscle.DomainCUDA()

## `CUDA` (uses cuTENSOR)
function Muscle.binary_einsum(::Muscle.BackendCuTENSOR, inds_c, a, inds_a, b, inds_b; kwargs...)
    size_dict = Dict{Index,Int}()
    for (ind, ind_size) in Iterators.flatten([inds_a .=> size(a), inds_b .=> size(b)])
        size_dict[ind] = ind_size
    end

    T = Base.promote_eltype(a, b)
    c = similar(a, T, Tuple(size_dict[i] for i in inds_c))
    binary_einsum!(Muscle.BackendCuTENSOR(), c, inds_c, a, inds_a, b, inds_b; kwargs...)
    return c
end

function Muscle.binary_einsum!(::Muscle.BackendCuTENSOR, c, inds_c, a, inds_a, b, inds_b; kwargs...)
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
