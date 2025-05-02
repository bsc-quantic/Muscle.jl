using OMEinsum: OMEinsum
using CUDA
using cuTENSOR: cuTENSOR

function binary_einsum end
function binary_einsum! end

# TODO add a preference system for some backends
choose_backend(::typeof(binary_einsum), A::Array, B::Array) = BackendOMEinsum()
choose_backend(::typeof(binary_einsum), A::CuArray, B::CuArray) = BackendCUDA()

choose_backend(::typeof(binary_einsum!), C::Array, A::Array, B::Array) = BackendOMEinsum()
choose_backend(::typeof(binary_einsum!), C::CuArray, A::CuArray, B::CuArray) = BackendCUDA()

function binary_einsum(a::Tensor, b::Tensor; dims=(∩(inds(a), inds(b))), out=nothing)
    backend = choose_backend(binary_einsum, parent(a), parent(b))

    inds_sum = ∩(dims, inds(a), inds(b))

    inds_c = if isnothing(out)
        # setdiff(inds(a) ∪ inds(b), inds_sum isa Base.AbstractVecOrTuple ? inds_sum : [inds_sum])
        setdiff(inds(a) ∪ inds(b), inds_sum)
    else
        out
    end

    data_c = binary_einsum(backend, inds_c, parent(a), inds(a), parent(b), inds(b))
    return Tensor(data_c, inds_c)
end

function binary_einsum!(c::Tensor, a::Tensor, b::Tensor)
    backend = choose_backend(binary_einsum!, parent(c), parent(a), parent(b))
    binary_einsum!(backend, parent(c), inds(c), parent(a), inds(a), parent(b), inds(b))
    return c
end

# backend implementations
## `OMEinsum`
function binary_einsum(::BackendOMEinsum, inds_c, a, inds_a, b, inds_b)
    size_dict = Dict{Index,Int}()
    for (ind, ind_size) in Iterators.flatten([inds_a .=> size(a), inds_b .=> size(b)])
        size_dict[ind] = ind_size
    end

    c = OMEinsum.get_output_array((a, b), Int[size_dict[i] for i in inds_c]; fillzero=false)
    OMEinsum.einsum!((inds_a, inds_b), inds_c, (a, b), c, true, false, size_dict)
    return c
end

function binary_einsum!(::BackendOMEinsum, c, inds_c, a, inds_a, b, inds_b)
    size_dict = Dict{Index,Int}()
    for (ind, ind_size) in Iterators.flatten([inds_a .=> size(a), inds_b .=> size(b)])
        size_dict[ind] = ind_size
    end

    OMEinsum.einsum!((inds_a, inds_b), inds_c, (a, b), c, true, false, size_dict)
    return c
end

## `CUDA` (uses cuTENSOR)
function binary_einsum(::BackendCUDA, inds_c, a, inds_a, b, inds_b; kwargs...)
    size_dict = Dict{Index,Int}()
    for (ind, ind_size) in Iterators.flatten([inds_a .=> size(a), inds_b .=> size(b)])
        size_dict[ind] = ind_size
    end

    T = Base.promote_eltype(a, b)
    c = similar(a, T, Int[size_dict[i] for i in inds_c])
    binary_einsum!(BackendCUDA(), c, inds_c, a, inds_a, b, inds_b; kwargs...)
    return c
end

function binary_einsum!(::BackendCUDA, c, inds_c, a, inds_a, b, inds_b; kwargs...)
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
    cuTENSOR.contract!(α, parent(a), inds_a, op_a, parent(b), inds_b, op_b, β, parent(c), inds_c, op_c, op_out; kwargs...)

    return c
end
