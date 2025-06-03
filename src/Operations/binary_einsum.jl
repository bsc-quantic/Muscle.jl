using Adapt
using OMEinsum: OMEinsum
using CUDA
using cuTENSOR: cuTENSOR

"""
    binary_einsum(a::Tensor, b::Tensor; dims=∩(inds(a), inds(b)), out=nothing)

Perform a binary tensor contraction operation.

# Keyword arguments

    - `dims`: indices to contract over. Defaults to the set intersection of the indices of `a` and `b`.
    - `out`: indices of the output tensor. Defaults to the set difference of the indices of `a` and `b`.
"""
function binary_einsum end

"""
    binary_einsum!(c::Tensor, a::Tensor, b::Tensor)

Perform a binary tensor contraction operation between `a` and `b` and store the result in `c`.
"""
function binary_einsum! end

# TODO add a preference system for some backends
choose_backend_rule(::typeof(binary_einsum), ::Type{<:Array}, ::Type{<:Array}) = BackendOMEinsum()
choose_backend_rule(::typeof(binary_einsum), ::Type{<:CuArray}, ::Type{<:CuArray}) = BackendCuTENSOR()
choose_backend_rule(::typeof(binary_einsum!), ::Type{<:Array}, ::Type{<:Array}, ::Type{<:Array}) = BackendOMEinsum()
choose_backend_rule(::typeof(binary_einsum!), ::Type{<:CuArray}, ::Type{<:CuArray}) = BackendCuTENSOR()
function choose_backend_rule(::typeof(binary_einsum!), ::Type{<:CuArray}, ::Type{<:CuArray}, ::Type{<:CuArray})
    BackendCuTENSOR()
end

function binary_einsum(a::Tensor, b::Tensor; dims=(∩(inds(a), inds(b))), out=nothing)
    inds_sum = ∩(dims, inds(a), inds(b))

    inds_c = if isnothing(out)
        setdiff(inds(a) ∪ inds(b), inds_sum isa Base.AbstractVecOrTuple ? inds_sum : [inds_sum])
    else
        out
    end

    data_a = parent(a)
    data_b = parent(b)
    backend = choose_backend(binary_einsum, data_a, data_b)
    # if ismissing(backend)
    #     @warn "No backend found for binary_einsum(::$(typeof(a)), ::$(typeof(b))), so unwrapping data"
    #     data_a = collect(data_a)
    #     data_b = collect(data_b)
    #     backend = choose_backend(binary_einsum, data_a, data_b)
    # end

    data_c = binary_einsum(backend, inds_c, data_a, inds(a), data_b, inds(b))
    return Tensor(data_c, inds_c)
end

function binary_einsum!(c::Tensor, a::Tensor, b::Tensor)
    data_c = parent(c)
    data_a = parent(a)
    data_b = parent(b)
    backend = choose_backend(binary_einsum!, data_c, data_a, data_b)
    if ismissing(backend)
        data_a = collect(data_a)
        data_b = collect(data_b)
        backend = choose_backend(binary_einsum, data_a, data_b)
    end

    binary_einsum!(backend, data_c, inds(c), data_a, inds(a), data_b, inds(b))
    return c
end

# backend implementations
## `OMEinsum`
function binary_einsum(::BackendOMEinsum, inds_c, a, inds_a, b, inds_b)
    size_dict = Dict{Index,Int}()
    for (ind, ind_size) in Iterators.flatten([inds_a .=> size(a), inds_b .=> size(b)])
        size_dict[ind] = ind_size
    end

    c = OMEinsum.get_output_array((a, b), Int[size_dict[i] for i in inds_c], false)
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
