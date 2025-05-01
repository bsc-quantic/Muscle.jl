using ArgCheck
using OMEinsum: OMEinsum
using CUDA
using cuTENSOR: cuTENSOR

function binary_einsum end
function binary_einsum! end

function binary_einsum(a::Tensor, b::Tensor; kwargs...)
    a, b = promote_memspace(a, b)
    c = allocate_result(binary_einsum, a, b; kwargs...)
    binary_einsum!(c, a, b; kwargs...)
    return c
end

# TODO dispatch based on memory-space
function allocate_result(
    ::typeof(binary_einsum), a::Tensor, b::Tensor; fillzero=false, dims=(∩(inds(a), inds(b))), out=nothing
)
    ia = collect(inds(a))
    ib = collect(inds(b))
    i = ∩(dims, ia, ib)

    ic = if isnothing(out)
        setdiff(ia ∪ ib, i isa Base.AbstractVecOrTuple ? i : [i])
    else
        out
    end

    data = OMEinsum.get_output_array((parent(a), parent(b)), Int[size(i in ia ? a : b, i) for i in ic]; fillzero)
    return Tensor(data, ic)
end

function binary_einsum!(c::Tensor, a::Tensor, b::Tensor; kwargs...)
    @argcheck arch(c) == arch(a) == arch(b)
    Muscle.binary_einsum!(arch(c), parent(c), inds(c), parent(a), inds(a), parent(b), inds(b); kwargs...)
    return c
end

function binary_einsum!(
    ::CPU,
    c::AbstractArray,
    ic,
    a::AbstractArray,
    ia,
    b::AbstractArray,
    ib;
    kwargs...
)
    ixs = (ia, ib)
    iy = ic
    xs = (a, b)
    y = c

    size_dict = Dict{Index,Int}()
    for (ind, ind_size) in Iterators.flatten([ia .=> size(a), ib .=> size(b)])
        size_dict[ind] = ind_size
    end

    OMEinsum.einsum!(ixs, iy, xs, y, true, false, size_dict)
    return c
end

function binary_einsum!(
    ::GPU,
    c::AbstractArray,
    ic,
    a::AbstractArray,
    ia,
    b::AbstractArray,
    ib;
    kwargs...
)
    # translate indices to mode numbers
    indmap = Dict{Index,Int}(ind => i for (i, ind) in enumerate(unique(ia ∪ ib)))
    ia = [indmap[ind] for ind in ia]
    ib = [indmap[ind] for ind in ib]
    ic = [indmap[ind] for ind in ic]

    # call cuTENSOR: op_out(C) := α * opA(A) * opB(B) + β * opC(C)
    α = 1
    β = 0
    op_a = cuTENSOR.OP_IDENTITY
    op_b = cuTENSOR.OP_IDENTITY
    op_c = cuTENSOR.OP_IDENTITY
    op_out = cuTENSOR.OP_IDENTITY
    cuTENSOR.contract!(α, parent(a), ia, op_a, parent(b), ib, op_b, β, parent(c), ic, op_c, op_out; kwargs...)

    return c
end
