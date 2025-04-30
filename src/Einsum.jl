using ArgCheck
using LinearAlgebra
using OMEinsum: OMEinsum

function unary_einsum end
function unary_einsum! end

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
    Operations.binary_einsum!(arch(c), parent(c), inds(c), parent(a), inds(a), parent(b), inds(b); kwargs...)
    return c
end
