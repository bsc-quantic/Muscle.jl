using OMEinsum
using CUDA

"""
    unary_einsum(a::Tensor; dims=∩(inds(a), inds(b)), out=nothing)

Perform a unary tensor contraction operation.

# Keyword arguments

    - `dims`: indices to contract over. Defaults to the repeated indices.
    - `out`: indices of the output tensor. Defaults to the unique indices.
"""
function unary_einsum end

"""
    unary_einsum!(c::Tensor, a::Tensor)

Perform a unary tensor contraction operation on `a` and store the result in `c`.
"""
function unary_einsum! end

choose_backend_rule(::typeof(unary_einsum), ::Type{<:Array}) = BackendOMEinsum()
choose_backend_rule(::typeof(unary_einsum), ::Type{<:CuArray}) = BackendOMEinsum()
choose_backend_rule(::typeof(unary_einsum!), ::Type{<:Array}, ::Type{<:Array}) =
    BackendOMEinsum()
choose_backend_rule(::typeof(unary_einsum!), ::Type{<:CuArray}, ::Type{<:CuArray}) =
    BackendOMEinsum()

function unary_einsum(x::Tensor; dims = nonunique(inds(x)), out = nothing)
    inds_sum = ∩(dims, inds(x))
    inds_y = if isnothing(out)
        setdiff(inds(x), inds_sum isa Base.AbstractVecOrTuple ? inds_sum : [inds_sum])
    else
        out
    end

    backend = choose_backend(unary_einsum, parent(x))
    data_y = unary_einsum(backend, inds_y, parent(x), inds(x))
    return Tensor(data_y, inds_y)
end

function unary_einsum!(y::Tensor, x::Tensor)
    backend = choose_backend(unary_einsum!, parent(y), parent(x))
    unary_einsum!(backend, parent(y), inds(y), parent(x), inds(x))
    return y
end

# implementations
## `OMEinsum`
function unary_einsum(::BackendOMEinsum, inds_y, x, inds_x)
    size_dict = Dict(inds_x .=> size(x))
    y = similar(x, Tuple(size_dict[i] for i in inds_y))
    unary_einsum!(BackendOMEinsum(), y, inds_y, x, inds_x)
    return y
end

function unary_einsum!(::BackendOMEinsum, y, inds_y, x, inds_x)
    size_dict = Dict(inds_x .=> size(x))
    einsum!((inds_x,), inds_y, (x,), y, true, false, size_dict)
    return y
end
