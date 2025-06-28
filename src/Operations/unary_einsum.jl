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

function unary_einsum(x::Tensor; dims=nonunique(inds(x)), out=nothing)
    inds_sum = ∩(dims, inds(x))
    inds_y = if isnothing(out)
        setdiff(inds(x), inds_sum isa Base.AbstractVecOrTuple ? inds_sum : [inds_sum])
    else
        out
    end

    backend = choose_backend(unary_einsum, parent(x))
    return unary_einsum(backend, inds_y, x)
end

function unary_einsum!(y::Tensor, x::Tensor)
    backend = choose_backend(unary_einsum!, parent(y), parent(x))
    unary_einsum!(backend, y, x)
    return y
end
