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

choose_backend_rule(::typeof(unary_einsum), ::DomainHost) = BackendOMEinsum()
choose_backend_rule(::typeof(unary_einsum), ::DomainCUDA) = BackendOMEinsum()

choose_backend_rule(::typeof(unary_einsum!), ::DomainHost, ::DomainHost) = BackendOMEinsum()
choose_backend_rule(::typeof(unary_einsum!), ::DomainCUDA, ::DomainCUDA) = BackendOMEinsum()

function unary_einsum(x::Tensor; dims=nonunique(inds(x)), out=nothing)
    inds_sum = ∩(dims, inds(x))
    inds_y = if isnothing(out)
        setdiff(inds(x), inds_sum isa Base.AbstractVecOrTuple ? inds_sum : [inds_sum])
    else
        out
    end

    backend = choose_backend(unary_einsum, x)
    return unary_einsum(backend, inds_y, x)
end

function unary_einsum(::Backend, x; kwargs...)
    throw(ArgumentError("`unary_einsum` not implemented or not loaded for backend $(typeof(x))"))
end

function unary_einsum!(y::Tensor, x::Tensor)
    backend = choose_backend(unary_einsum!, y, x)
    unary_einsum!(backend, y, x)
    return y
end

function unary_einsum(::Backend, y, x; kwargs...)
    throw(ArgumentError("`unary_einsum!` not implemented or not loaded for backend $(typeof(x))"))
end
