function hadamard end
function hadamard! end

choose_backend_rule(::typeof(hadamard), ::Type{<:Array}, ::Type{<:Array}) = BackendBase()
choose_backend_rule(::typeof(hadamard!), ::Type{<:Array}, ::Type{<:Array}, ::Type{<:Array}) = BackendBase()

function hadamard(a::Tensor, b::Tensor)
    # `b` must be broadcastable to `a`
    ndims(a) >= ndims(b) || return hadamard(b, a)
    @argcheck inds(b) ⊆ inds(a)
    backend = choose_backend(hadamard, parent(a), parent(b))
    return hadamard(backend, a, b)
end

function hadamard!(c::Tensor, a::Tensor, b::Tensor)
    # `b` must be broadcastable to `a`
    ndims(a) >= ndims(b) || return hadamard(c, b, a)

    @argcheck inds(c) == inds(a)
    @argcheck inds(b) ⊆ inds(a)

    backend = choose_backend(hadamard!, parent(c), parent(a), parent(b))
    return hadamard!(backend, c, a, b)
end

function hadamard(::BackendBase, a::Tensor, b::Tensor)
    # `b` must be broadcastable to `a`
    ndims(a) >= ndims(b) || return hadamard(BackendBase(), b, a)

    c = copy(a)
    return hadamard!(BackendBase(), c, a, b)
end

function hadamard!(::BackendBase, c::Tensor, a::Tensor, b::Tensor)
    # `b` must be broadcastable to `a`
    ndims(a) >= ndims(b) || return hadamard!(BackendBase(), c, b, a)

    @argcheck inds(c) == inds(a)

    # check if this is just a tensor-scalar multiplication
    if ndims(b) == 0
        c .= a .* b
        return c
    end

    data_a = parent(a)
    data_b = parent(b)
    data_c = parent(c)

    # compute the broadcast shape for `b`
    shape_b_bcast = ones(Int, ndims(a))
    for (i, ind) in enumerate(inds(a))
        if ind ∈ inds(b)
            shape_b_bcast[i] = size(b, ind)
        end
    end

    # if `b` is not a vector, it may need permutation to have the inds in the same order as `a`
    if ndims(b) > 1
        perm = findperm(inds(b), filter(∈(inds(b)), inds(a)))
        data_b = permutedims(data_b, perm)
    end

    # broadcast element-wise multiplication (Hadamard product)
    data_b = reshape(data_b, Tuple(shape_b_bcast))
    data_c .= data_a .* data_b

    return c
end
