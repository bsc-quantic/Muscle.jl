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

choose_backend_rule(::typeof(binary_einsum), ::DomainHost, ::DomainHost) = BackendBase()
choose_backend_rule(::typeof(binary_einsum), ::DomainCUDA, ::DomainCUDA) = BackendCuTENSOR()
choose_backend_rule(::typeof(binary_einsum), ::DomainReactant, ::DomainReactant) = BackendReactant()
choose_backend_rule(::typeof(binary_einsum), ::DomainReactant, ::DomainHost) = BackendReactant()
choose_backend_rule(::typeof(binary_einsum), ::DomainHost, ::DomainReactant) = BackendReactant()
choose_backend_rule(::typeof(binary_einsum), ::DomainDagger, ::DomainDagger) = BackendDagger()
choose_backend_rule(::typeof(binary_einsum), ::DomainHost, ::DomainDagger) = BackendDagger()
choose_backend_rule(::typeof(binary_einsum), ::DomainDagger, ::DomainHost) = BackendDagger()

choose_backend_rule(::typeof(binary_einsum!), ::DomainHost, ::DomainHost, ::DomainHost) = BackendBase()
choose_backend_rule(::typeof(binary_einsum!), ::DomainCUDA, ::DomainCUDA, ::DomainCUDA) = BackendCuTENSOR()
choose_backend_rule(::typeof(binary_einsum!), ::DomainReactant, ::DomainReactant, ::DomainReactant) = BackendReactant()

function binary_einsum(a::Tensor, b::Tensor; dims=(∩(inds(a), inds(b))), out=nothing)

    reorder_inds = false
    inds_sum = ∩(dims, inds(a), inds(b))

    inds_c = if isnothing(out)
        setdiff(inds(a) ∪ inds(b), inds_sum isa Base.AbstractVecOrTuple ? inds_sum : [inds_sum])
    else
        reorder_inds = true
        out
    end
 
    backend = choose_backend(binary_einsum, parent(a), parent(b))
    # if ismissing(backend)
    #     @warn "No backend found for binary_einsum(::$(typeof(a)), ::$(typeof(b))), so unwrapping data"
    #     data_a = collect(data_a)
    #     data_b = collect(data_b)
    #     backend = choose_backend(binary_einsum, data_a, data_b)
    # end

    c = if backend == Muscle.BackendBase()
            binary_einsum(backend, inds_c, a, b; reorder_inds)
        else
            binary_einsum(backend, inds_c, a, b)
        end

    return c

end

function binary_einsum(::Backend, inds_c, a, b)
    throw(ArgumentError("`binary_einsum` not implemented or not loaded for backend $(typeof(a))"))
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

    binary_einsum!(backend, c, a, b)
    return c
end

function binary_einsum!(::Backend, c, a, b)
    throw(ArgumentError("`binary_einsum!` not implemented or not loaded for backend $(typeof(a))"))
end

function binary_einsum(::BackendBase, inds_c, a::Tensor, b::Tensor; reorder_inds=true)
    inds_contract = inds(a) ∩ inds(b)
    inds_left = setdiff(inds(a), inds_contract)
    inds_right = setdiff(inds(b), inds_contract)

    # can't deal with hyperindices
    @argcheck isdisjoint(inds_c, inds_contract) "`BackendBase` can't deal with hyperindices. Load OMEinsum and use `BackendOMEinsum` instead."
    @argcheck issetequal(inds_c, symdiff(inds(a), inds(b))) "`BackendBase` can't deal with hyperindices. Load OMEinsum and use `BackendOMEinsum` instead."

    sizes_left = map(Base.Fix1(size, a), inds_left)
    sizes_right = map(Base.Fix1(size, b), inds_right)
    sizes_contract = map(Base.Fix1(size, a), inds_contract)

    a_mat = reshape(parent(permutedims(a, [inds_left; inds_contract])), prod(sizes_left), prod(sizes_contract))
    b_mat = reshape(parent(permutedims(b, [inds_contract; inds_right])), prod(sizes_contract), prod(sizes_right))

    c_mat = a_mat * b_mat

    c = Tensor(reshape(c_mat, sizes_left..., sizes_right...), [inds_left; inds_right])
    if reorder_inds
        c = permutedims(c, inds_c)
    end
    return c
end

function binary_einsum!(::BackendBase, c::Tensor, a::Tensor, b::Tensor)
    inds_contract = inds(a) ∩ inds(b)
    inds_left = setdiff(inds(a), inds_contract)
    inds_right = setdiff(inds(b), inds_contract)

    # can't deal with hyperindices
    @argcheck isdisjoint(inds(c), inds_contract) "`BackendBase` can't deal with hyperindices. Load OMEinsum and use `BackendOMEinsum` instead."
    @argcheck issetequal(inds(c), symdiff(inds(a), inds(b))) "`BackendBase` can't deal with hyperindices. Load OMEinsum and use `BackendOMEinsum` instead."

    # can't deal with inplace permutedims
    @argcheck inds(c) == [inds_left; inds_right]

    sizes_left = map(Base.Fix1(size, a), inds_left)
    sizes_right = map(Base.Fix1(size, b), inds_right)
    sizes_contract = prod(Base.Fix1(size, a), inds_contract)

    a_mat = reshape(parent(permutedims(a, [inds_left; inds_contract])), prod(sizes_left), prod(sizes_contract))
    b_mat = reshape(parent(permutedims(b, [inds_contract; inds_right])), prod(sizes_contract), prod(sizes_right))
    c_mat = reshape(c, prod(sizes_left), prod(sizes_right))

    LinearAlgebra.mul!(c_mat, a_mat, b_mat)

    return reshape(c_mat, sizes_left..., sizes_right...)
end
