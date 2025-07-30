using LinearAlgebra: LinearAlgebra
using ..Muscle: factorinds

# TODO implement low-rank approximations (truncated SVD, reduced SVD...)

"""
    Muscle.tensor_svd_thin(tensor::Tensor; inds_u, inds_v, ind_s, kwargs...)

Perform SVD factorization on a tensor. Either `inds_u` or `inds_v` must be specified.

# Keyword arguments

  - `inds_u`: left / U indices to be used in the SVD factorization, except for `ind_s`.
  - `inds_v`: right / right indices to be used in the SVD factorization, except for `ind_s`.
  - `ind_s`: name of the virtual bond.
  - `inplace`: If `true`, it will use `A` as workspace variable to save space. Defaults to `false`.
  - `kwargs...`: additional keyword arguments to be passed to `LinearAlgebra.svd`.
"""
function tensor_svd_thin end
function tensor_svd_thin! end

choose_backend_rule(::typeof(tensor_svd_thin), ::DomainHost) = BackendBase()
choose_backend_rule(::typeof(tensor_svd_thin), ::DomainCUDA) = BackendCuTensorNet()

choose_backend_rule(::typeof(tensor_svd_thin!), ::Vararg{DomainHost,4}) = BackendBase()
choose_backend_rule(::typeof(tensor_svd_thin!), ::Vararg{DomainCUDA,4}) = BackendCuTensorNet()

# function allocate_result(::typeof(tensor_svd_thin), A; inds_u=(), inds_v=(), ind_s=Index(gensym(:s)), kwargs...)
#     inds_u, inds_v = factorinds(inds(A), inds_u, inds_v)
#     left_extent = prod(Base.Fix1(size, A), inds_u)
#     right_extent = prod(Base.Fix1(size, A), inds_v)
#     s_extent = min(left_extent, right_extent)

#     U = Tensor(similar(parent(A), left_extent..., s_extent), [inds_u..., ind_s])
#     s = Tensor(similar(parent(A), s_extent), [ind_s])
#     V = Tensor(similar(parent(A), right_extent..., s_extent), [inds_v..., ind_s])

#     return U, s, V
# end

# function tensor_svd_thin!(A::Tensor, U::Tensor, s::Tensor, V::Tensor; kwargs...)
#     @argcheck arch(A) == arch(U) == arch(s) == arch(V)
#     tensor_svd_thin!(arch(A), A, U, s, V; kwargs...)
# end

function tensor_svd_thin(A::Tensor; inds_u=(), inds_v=(), ind_s=Index(gensym(:svd)), inplace=false, kwargs...)
    backend = choose_backend(tensor_svd_thin, A)
    return tensor_svd_thin(backend, A; inds_u, inds_v, ind_s, inplace, kwargs...)
end

function tensor_svd_thin(::Backend, A; kwargs...)
    throw(ArgumentError("`tensor_svd_thin` not implemented or not loaded for backend $(typeof(A))"))
end

function tensor_svd_thin!(Q::Tensor, R::Tensor, A::Tensor; kwargs...)
    backend = choose_backend(tensor_svd_thin!, Q, R, A)
    return tensor_svd_thin!(backend, Q, R, A; kwargs...)
end

function tensor_svd_thin!(::Backend, args...; kwargs...)
    throw(ArgumentError("`tensor_svd_thin!` not implemented or not loaded for backend $(typeof(A))"))
end

## `Base`
function tensor_svd_thin(
    ::BackendBase, A::Tensor; inds_u=(), inds_v=(), ind_s=Index(gensym(:vind)), inplace=false, kwargs...
)
    inds_u, inds_v = factorinds(inds(A), inds_u, inds_v)
    @argcheck isdisjoint(inds_u, inds_v)
    @argcheck issetequal(inds_u ∪ inds_v, inds(A))
    @argcheck ind_s ∉ inds(A)

    # permute array
    left_sizes = map(Base.Fix1(size, A), inds_u)
    right_sizes = map(Base.Fix1(size, A), inds_v)
    Amat = permutedims(A, [inds_u..., inds_v...])
    Amat = reshape(parent(Amat), prod(left_sizes), prod(right_sizes))

    # compute SVD
    U, s, V = if inplace
        LinearAlgebra.svd!(Amat; kwargs...)
    else
        LinearAlgebra.svd(Amat; kwargs...)
    end

    # tensorify results
    U = Tensor(reshape(U, left_sizes..., size(U, 2)), [inds_u; ind_s])
    s = Tensor(s, [ind_s])
    Vt = Tensor(reshape(conj(V), right_sizes..., size(V, 2)), [inds_v; ind_s])

    return U, s, Vt
end

function tensor_svd_thin!(::BackendBase, U::Tensor, s::Tensor, V::Tensor, A::Tensor; kwargs...)
    @warn "tensor_svd_thin! on BackendBase does intermediate copying. Consider using `tensor_svd_thin`."

    tmp_U, tmp_s, tmp_V = tensor_svd_thin(
        BackendBase(), A; inds_u=inds(U), inds_v=inds(V), ind_s=only(inds(s)), kwargs...
    )

    @argcheck arch(tmp_U) == arch(U)
    @argcheck arch(tmp_s) == arch(s)
    @argcheck arch(tmp_V) == arch(V)

    @argcheck inds(tmp_U) == inds(U)
    @argcheck inds(tmp_s) == inds(s)
    @argcheck inds(tmp_V) == inds(V)

    @argcheck size(tmp_U) == size(U)
    @argcheck size(tmp_s) == size(s)
    @argcheck size(tmp_V) == size(V)

    copyto!(U, tmp_U)
    copyto!(s, tmp_s)
    copyto!(V, tmp_V)

    return U, s, V
end


# TODO backend 
""" Truncate SVD. With these defaults, it could be used as inplace replacement for tensor_svd_thin """ 
function tensor_svd_trunc(A::Tensor; inds_u=(), inds_v=(), ind_s=Index(gensym(:vind)), inplace=false, cutoff=nothing, maxdim=nothing, kwargs...
)
    inds_u, inds_v = factorinds(inds(A), inds_u, inds_v)
    @argcheck isdisjoint(inds_u, inds_v)
    @argcheck issetequal(inds_u ∪ inds_v, inds(A))
    @argcheck ind_s ∉ inds(A)

    # permute array
    left_sizes = map(Base.Fix1(size, A), inds_u)
    right_sizes = map(Base.Fix1(size, A), inds_v)
    Amat = permutedims(A, [inds_u..., inds_v...])
    Amat = reshape(parent(Amat), prod(left_sizes), prod(right_sizes))

    # compute SVD
    U, s, V = if inplace
        LinearAlgebra.svd!(Amat; kwargs...)
    else
        LinearAlgebra.svd(Amat; kwargs...)
    end

    # cutoff is relative cutoff 
    k = isnothing(cutoff) ? nothing : findfirst(sv -> sv < cutoff, s/norm(s))
    k = isnothing(k) ? length(s) : k

    # keep at most maxdim SV
    k = isnothing(maxdim) ? k : min(k, maxdim)

    if k < length(s)
        # TODO do we want to use views here ? 
        @views begin
        U = U[:, 1:k]
        s = s[1:k]
        V = V[:, 1:k]
        end
    end
    # tensorify results
    U = Tensor(reshape(U, left_sizes..., k), [inds_u; ind_s])
    s = Tensor(s, [ind_s])
    Vt = Tensor(reshape(conj(V), right_sizes..., k), [inds_v; ind_s])

    return U, s, Vt
end


# other version trying to reuse tensor_svd_thin but prob not worth it
function tensor_svd_trunc_alt(A::Tensor; inds_u=(), inds_v=(), ind_s=Index(gensym(:vind)), inplace=false, cutoff, maxdim, kwargs...
)
    inds_u, inds_v = factorinds(inds(A), inds_u, inds_v)

    left_sizes = map(Base.Fix1(size, A), inds_u)
    right_sizes = map(Base.Fix1(size, A), inds_v)

    # TODO I don't know how to pass the backend here 
    U, s, Vt = tensor_svd_thin(A; inds_u, inds_v, ind_s, inplace, kwargs...)

    keep = findall(sv -> sv ≥ cutoff, s)
    # Keep at most maxsv largest valid singular values
    k = min(length(keep), maxdim)
    keep_inds = keep[1:k]

    U = U[:, keep_inds]
    s = s[keep_inds]
    Vt = Vt[keep_inds, :]

    # tensorify results
    U = Tensor(reshape(U, left_sizes..., size(U, 2)), [inds_u; ind_s])
    s = Tensor(s, [ind_s])
    Vt = Tensor(reshape(Vt, size(s,1), right_sizes...), [ind_s; inds_v])

    return U, s, Vt
end

