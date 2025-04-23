using LinearAlgebra: LinearAlgebra
using cuTensorNet: cuTensorNet
using ..Muscle: factorinds

"""
    Operations.tensor_svd_thin(tensor::Tensor; inds_u, inds_v, s_ind, kwargs...)

Perform SVD factorization on a tensor. Either `inds_u` or `inds_v` must be specified.

# Keyword arguments

  - `inds_u`: left / U indices to be used in the SVD factorization, except for `s_ind`.
  - `inds_v`: right / right indices to be used in the SVD factorization, except for `s_ind`.
  - `s_ind`: name of the virtual bond.
  - `maxdim`: maximum dimension of the virtual bond.
  - `inplace`: If `true`, it will use `A` as workspace variable to save space. Defaults to `false`.
  - `kwargs...`: additional keyword arguments to be passed to `LinearAlgebra.svd`.
"""
function tensor_svd_thin end
function tensor_svd_thin! end

# dispatch to correct architecture
tensor_svd_thin(A::Tensor; kwargs...) = tensor_svd_thin(arch(A), A; kwargs...)

function allocate_result(::typeof(tensor_svd_thin), A; left_inds=(), right_inds=(), virtualind=Index(Symbol(uuid4())), maxdim=nothing, kwargs...)
    left_inds, right_inds = factorinds(inds(A), left_inds, right_inds)
    left_extent = prod(Base.Fix1(size, A), left_inds)
    right_extent = prod(Base.Fix1(size, A), right_inds)
    maxdim = isnothing(maxdim) ? min(left_extent, right_extent) : maxdim

    U = Tensor(similar(parent(A), left_extent..., maxdim), [left_inds..., virtualind])
    s = Tensor(similar(parent(A), maxdim), [virtualind])
    V = Tensor(similar(parent(A), right_extent..., maxdim), [right_inds..., virtualind])

    return U, s, V
end

# function tensor_svd_thin!(A::Tensor, U::Tensor, s::Tensor, V::Tensor; kwargs...)
#     @argcheck arch(A) == arch(U) == arch(s) == arch(V)
#     tensor_svd_thin!(arch(A), A, U, s, V; kwargs...)
# end

# CPU
function tensor_svd_thin!(
    ::CPU,
    A::Tensor,
    U::Tensor,
    s::Tensor,
    V::Tensor;
    kwargs...
)
    @warn "tensor_svd_thing! on CPU does intermediate copying. Consider using `tensor_svd_thin`."

    tmp_U, tmp_s, tmp_V = tensor_svd_thin(CPU(), parent(A), inds(A); kwargs...)

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

# TODO use low-rank approximations
function tensor_svd_thin(
    ::CPU,
    A::Tensor;
    left_inds=(),
    right_inds=(),
    virtualind=Index(gensym(:vind)),
    inplace=false,
    maxdim=nothing,
    kwargs...,
)
    left_inds, right_inds = factorinds(inds(A), left_inds, right_inds)
    @argcheck isdisjoint(left_inds, right_inds)
    @argcheck issetequal(left_inds ∪ right_inds, inds(A))
    @argcheck virtualind ∉ inds(A)

    inds_u = [left_inds; virtualind]
    inds_v = [right_inds; virtualind]
    inds_s = [virtualind]

    # permute array
    left_sizes = map(Base.Fix1(size, A), left_inds)
    right_sizes = map(Base.Fix1(size, A), right_inds)
    Amat = permutedims(A, [left_inds..., right_inds...])
    Amat = reshape(parent(Amat), prod(left_sizes), prod(right_sizes))

    # compute SVD
    U, s, V = if inplace
        LinearAlgebra.svd!(Amat; kwargs...)
    else
        LinearAlgebra.svd(Amat; kwargs...)
    end

    # tensorify results
    U = Tensor(reshape(U, left_sizes..., size(U, 2)), inds_u)
    s = Tensor(s, inds_s)
    Vt = Tensor(reshape(conj(V), right_sizes..., size(V, 2)), inds_v)

    # ad-hoc truncation
    # TODO use low-rank approximations
    if !isnothing(maxdim)
        U = view(U, inds_s => 1:maxdim)
        s = view(s, inds_s => 1:maxdim)
        Vt = view(Vt, inds_s => 1:maxdim)
    end

    return U, s, Vt
end

# GPU - CUDA
function tensor_svd_thin(arch::GPU, A::Tensor; kwargs...)
    U, s, V = allocate_result(tensor_svd_thin, A; kwargs...)
    tensor_svd_thin!(arch, A, U, s, V; kwargs...)
end

function tensor_svd_thin!(
    ::GPU,
    A::Tensor,
    U::Tensor,
    s::Tensor,
    V::Tensor;
    kwargs...
)
    tensor_svd_thin!(GPU(), parent(A), inds(A), U, inds(U), s, V, inds(V); kwargs...)
end

function tensor_svd_thin!(
    ::GPU,
    A::AbstractArray,
    inds_a,
    U::AbstractArray,
    inds_u,
    s::AbstractArray,
    V::AbstractArray,
    inds_v;
    kwargs...
)
    modemap = Dict{Index,Int}(ind => i for (i, ind) in enumerate(unique(inds_a ∪ inds_u ∪ inds_v)))
    modes_a = [modemap[ind] for ind in inds(A)]
    modes_u = [modemap[ind] for ind in inds(U)]
    modes_v = [modemap[ind] for ind in inds(V)]

    # call to cuTensorNet SVD method is implemented as `LinearAlgebra.svd!`
    LinearAlgebra.svd!(A, modes_a, U, modes_u, s, V, modes_v; kwargs...)

    return u, s, v
end
