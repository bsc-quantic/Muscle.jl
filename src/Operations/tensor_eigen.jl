using LinearAlgebra: LinearAlgebra
using ..Muscle: factorinds

# TODO implement low-rank approximations (truncated eigen, reduced eigen...)

"""
    Muscle.tensor_eigen_thin(tensor::Tensor; inds_u, inds_uinv, ind_lambda, kwargs...)

Perform eigen factorization on a tensor. Either `inds_u` or `inds_uinv` must be specified.

# Keyword arguments

  - `inds_u`: left / U indices to be used in the eigen factorization, except for `ind_lambda`.
  - `inds_uinv`: right / right indices to be used in the eigen factorization, except for `ind_lambda`.
  - `ind_lambda`: name of the virtual bond.
  - `inplace`: If `true`, it will use `A` as workspace variable to save space. Defaults to `false`.
  - `kwargs...`: additional keyword arguments to be passed to `LinearAlgebra.eigen`.
"""
function tensor_eigen_thin end
function tensor_eigen_thin! end

function tensor_bieigen_thin end
function tensor_bieigen_thin! end

choose_backend_rule(::typeof(tensor_eigen_thin), ::Type{<:Array}) = BackendBase()
function choose_backend_rule(::typeof(tensor_eigen_thin!), ::Type{<:Array}, ::Type{<:Array}, ::Type{<:Array})
    BackendBase()
end

choose_backend_rule(::typeof(tensor_bieigen_thin), ::Type{<:Array}) = BackendBase()
function choose_backend_rule(
    ::typeof(tensor_bieigen_thin!), ::Type{<:Array}, ::Type{<:Array}, ::Type{<:Array}, ::Type{<:Array}
)
    BackendBase()
end

function tensor_eigen_thin(
    A::Tensor; inds_u=(), inds_uinv=(), ind_lambda=Index(gensym(:eigen)), inplace=false, kwargs...
)
    backend = choose_backend(tensor_eigen_thin, A)
    return tensor_eigen_thin(backend, A; inds_u, inds_uinv, ind_lambda, inplace, kwargs...)
end

function tensor_bieigen_thin(
    A::Tensor; inds_u=(), inds_uinv=(), ind_lambda=Index(gensym(:eigen)), inplace=false, kwargs...
)
    backend = choose_backend(tensor_bieigen_thin, A)
    return tensor_bieigen_thin(backend, A; inds_u, inds_uinv, ind_lambda, inplace, kwargs...)
end

## `Base`
function tensor_eigen_thin(
    ::BackendBase, A::Tensor; inds_u=(), inds_uinv=(), ind_lambda=Index(gensym(:vind)), inplace=false, kwargs...
)
    inds_u, inds_uinv = factorinds(inds(A), inds_u, inds_uinv)
    @argcheck isdisjoint(inds_u, inds_uinv)
    @argcheck issetequal(inds_u ∪ inds_uinv, inds(A))
    @argcheck ind_lambda ∉ inds(A)

    # permute array
    left_sizes = map(Base.Fix1(size, A), inds_uinv)
    right_sizes = map(Base.Fix1(size, A), inds_u)

    Amat = permutedims(A, [inds_u; inds_uinv])
    Amat = reshape(parent(Amat), prod(left_sizes), prod(right_sizes))

    # compute eigen
    F = if inplace
        LinearAlgebra.eigen!(Amat; kwargs...)
    else
        LinearAlgebra.eigen(Amat; kwargs...)
    end

    # tensorify results
    Λ = Tensor(F.values, [ind_lambda])
    #U = Tensor(reshape(F.vectors, size(F.vectors, 2), right_sizes...), [inds_u; ind_lambda])
    U = Tensor(reshape(F.vectors, left_sizes..., size(F.vectors, 2)), [inds_u; ind_lambda])

    return Λ, U
end

# TODO how can we get the left_sizes for the Uinv matrix from `U`?
function tensor_bieigen_thin(
    ::BackendBase, A::Tensor; inds_u=(), inds_uinv=(), ind_lambda=Index(gensym(:vind)), inplace=false, kwargs...
)
    inds_u, inds_uinv = factorinds(inds(A), inds_u, inds_uinv)
    Λ, U = tensor_eigen_thin(BackendBase(), A; inds_u, inds_uinv, ind_lambda, inplace, kwargs...)

    Umat = reshape(parent(U), size(U, ind_lambda), prod(map(Base.Fix1(size, U), inds_u)))

    left_sizes = map(Base.Fix1(size, A), inds_uinv)

    # TODO probably a better way to get the `Uinv` matrix
    Uinv = Tensor(reshape(inv(Umat), left_sizes..., size(Λ, ind_lambda)), [ind_lambda; inds_uinv])

    return U, Λ, Uinv
end

function tensor_eigen_thin!(::BackendBase, Λ::Tensor, U::Tensor, A::Tensor; kwargs...)
    @warn "tensor_eigen_thin! on BackendBase does intermediate copying. Consider using `tensor_eigen_thin`."

    tmp_Λ, tmp_U = tensor_eigen_thin(BackendBase(), A; inds_u=inds(U), ind_lambda=only(inds(Λ)), kwargs...)

    @argcheck arch(tmp_Λ) == arch(Λ)
    @argcheck arch(tmp_U) == arch(U)

    @argcheck inds(tmp_Λ) == inds(Λ)
    @argcheck inds(tmp_U) == inds(U)

    @argcheck size(tmp_Λ) == size(Λ)
    @argcheck size(tmp_U) == size(U)

    copyto!(Λ, tmp_Λ)
    copyto!(U, tmp_U)

    return Λ, U
end

function tensor_bieigen_thin!(::BackendBase, Uinv::Tensor, Λ::Tensor, U::Tensor, A::Tensor; kwargs...)
    @warn "tensor_eigen_thin! on BackendBase does intermediate copying. Consider using `tensor_eigen_thin`."

    tmp_Uinv, tmp_Λ, tmp_U = tensor_bieigen_thin(
        BackendBase(), A; inds_u=inds(U), inds_uinv=inds(Uinv), ind_lambda=only(inds(Λ)), kwargs...
    )

    @argcheck arch(tmp_Uinv) == arch(Uinv)
    @argcheck arch(tmp_Λ) == arch(Λ)
    @argcheck arch(tmp_U) == arch(U)

    @argcheck inds(tmp_Uinv) == inds(Uinv)
    @argcheck inds(tmp_Λ) == inds(Λ)
    @argcheck inds(tmp_U) == inds(U)

    @argcheck size(tmp_Uinv) == size(Uinv)
    @argcheck size(tmp_Λ) == size(Λ)
    @argcheck size(tmp_U) == size(U)

    copyto!(Uinv, tmp_Uinv)
    copyto!(Λ, tmp_Λ)
    copyto!(U, tmp_U)

    return Uinv, Λ, U
end
