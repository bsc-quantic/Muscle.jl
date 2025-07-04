using LinearAlgebra: LinearAlgebra
using ..Muscle: factorinds

# TODO implement low-rank approximations (truncated QR, reduced QR...)

"""
    tensor_qr_thin(tensor::Tensor; inds_q, inds_r, ind_virtual, kwargs...)

Perform QR factorization on a tensor. Either `inds_q` or `inds_r` must be specified.

# Keyword arguments

  - `inds_q`: left indices to be used in the QR factorization. Defaults to all indices of `t` except `inds_r`.
  - `inds_r`: right indices to be used in the QR factorization. Defaults to all indices of `t` except `inds_q`.
  - `ind_virtual`: name of the virtual bond. Defaults to a random `Index{Symbol}`.
"""
function tensor_qr_thin end
function tensor_qr_thin! end

# TODO add a preference system for some backends
choose_backend_rule(::typeof(tensor_qr_thin), ::DomainHost) = BackendBase()
choose_backend_rule(::typeof(tensor_qr_thin), ::DomainCUDA) = BackendCuTensorNet()

choose_backend_rule(::typeof(tensor_qr_thin!), ::DomainHost, ::DomainHost, ::DomainHost) = BackendBase()
choose_backend_rule(::typeof(tensor_qr_thin!), ::DomainCUDA, ::DomainCUDA, ::DomainCUDA) = BackendCuTensorNet()

function tensor_qr_thin(A::Tensor; inds_q=(), inds_r=(), ind_virtual=Index(gensym(:qr)), inplace=false, kwargs...)
    backend = choose_backend(tensor_qr_thin, A)
    return tensor_qr_thin(backend, A; inds_q, inds_r, ind_virtual, inplace, kwargs...)
end

function tensor_qr_thin(::Backend, A; kwargs...)
    throw(ArgumentError("`tensor_qr_thin` not implemented or not loaded for backend $(typeof(A))"))
end

function tensor_qr_thin!(Q::Tensor, R::Tensor, A::Tensor; kwargs...)
    backend = choose_backend(tensor_qr_thin!, Q, R, A)
    return tensor_qr_thin!(backend, Q, R, A; kwargs...)
end

function tensor_qr_thin!(::Backend, Q, R, A; kwargs...)
    throw(ArgumentError("`tensor_qr_thin!` not implemented or not loaded for backend $(typeof(A))"))
end

function tensor_qr_thin(
    ::BackendBase, A; inds_q=(), inds_r=(), ind_virtual=Index(gensym(:qr)), inplace=false, kwargs...
)
    ind_virtual ∉ inds(A) || throw(ArgumentError("new virtual bond name ($ind_virtual) cannot be already be present"))

    inds_q, inds_r = factorinds(inds(A), inds_q, inds_r)
    @argcheck issetequal(inds_q ∪ inds_r, inds(A))

    # permute array
    left_sizes = map(Base.Fix1(size, A), inds_q)
    right_sizes = map(Base.Fix1(size, A), inds_r)
    Amat = permutedims(A, [inds_q..., inds_r...])
    Amat = reshape(parent(Amat), prod(left_sizes), prod(right_sizes))

    # compute QR
    F = LinearAlgebra.qr(Amat; kwargs...)
    Q, R = Matrix(F.Q), Matrix(F.R)

    # tensorify results
    Q = Tensor(reshape(Q, left_sizes..., size(Q, 2)), [inds_q..., ind_virtual])
    R = Tensor(reshape(R, size(R, 1), right_sizes...), [ind_virtual, inds_r...])

    return Q, R
end

function tensor_qr_thin!(::BackendBase, Q::Tensor, R::Tensor, A::Tensor; kwargs...)
    @warn "tensor_qr_thin! on BackendBase does intermediate copying. Consider using `tensor_qr_thin`."

    tmp_Q, tmp_R = tensor_qr_thin(
        BackendBase(), A; inds_q=setdiff(inds(Q), inds(R)), inds_r=setdiff(inds(R), inds(Q)), kwargs...
    )

    @argcheck arch(tmp_Q) == arch(Q)
    @argcheck arch(tmp_R) == arch(R)

    @argcheck inds(tmp_Q) == inds(Q)
    @argcheck inds(tmp_R) == inds(R)

    @argcheck size(tmp_Q) == size(Q)
    @argcheck size(tmp_R) == size(R)

    copyto!(Q, tmp_Q)
    copyto!(R, tmp_R)

    return Q, R
end
