using LinearAlgebra: LinearAlgebra
using cuTensorNet: cuTensorNet
using ..Muscle: factorinds

# TODO implement low-rank approximations (truncated QR, reduced QR...)

function tensor_qr_thin end
function tensor_qr_thin! end

# dispatch to correct architecture
tensor_qr_thin(A::Tensor; kwargs...) = tensor_qr_thin(arch(A), A; kwargs...)

function allocate_result(::typeof(tensor_qr_thin), A; inds_q=(), inds_r=(), kwargs...)
    inds_q, inds_r = factorinds(inds(A), inds_q, inds_r)
    left_extent = prod(Base.Fix1(size, A), inds_q)
    right_extent = prod(Base.Fix1(size, A), inds_r)
    virtual_extent = min(left_extent, right_extent)

    Q = Tensor(similar(parent(A), left_extent..., virtual_extent), [inds_q..., ind_s])
    R = Tensor(similar(parent(A), right_extent..., virtual_extent), [inds_r..., ind_s])

    return Q, R
end

# CPU
function tensor_qr_thin!(
    ::CPU,
    A::Tensor,
    Q::Tensor,
    R::Tensor;
    kwargs...
)
    @warn "tensor_qr_thin! on CPU does intermediate copying. Consider using `tensor_qr_thin`."

    tmp_Q, tmp_R = tensor_qr_thin(CPU(), A; inds_q=setdiff(inds(Q), inds(R)), inds_r=setdiff(inds(R), inds(Q)), kwargs...)

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

# TODO use low-rank approximations
function tensor_qr_thin(
    ::CPU,
    A::Tensor;
    inds_q=(),
    inds_r=(),
    ind_virtual=Index(gensym(:qr)),
    inplace=false,
    kwargs...,
)
    ind_virtual ∉ inds(A) ||
        throw(ArgumentError("new virtual bond name ($ind_virtual) cannot be already be present"))

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

# GPU - CUDA
function tensor_qr_thin(arch::GPU, A::Tensor; kwargs...)
    U, s, V = allocate_result(tensor_qr_thin, A; kwargs...)
    tensor_qr_thin!(arch, A, U, s, V; kwargs...)

    # TODO tensorify results
end

function tensor_qr_thin!(
    ::GPU,
    A::Tensor,
    U::Tensor,
    s::Tensor,
    V::Tensor;
    kwargs...
)
    tensor_qr_thin!(GPU(), parent(A), inds(A), U, inds(U), s, V, inds(V); kwargs...)
end

function tensor_qr_thin!(
    ::GPU,
    A::AbstractArray,
    inds_a,
    Q::AbstractArray,
    inds_q,
    R::AbstractArray,
    inds_r;
    kwargs...
)
    modemap = Dict{Index,Int}(ind => i for (i, ind) in enumerate(unique(inds_a ∪ inds_q ∪ inds_r)))
    modes_a = [modemap[ind] for ind in inds(A)]
    modes_q = [modemap[ind] for ind in inds(U)]
    modes_r = [modemap[ind] for ind in inds(V)]

    # call to cuTensorNet SVD method is implemented as `LinearAlgebra.qr!`
    LinearAlgebra.qr!(A, modes_a, Q, modes_q, R, modes_r; kwargs...)

    return Q, R
end
