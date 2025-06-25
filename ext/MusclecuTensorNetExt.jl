module MusclecuTensorNetExt

using ArgCheck
using CUDA
using cuTensorNet: cuTensorNet
using Muscle
using Muscle: AbsorbBehavior, BackendCuTensorNet

# TODO customize SVD algorithm
# TODO configure GPU stream
# TODO cache workspace memory
# TODO do QR before SU to reduce computational cost on A,B with ninds > 3 but not when size(extent) ~ size(rest)
function simple_update(
    ::BackendCuTensorNet,
    A::Tensor,
    ind_physical_a::Index,
    B::Tensor,
    ind_physical_b::Index,
    ind_bond_ab::Index,
    G::Tensor,
    ind_physical_g_a::Index,
    ind_physical_g_b::Index;
    normalize::Bool=false,
    absorb::AbsorbBehavior=DontAbsorb(),
    atol::Float64=0.0,
    rtol::Float64=0.0,
    maxdim=nothing,
)
    all_inds = unique(∪(inds(A), inds(B), inds(G)))
    modes_a = Int[findfirst(==(i), all_inds) for i in inds(A)]
    modes_b = Int[findfirst(==(i), all_inds) for i in inds(B)]
    modes_g = Int[findfirst(==(i), all_inds) for i in inds(G)]

    # implement maxdim for simple_update on GPU (i think we just need to correctly size U and V beforehand)
    new_size_bond_ab = min(length(A) ÷ size(A, ind_bond_ab), length(B) ÷ size(B, ind_bond_ab))
    if !isnothing(maxdim)
        new_size_bond_ab = min(new_size_bond_ab, maxdim)
    end

    # create U and V tensors with the same size as A and B, but with the new bond index size
    U = Tensor(
        CUDA.zeros(eltype(A), Tuple([i == ind_bond_ab ? new_size_bond_ab : size(A, i) for i in inds(A)])), inds(A)
    )
    V = Tensor(
        CUDA.zeros(eltype(B), Tuple([i == ind_bond_ab ? new_size_bond_ab : size(B, i) for i in inds(B)])), inds(B)
    )

    # cuTensorNet doesn't like to reuse the physical indices of a and b, so we rename them here
    U = replace(U, ind_physical_a => ind_physical_g_a)
    V = replace(V, ind_physical_b => ind_physical_g_b)

    modes_u = Int[findfirst(==(i), all_inds) for i in inds(U)]
    modes_v = Int[findfirst(==(i), all_inds) for i in inds(V)]

    svd_config = cuTensorNet.SVDConfig(;
        abs_cutoff=atol,
        rel_cutoff=rtol,
        s_partition=if absorb isa DontAbsorb
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_NONE
        elseif absorb isa AbsorbU
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_US
        elseif absorb isa AbsorbV
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_SV
        elseif absorb isa AbsorbEqually
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL
        else
            throw(ArgumentError("Unknown value for absorb: $absorb"))
        end,
        s_normalization=if normalize
            cuTensorNet.CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2
        else
            cuTensorNet.CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE
        end,
    )

    S_data = similar(parent(A), real(eltype(A)), (size(A, ind_bond_ab),))

    # TODO use svd_info
    _, _, _, svd_info = cuTensorNet.gateSplit!(
        parent(A),
        modes_a,
        parent(B),
        modes_b,
        parent(G),
        modes_g,
        parent(U),
        modes_u,
        S_data,
        parent(V),
        modes_v;
        svd_config,
    )

    S = Tensor(S_data, [ind_bond_ab])

    # undo the index rename to keep cuTensorNet happy
    U = replace(U, ind_physical_g_a => ind_physical_a)
    V = replace(V, ind_physical_g_b => ind_physical_b)

    if absorb isa DontAbsorb
        return U, S, V
    else
        return U, V
    end
end

## `cuTensorNet`
function tensor_qr_thin(
    ::BackendCuTensorNet, A::Tensor; inds_q=(), inds_r=(), ind_virtual=Index(gensym(:qr)), inplace=false, kwargs...
)
    ind_virtual ∉ inds(A) || throw(ArgumentError("new virtual bond name ($ind_virtual) cannot be already be present"))

    inds_q, inds_r = factorinds(inds(A), inds_q, inds_r)
    @argcheck issetequal(inds_q ∪ inds_r, inds(A))

    size_q = map(Base.Fix1(size, A), inds_q)
    size_r = map(Base.Fix1(size, A), inds_r)
    size_virtual = min(prod(size_q), prod(size_r))

    inds_q = [inds_q..., ind_virtual]
    inds_r = [ind_virtual, inds_r...]

    Q = Tensor(CUDA.zeros(eltype(A), size_q..., size_virtual), inds_q)
    R = Tensor(CUDA.zeros(eltype(A), size_virtual, size_r...), inds_r)

    tensor_qr_thin!(BackendCuTensorNet(), Q, R, A; kwargs...)
    return Q, R
end

function tensor_qr_thin!(::BackendCuTensorNet, Q::Tensor, R::Tensor, A::Tensor; kwargs...)
    return tensor_qr_thin!(BackendCuTensorNet(), parent(Q), inds(Q), parent(R), inds(R), parent(A), inds(A); kwargs...)
end

function tensor_qr_thin!(::BackendCuTensorNet, Q, inds_q, R, inds_r, A, inds_a; kwargs...)
    modemap = Dict(ind => i for (i, ind) in enumerate(unique(inds_a ∪ inds_q ∪ inds_r)))
    modes_a = [modemap[ind] for ind in inds_a]
    modes_q = [modemap[ind] for ind in inds_q]
    modes_r = [modemap[ind] for ind in inds_r]

    # call to cuTensorNet SVD method is implemented as `LinearAlgebra.qr!`
    LinearAlgebra.qr!(A, modes_a, Q, modes_q, R, modes_r; kwargs...)

    return Q, R
end

## `cuTensorNet`
# function tensor_svd_thin(arch::GPU, A::Tensor; kwargs...)
#     U, s, V = allocate_result(tensor_svd_thin, A; kwargs...)
#     tensor_svd_thin!(arch, A, U, s, V; kwargs...)
# end

function tensor_svd_thin!(::BackendCuTensorNet, U::Tensor, s::Tensor, V::Tensor, A::Tensor; kwargs...)
    tensor_svd_thin!(
        BackendCuTensorNet(), parent(U), inds(U), parent(s), parent(V), inds(V), parent(A), inds(A); kwargs...
    )
end

function tensor_svd_thin!(::BackendCuTensorNet, U, inds_u, s, V, inds_v, A, inds_a; kwargs...)
    modemap = Dict{Index,Int}(ind => i for (i, ind) in enumerate(unique(inds_a ∪ inds_u ∪ inds_v)))
    modes_a = [modemap[ind] for ind in inds(A)]
    modes_u = [modemap[ind] for ind in inds(U)]
    modes_v = [modemap[ind] for ind in inds(V)]

    # call to cuTensorNet SVD method is implemented as `LinearAlgebra.svd!`
    LinearAlgebra.svd!(A, modes_a, U, modes_u, s, V, modes_v; kwargs...)

    return u, s, v
end

end
