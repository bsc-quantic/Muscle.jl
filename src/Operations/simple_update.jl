function simple_update end
function simple_update! end

choose_backend_rule(::typeof(simple_update), ::DomainHost, ::DomainHost, ::DomainHost) = BackendBase()
choose_backend_rule(::typeof(simple_update), ::DomainCUDA, ::DomainCUDA, ::DomainCUDA) = BackendCuTensorNet()

# absorb behavior trait
# used to keep type-inference happy (`DontAbsorb` returns 3 tensors, while the rest return 2)
abstract type AbsorbBehavior end
struct DontAbsorb <: AbsorbBehavior end
struct AbsorbU <: AbsorbBehavior end
struct AbsorbV <: AbsorbBehavior end
struct AbsorbEqually <: AbsorbBehavior end

# TODO automatically move to GPU if G are on CPU?
function simple_update(
    A::Tensor,
    ind_physical_a,
    B::Tensor,
    ind_physical_b,
    ind_bond_ab,
    G::Tensor,
    ind_physical_g_a,
    ind_physical_g_b;
    kwargs...,
)
    backend = choose_backend(simple_update, parent(A), parent(B), parent(G))

    return simple_update(
        backend, A, ind_physical_a, B, ind_physical_b, ind_bond_ab, G, ind_physical_g_a, ind_physical_g_b; kwargs...
    )
end

# TODO change and document the way we indicate the physical indices to contract
function simple_update(
    ::Backend,
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
    Θ = Muscle.binary_einsum(Muscle.binary_einsum(A, B; dims=[ind_bond_ab]), G; dims=[ind_physical_a, ind_physical_b])
    Θ = replace(Θ, ind_physical_g_a => ind_physical_a, ind_physical_g_b => ind_physical_b)

    inds_u = setdiff(inds(A), [ind_bond_ab])
    inds_v = setdiff(inds(B), [ind_bond_ab])
    ind_s = ind_bond_ab
    U, S, V = tensor_svd_thin(Θ; inds_u, inds_v, ind_s)

    # TODO use low-rank approximations
    # ad-hoc truncation
    if !isnothing(maxdim)
        U = view(U, ind_s => 1:min(maxdim, length(S)))
        S = view(S, ind_s => 1:min(maxdim, length(S)))
        V = view(V, ind_s => 1:min(maxdim, length(S)))
    end

    normalize && LinearAlgebra.normalize!(S)

    if absorb isa DontAbsorb
        return U, S, V
    elseif absorb isa AbsorbU
        U = Muscle.hadamard!(U, U, S)
    elseif absorb isa AbsorbV
        V = Muscle.hadamard!(V, V, S)
    elseif absorb isa AbsorbEqually
        S_sqrt = sqrt.(S)
        U = Muscle.hadamard!(U, U, S_sqrt)
        V = Muscle.hadamard!(V, V, S_sqrt)
    end

    return U, V
end
