module MuscleReactantExt

using Muscle
using Muscle: BackendReactant
using Reactant
using Reactant: TracedRNumber, TracedRArray
const MLIR = Reactant.MLIR
const stablehlo = MLIR.Dialects.stablehlo

Muscle.memory_space(::Type{<:TracedRArray}) = Muscle.ReactantMemorySpace()
Muscle.memory_space(::Type{<:Reactant.AnyConcreteRArray}) = Muscle.ReactantMemorySpace()

# we specify `mode` and `track_numbers` types due to ambiguity
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    return Tensor
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor{T}}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T}
    return Tensor{TracedRNumber{T}}
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor{T,N}}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T,N}
    return Tensor{TracedRNumber{T,N}}
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor{T,N,A}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    sharding,
    runtime,
) where {T,N,A}
    A_traced = Reactant.traced_type_inner(A, seen, mode, track_numbers, sharding, runtime)
    T_traced = eltype(A_traced)
    return Tensor{T_traced,N,A_traced}
end

Base.Base.@nospecializeinfer function Muscle.choose_backend_rule(
    ::typeof(Muscle.binary_einsum!),
    @nospecialize(_::TracedRArray),
    @nospecialize(_::TracedRArray),
    @nospecialize(_::TracedRArray)
)
    Muscle.BackendReactant()
end

function Muscle.unary_einsum(@nospecialize(a::Tensor{TracedRNumber{T}}); dims=nonunique(inds(a)), out=nothing) where {T}
    error("compilation of `Muscle.unary_einsum` is not yet supported")
end

Base.Base.@nospecializeinfer function Muscle.choose_backend_rule(
    ::typeof(Muscle.binary_einsum),
    @nospecialize(_::Type{<:Union{<:TracedRArray,<:TracedRNumber}}),
    @nospecialize(_::Type{<:Union{<:TracedRArray,<:TracedRNumber}}),
)
    Muscle.BackendReactant()
end

Base.Base.@nospecializeinfer function Muscle.choose_backend_rule(
    ::typeof(Muscle.binary_einsum),
    @nospecialize(_::Type{<:Union{<:TracedRArray,<:TracedRNumber}}),
    @nospecialize(_::Type{<:AbstractArray}),
)
    Muscle.BackendReactant()
end

Base.Base.@nospecializeinfer function Muscle.choose_backend_rule(
    ::typeof(Muscle.binary_einsum),
    @nospecialize(_::Type{<:AbstractArray}),
    @nospecialize(_::Type{<:Union{<:TracedRArray,<:TracedRNumber}}),
)
    Muscle.BackendReactant()
end

Base.Base.@nospecializeinfer function Muscle.choose_backend_rule(
    ::typeof(Muscle.binary_einsum!),
    @nospecialize(_::Type{<:TracedRArray}),
    @nospecialize(_::Type{<:Union{<:TracedRArray,<:TracedRNumber}}),
    @nospecialize(_::Type{<:Union{<:TracedRArray,<:TracedRNumber}})
)
    Muscle.BackendReactant()
end

Base.@nospecializeinfer @noinline function Muscle.binary_einsum(
    ::BackendReactant, inds_c, @nospecialize(a::Tensor{TracedRNumber{Ta}}), @nospecialize(b::Tensor{TracedRNumber{Tb}})
) where {Ta,Tb}
    out = inds_c
    dims = setdiff(inds(a) ∩ inds(b), out)

    ia, ib = collect(inds(a)), collect(inds(b))
    @assert allunique(ia) "can't perform unary einsum operations on binary einsum"
    @assert allunique(ib) "can't perform unary einsum operations on binary einsum"
    @assert dims ⊆ ia ∩ ib "`dims` must be a subset of the intersection of the indices of the two tensors"
    @assert isnothing(out) || out ⊆ ia ∪ ib "`out` must be a subset of the union of the indices of the two tensors"
    @assert isnothing(out) || allunique(out) "indices in `out` for a binary einsum must be unique (no repetitions)"

    contracting_inds = ∩(dims, ia, ib)
    contracting_dimensions = if isempty(contracting_inds)
        (Int[], Int[])
    else
        (map(i -> findfirst(==(i), ia), contracting_inds), map(i -> findfirst(==(i), ib), contracting_inds))
    end

    batching_inds = setdiff(ia ∩ ib, dims)
    batching_dimensions = if isempty(batching_inds)
        (Int[], Int[])
    else
        (map(i -> findfirst(==(i), ia), batching_inds), map(i -> findfirst(==(i), ib), batching_inds))
    end

    result_inds = setdiff(ia, contracting_inds, batching_inds) ∪ setdiff(ib, contracting_inds, batching_inds)
    ic = vcat(batching_inds, result_inds)

    # StableHLO expects matching element types
    T = Base.promote_eltype(a, b)
    da = T.(Reactant.materialize_traced_array(parent(a)))
    db = T.(Reactant.materialize_traced_array(parent(b)))

    data = Reactant.Ops.dot_general(da, db; contracting_dimensions, batching_dimensions)

    # if `out` is provided, emit `stablehlo.transpose` to correct dimension order
    if !isempty(out)
        data = Reactant.Ops.transpose(data, map(i -> findfirst(==(i), ic), out))
        ic = out
    end

    return Tensor(data, ic)
end

function Muscle.binary_einsum(
    ::BackendReactant, inds_c, @nospecialize(a::Tensor), @nospecialize(b::Tensor{TracedRNumber{T}}); kwargs...
) where {T}
    Muscle.binary_einsum(BackendReactant(), inds_c, b, a; kwargs...)
end

function Muscle.binary_einsum(
    ::BackendReactant, inds_c, @nospecialize(a::Tensor{TracedRNumber{T}}), @nospecialize(b::Tensor); kwargs...
) where {T}
    return Muscle.binary_einsum(
        BackendReactant(), inds_c, a, Tensor(Reactant.Ops.constant(parent(b)), inds(b)); kwargs...
    )
end

# TODO binary_einsum!

# fixes issue with default `conj(x::AbstractArray) = x` method from Base (it might be overlayed in Reactant.jl)
Base.conj(@nospecialize(x::Tensor{<:TracedRNumber})) = x
Base.conj(@nospecialize(x::Tensor{TracedRNumber{T}})) where {T<:Complex} = Tensor(conj(parent(x)), inds(x))

function __init__()
    Reactant.@skip_rewrite_func Muscle.binary_einsum
    Reactant.@skip_rewrite_func Muscle.nonunique
    Reactant.@skip_rewrite_type Type{<:Muscle.Index}
    Reactant.@skip_rewrite_type Type{<:Muscle.Tensor}
end

end
