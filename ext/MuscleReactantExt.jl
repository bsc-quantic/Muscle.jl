module MuscleReactantExt

using Muscle
using Reactant
using Reactant: TracedRNumber, TracedRArray
const MLIR = Reactant.MLIR
const stablehlo = MLIR.Dialects.stablehlo

Muscle.memory_space(::Type{<:TracedRArray}) = Muscle.ReactantMemorySpace()
Muscle.memory_space(::Type{<:Reactant.AnyConcreteRArray}) = Muscle.ReactantMemorySpace()

Muscle.choose_backend(
    ::typeof(Muscle.binary_einsum!),
    ::TracedRArray,
    ::TracedRArray,
    ::TracedRArray,
) = Muscle.BackendReactant()

# we specify `mode` and `track_numbers` types due to ambiguity
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    return Tensor
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor{T}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T}
    return Tensor{TracedRNumber{T}}
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor{T,N}}),
    seen,
    mode::Reactant.TraceMode,
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

function Reactant.Compiler.make_tracer(
    seen,
    @nospecialize(prev::RT),
    @nospecialize(path),
    mode;
    kwargs...,
) where {RT<:Tensor}
    traced_data = Reactant.Compiler.make_tracer(
        seen,
        parent(prev),
        Reactant.append_path(path, :data),
        mode;
        kwargs...,
    )
    return Tensor(traced_data, inds(prev))
end

function Reactant.Compiler.create_result(
    @nospecialize(tocopy::Tensor),
    @nospecialize(path),
    args...,
)
    data = Reactant.Compiler.create_result(
        parent(tocopy),
        Reactant.append_path(path, :data),
        args...,
    )
    return :($Tensor($data, $(inds(tocopy))))
end

Muscle.memory_space(::TracedRArray) = Muscle.ReactantMemorySpace()
Muscle.memory_space(::Reactant.AnyConcreteRArray) = Muscle.ReactantMemorySpace()

function Muscle.unary_einsum(
    @nospecialize(a::Tensor{TracedRNumber{T},N,TracedRArray{T,N}});
    dims = nonunique(inds(a)),
    out = nothing,
) where {T,N}
    error("compilation of `Muscle.unary_einsum` is not yet supported")
end

Base.@nospecializeinfer @noinline function Muscle.binary_einsum(
    @nospecialize(a::Tensor{TracedRNumber{Ta},Na,TracedRArray{Ta,Na}}),
    @nospecialize(b::Tensor{TracedRNumber{Tb},Nb,TracedRArray{Tb,Nb}});
    kwargs...,
) where {Ta,Na,Tb,Nb}
    dims = get(kwargs, :dims) do
        ∩(inds(a), inds(b))
    end
    out = get(kwargs, :out, nothing)

    ia, ib = collect(inds(a)), collect(inds(b))
    @assert allunique(ia) "can't perform unary einsum operations on binary einsum"
    @assert allunique(ib) "can't perform unary einsum operations on binary einsum"
    @assert dims ⊆ ia ∩ ib "`dims` must be a subset of the intersection of the indices of the two tensors"
    @assert isnothing(out) || out ⊆ ia ∪ ib "`out` must be a subset of the union of the indices of the two tensors"
    @assert isnothing(out) || allunique(out) "indices in `out` for a binary einsum must be unique (no repetitions)"

    # override `dims` if `out` is provided
    dims = !isnothing(out) ? setdiff(dims, out) : dims

    contracting_inds = ∩(dims, ia, ib)
    contracting_dimensions = if isempty(contracting_inds)
        (Int[], Int[])
    else
        (
            map(i -> findfirst(==(i), ia), contracting_inds),
            map(i -> findfirst(==(i), ib), contracting_inds),
        )
    end

    batching_inds = setdiff(ia ∩ ib, dims)
    batching_dimensions = if isempty(batching_inds)
        (Int[], Int[])
    else
        (
            map(i -> findfirst(==(i), ia), batching_inds),
            map(i -> findfirst(==(i), ib), batching_inds),
        )
    end

    result_inds =
        setdiff(ia, contracting_inds, batching_inds) ∪
        setdiff(ib, contracting_inds, batching_inds)
    ic = vcat(batching_inds, result_inds)

    # TODO replace for `Ops.convert`/`adapt` when it's available (there can be problems with nested array structures)
    T = Base.promote_eltype(a, b)
    da =
        eltype(a) != T ? TracedRArray{Reactant.unwrapped_eltype(T),ndims(a)}(parent(a)) :
        parent(a)
    db =
        eltype(b) != T ? TracedRArray{Reactant.unwrapped_eltype(T),ndims(b)}(parent(b)) :
        parent(b)

    data = Reactant.Ops.dot_general(da, db; contracting_dimensions, batching_dimensions)

    # if `out` is provided, emit `stablehlo.transpose` to correct dimension order
    if !isnothing(out)
        data = Reactant.Ops.transpose(data, map(i -> findfirst(==(i), ic), out))
        ic = out
    end

    return Tensor(data, ic)
end

function Muscle.binary_einsum(
    @nospecialize(a::Tensor),
    @nospecialize(b::Tensor{TracedRNumber{T},N,TracedRArray{T,N}});
    kwargs...,
) where {T,N}
    Muscle.binary_einsum(b, a; kwargs...)
end

function Muscle.binary_einsum(
    @nospecialize(a::Tensor{TracedRNumber{T},N,TracedRArray{T,N}}),
    @nospecialize(b::Tensor);
    kwargs...,
) where {T,N}
    return Muscle.binary_einsum(
        a,
        Tensor(Reactant.Ops.constant(parent(b)), inds(b));
        kwargs...,
    )
end

# TODO binary_einsum!

# fixes issue with default `conj(x::AbstractArray) = x` method from Base (it might be overlayed in Reactant.jl)
Base.conj(@nospecialize(x::Tensor{<:TracedRNumber,N,<:TracedRArray})) where {N} = x
function Base.conj(
    @nospecialize(x::Tensor{TracedRNumber{T},N,<:TracedRArray})
) where {T<:Complex,N}
    Tensor(conj(parent(x)), inds(x))
end

end
