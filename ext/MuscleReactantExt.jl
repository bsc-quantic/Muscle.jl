module MuscleReactantExt

using Muscle
using Muscle: BackendReactant
using Reactant
using Reactant: TracedRNumber, TracedRArray
const MLIR = Reactant.MLIR
const stablehlo = MLIR.Dialects.stablehlo
using PrecompileTools

Muscle.Domain(::Type{<:TracedRArray}) = Muscle.DomainReactant()
Muscle.Domain(::Type{<:Reactant.AnyConcreteRArray}) = Muscle.DomainReactant()

# we specify `mode` and `track_numbers` types due to ambiguity
# TODO in Reactant v0.3, rename it to `Reactant.transmute_type`
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

# TODO in Reactant v0.3, rename it to `Reactant.transmute_type`
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

# TODO in Reactant v0.3, rename it to `Reactant.transmute_type`
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

# TODO in Reactant v0.3, rename it to `Reactant.transmute_type`
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor{T,N,A}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    sharding,
    runtime,
) where {T,N,A}
    # TODO in Reactant v0.3, rename it to `Reactant.transmute_type`
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

function Muscle.unary_einsum(
    ::Muscle.DomainReactant, @nospecialize(a::Tensor{TracedRNumber{T}}); dims=nonunique(inds(a)), out=nothing
) where {T}
    error("compilation of `Muscle.unary_einsum` is not yet supported")
end

Base.@nospecializeinfer @noinline function Muscle.binary_einsum(
    ::BackendReactant, inds_c, @nospecialize(a::Tensor{TracedRNumber{Ta}}), @nospecialize(b::Tensor{TracedRNumber{Tb}}); kwargs...
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

# This function is used to skip rewriting of certain functions and type constructors in Reactant.jl, which is necessary
# for overlaying methods called dynamically. By skipping the rewrite where we know it's ok, Julia compilation should 
# take less time. It must be called on the top level for precompilation and in `__init__` for runtime.
function muscle_skip_rewrites()
    Reactant.@skip_rewrite_func Muscle.binary_einsum
    Reactant.@skip_rewrite_func Muscle.nonunique
    Reactant.@skip_rewrite_type Type{<:Muscle.Index}
    Reactant.@skip_rewrite_type Type{<:Muscle.Tensor}
end

function __init__()
    muscle_skip_rewrites()
end

# @static if Reactant.Reactant_jll.is_available() && Reactant.precompilation_supported()
#     @setup_workload begin
#         muscle_skip_rewrites()

#         # Initialize the MLIR dialects and set up the XLA client
#         # NOTE taken from https://github.com/EnzymeAD/Reactant.jl/blob/77a9c694c4004cf08b270d08f8a5f51b7bdbf97e/src/Precompile.jl#L57-L83
#         Reactant.initialize_dialect()
#         if Reactant.XLA.REACTANT_XLA_RUNTIME == "PJRT"
#             client = Reactant.XLA.PJRT.CPUClient(; checkcount=false)
#         elseif Reactant.XLA.REACTANT_XLA_RUNTIME == "IFRT"
#             client = Reactant.XLA.IFRT.CPUClient(; checkcount=false)
#         else
#             error("Unsupported runtime: $(Reactant.XLA.REACTANT_XLA_RUNTIME)")
#         end

#         @compile_workload begin
#             for (Ta, Tb) in [
#                 (Float32, Float32),
#                 (Float64, Float64),
#                 (ComplexF32, ComplexF32),
#                 (ComplexF64, ComplexF64),
#                 (Float64, ComplexF64),
#                 (ComplexF64, Float64),
#             ]
#                 a = Tensor(Reactant.to_rarray(ones(Ta, 2, 2); client), [:i, :j])
#                 b = Tensor(Reactant.to_rarray(ones(Tb, 2, 2); client), [:j, :k])
#                 Reactant.compile(Muscle.binary_einsum, (a, b); client, optimize=:all)
#             end
#         end

#         # clean deinitialization
#         Reactant.XLA.free_client(client)
#         client.client = C_NULL
#         Reactant.deinitialize_dialect()
#         Reactant.clear_oc_cache()
#     end
# end

end
