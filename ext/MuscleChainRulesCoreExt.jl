module MuscleChainRulesCoreExt

# Implementation details:
# - `Tensor` is NOT an `AbstractArray` anymore (too many conflicting dispatches), but shows a similar interface
# - `Tensor` is it's own tangent type 
# TODO check out how we can tell AD engines that `Tensor` is its own tangent type

using Muscle
using Muscle: nonunique
using ChainRulesCore

@non_differentiable Muscle.ImmutableArray(x)
@non_differentiable Muscle.Index(x)
@non_differentiable Muscle.inds(x)

function ChainRulesCore.ProjectTo(tensor::T) where {T<:Tensor}
    return ProjectTo{T}(; data=ProjectTo(parent(tensor)), inds=inds(tensor))
end

function (projector::ProjectTo{T})(dx::T) where {T<:Tensor}
    @assert projector.inds == inds(dx)
    return T(projector.data(parent(dx)), projector.inds)
end

# function (projector::ProjectTo{T})(dx::Tangent{T}) where {T<:Tensor}
#     return Tangent{Tensor}(projector.data(dx.data), projector.inds)
# end

function (projector::ProjectTo{Tensor{T,0}})(dx::T) where {T}
    return T(projector.data(fill(dx)), projector.inds)
end

# function (projector::ProjectTo{Tensor{T,N,A}})(dx::A) where {T,N,A<:AbstractArray{T,N}}
#     return Tensor{T,N,A}(projector.data(dx), projector.inds)
# end

# `Tensor` constructor
ChainRulesCore.frule((_, Δ, _), T::Type{<:Tensor}, data, inds) = T(data, inds), T(Δ, inds)

Tensor_pullback(Δ::Tensor) = (NoTangent(), parent(Δ), NoTangent())
Tensor_pullback(Δ::Tangent{<:Tensor}) = (NoTangent(), Δ.data, NoTangent())
Tensor_pullback(Δ::AbstractThunk) = Tensor_pullback(unthunk(Δ))
ChainRulesCore.rrule(T::Type{<:Tensor}, data::AbstractArray, inds) = T(data, inds), Tensor_pullback

# `conj`
ChainRulesCore.frule((_, Δ), ::typeof(Base.conj), x::Tensor) = conj(x), conj(Δ)

conj_pullback(Δ) = (NoTangent(), conj(Δ))
conj_pullback(Δ::AbstractThunk) = conj_pullback(unthunk(Δ))
ChainRulesCore.rrule(::typeof(Base.conj), x::Tensor) = conj(x), conj_pullback

function ChainRulesCore.frule((_, ẋ), ::typeof(unary_einsum), x::Tensor; kwargs...)
    return unary_einsum(x; kwargs...), unary_einsum(ẋ; kwargs...)
end

function ChainRulesCore.frule((_, ȧ, ḃ), ::typeof(binary_einsum), a::Tensor, b::Tensor; kwargs...)
    c = binary_einsum(a, b; kwargs...)
    ċ = binary_einsum(ȧ, b; kwargs...) + binary_einsum(a, ḃ; kwargs...)
    return c, ċ
end

function ChainRulesCore.rrule(::typeof(unary_einsum), x::Tensor; dims=nonunique(inds(x)), out=nothing)
    y = unary_einsum(x; dims, out)
    proj = ProjectTo(parent(x))
    ix = inds(x)
    sx = size(x)

    function pullback_unary_einsum(ȳ)
        y_shape_with_singletons = map(ix) do i
            loc = findfirst(==(i), inds(ȳ))
            isnothing(loc) ? 1 : size(ȳ, loc)
        end

        dims_to_repeat = map(zip(sx, y_shape_with_singletons .== 1)) do (dₐ, issingleton)
            issingleton ? dₐ : 1
        end

        x̄ = Tensor(proj(repeat(reshape(parent(ȳ), y_shape_with_singletons...), dims_to_repeat...)), ix)

        return (NoTangent(), x̄)
    end
    pullback_unary_einsum(ȳ::AbstractThunk) = pullback_unary_einsum(unthunk(ȳ))

    return y, pullback_unary_einsum
end

function ChainRulesCore.rrule(::typeof(binary_einsum), a, b; kwargs...)
    c = binary_einsum(a, b; kwargs...)
    proj_a = ProjectTo(a)
    proj_b = ProjectTo(b)

    function pullback_binary_einsum(c̄)
        ā = @thunk proj_a(binary_einsum(c̄, conj(b); out=inds(a)))
        b̄ = @thunk proj_b(binary_einsum(conj(a), c̄; out=inds(b)))
        return (NoTangent(), ā, b̄)
    end
    pullback_binary_einsum(c̄::AbstractThunk) = pullback_binary_einsum(unthunk(c̄))

    return c, pullback_binary_einsum
end

end
