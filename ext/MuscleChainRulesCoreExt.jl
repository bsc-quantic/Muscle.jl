module MuscleChainRulesCoreExt

using Muscle
using ChainRulesCore

# function ChainRulesCore.frule((_, _, ȧ, _), ::typeof(einsum), ic, a, ia)
#     return einsum(ic, a, ia), einsum(ic, ȧ, ia)
# end

# function ChainRulesCore.frule((_, _, ȧ, _, ḃ, _), ::typeof(einsum), ic, a, ia, b, ib)
#     c = einsum(ic, a, ia, b, ib)
#     ċ = einsum(ic, ȧ, ia, b, ib) + einsum(ic, a, ia, ḃ, ib)
#     return c, ċ
# end

# function ChainRulesCore.rrule(::typeof(einsum), ic, a, ia)
#     c = einsum(ic, a, ia)
#     proj = ProjectTo(a)

#     function einsum_pullback(c̄)
#         c_shape_with_singletons = map(ia) do i
#             loc = findfirst(==(i), ic)
#             isnothing(loc) ? 1 : size(c̄, loc)
#         end

#         dims_to_repeat = map(zip(size(a), c_shape_with_singletons .== 1)) do (dₐ, issingleton)
#             issingleton ? dₐ : 1
#         end
#         ā::typeof(a) = proj(repeat(reshape(c̄, c_shape_with_singletons...), dims_to_repeat...))

#         return (NoTangent(), NoTangent(), ā, NoTangent())
#     end
#     einsum_pullback(c̄::AbstractThunk) = einsum_pullback(unthunk(c̄))

#     return c, einsum_pullback
# end

# function ChainRulesCore.rrule(::typeof(einsum), ic, a, ia, b, ib)
#     c = einsum(ic, a, ia, b, ib)
#     proj_a = ProjectTo(a)
#     proj_b = ProjectTo(b)

#     function einsum_pullback(c̄)
#         ā = @thunk proj_a(einsum(ia, c̄, ic, conj(b), ib))
#         b̄ = @thunk proj_b(einsum(ib, conj(a), ia, c̄, ic))
#         return (NoTangent(), NoTangent(), ā, NoTangent(), b̄, NoTangent())
#     end
#     einsum_pullback(c̄::AbstractThunk) = einsum_pullback(unthunk(c̄))

#     return c, einsum_pullback
# end

end
