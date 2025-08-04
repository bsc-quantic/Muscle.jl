module MuscleChainRulesTestUtilsExt

using Muscle
using ChainRulesCore
using ChainRulesTestUtils

const Random = ChainRulesTestUtils.Random

ChainRulesTestUtils.rand_tangent(::Random.AbstractRNG, x::Index) = NoTangent()
ChainRulesTestUtils.rand_tangent(::Random.AbstractRNG, x::Base.AbstractVecOrTuple{<:Index}) = NoTangent()
function ChainRulesTestUtils.rand_tangent(rng::Random.AbstractRNG, x::Tensor)
    Tensor(ChainRulesTestUtils.rand_tangent(rng, parent(x)), inds(x))
end

end
