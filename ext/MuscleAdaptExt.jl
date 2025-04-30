module MuscleAdaptExt

using Muscle
using Adapt

Adapt.adapt_structure(to, x::Tensor) = Tensor(adapt(to, parent(x)), inds(x))

end
