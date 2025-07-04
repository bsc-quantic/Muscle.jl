module MuscleDaggerExt

using Muscle
using Dagger

Muscle.Domain(::Type{<:Dagger.DArray}) = Muscle.DomainDagger()

Dagger.domainchunks(t::Tensor) = Dagger.domainchunks(parent(t))

include("binary_einsum.jl")

end
