module MuscleDaggerExt

using Muscle
using Dagger

Muscle.Domain(::Type{<:Dagger.DArray}) = Muscle.DomainDagger()

include("binary_einsum.jl")

end
