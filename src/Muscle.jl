module Muscle

abstract type Implementation end
abstract type Naive <: Implementation end
abstract type Vectorized <: Implementation end

include("mapswap.jl")
export mapswap!

include("blocktranspose.jl")
export blocktranspose!

include("Numerics/gramschmidt.jl")
export gramschmidt!

include("Einsum/Einsum.jl")
import .Einsum: @einengine_str, einsum, einsum!
export @einengine_str, einsum, einsum!

end
