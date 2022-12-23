module Kernel

abstract type Implementation end
abstract type Naive <: Implementation end
abstract type Vectorized <: Implementation end

include("mapswap.jl")

end