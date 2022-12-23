module Muscle

abstract type Implementation end
abstract type Naive <: Implementation end
abstract type Vectorized <: Implementation end

include("Kernel/Kernel.jl")

end
