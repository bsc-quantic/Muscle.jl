module Einsum

"""
    einsum(a,b)

Compute the Einstein summation convention on the operands.
"""
function einsum end
function einsum! end

struct Engine{Name} end

macro einengine_str(name)
    (:(Engine{Symbol($name)}))
end

einsum(::Engine{X}, args...; kwargs...) where {X} = error("""
    Tried to use $X engine without loading it.
    Please load it by running `using $X`.
    """)

einsum!(::Engine{X}, args...; kwargs...) where {X} = error("""
    Tried to use $X engine without loading it.
    Please load it by running `using $X`.
    """)

include("BLAS.jl")

end
