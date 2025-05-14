abstract type Implementation end
abstract type Naive <: Implementation end
abstract type Vectorized <: Implementation end

"""
	mapswap!([], A, B, f, g)

Swaps the content of two `StridedArray`s or `SubArray`s and applies the `f`,`g` maps. It is equivalent to:
```julia
map!(f, A)
map!(g, B)
swap!(A, B)
```
"""
function mapswap! end

mapswap!(A::AbstractArray, B::AbstractArray, f = identity, g = identity) =
    mapswap!(Naive, A, B, f, g)

function mapswap!(
    ::Type{Naive},
    A::AbstractArray{T},
    B::AbstractArray{T},
    f = identity,
    g = identity,
) where {T}
    @inbounds for i in eachindex(A, B)
        (A[i], B[i]) = (g(B[i]), f(A[i]))
    end
end

function mapswap!(
    ::Type{Vectorized},
    A::AbstractArray{T},
    B::AbstractArray{T},
    f = identity,
    g = identity,
) where {T}
    @inbounds @simd for i in eachindex(A, B)
        (A[i], B[i]) = (g(B[i]), f(A[i]))
    end
end
