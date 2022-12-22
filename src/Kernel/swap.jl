using StaticArrays: MVector
using SIMD: VecRange

"""
	swap!(A, B)

Swaps the content of two `StridedArray`s or `SubArray`s.
"""
function swap! end

"""
    naiveswap!(A,B)

Naïve implementation of [`swap!`](@ref).
"""
function naiveswap!(A::AbstractVector{T}, B::AbstractVector{T}) where {T}
    n = length(A)
    @inbounds for i in 1:n
        tmp = A[i]
        A[i] = B[i]
        B[i] = tmp
    end
end

function swap!(A::Ptr{T}, B::Ptr{T}, buffer::MVector{N,T}) where {T,N}
    unsafe_copyto!(buffer, A, N)
    unsafe_copyto!(A, B, N)
    unsafe_copyto!(B, buffer, N)
end

"""
    vswap!(A,B)

Vectorized implementation of [`swap!`](@ref).
"""
function vswap!(A, B) end

# NOTE `N` must be a power of 2
function vswap!(A::AbstractVector{T}, B::AbstractVector{T}, ::Val{N}) where {T,N}
    @inbounds for i in 1:N:length(A)
        idx = VecRange{N}(i)
        tmp = A[idx]
        A[idx] = B[idx]
        B[idx] = tmp
    end
end
