using StaticArrays: MVector

"""
	swap!(A, B)

Swaps the content of two `StridedArray`s or `SubArray`s.
"""
function swap! end

"""
    naiveswap!(A,B) where {N}

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

