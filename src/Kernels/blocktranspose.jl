using StaticArrays

@doc raw"""
    blocktranspose!(A, B)

```latex
    B = A^\dagger
```
"""
function blocktranspose! end

function __blocktranspose_block_fetch!(A, B)
    @simd for i in eachindex(A, B)
        @inbounds B[i] = A[i]
    end
end

# NOTE `@generated` is used for manual loop unrolling
@generated function __blocktranspose_block_transpose!(B::MMatrix{N,N}) where {N}
    swaps = map(Iterators.filter(splat(>), Iterators.map(Tuple, Iterators.product(1:N, 1:N)))) do (i, j)
        quote
            B[$i, $j], B[$j, $i] = B[$j, $i], B[$i, $j]
        end
    end

    quote
        @inbounds begin
            $(swaps...)
        end
    end
end

# NOTE Apple M1: optimal performance for Float64 at N=16, Float32 at N=32
function blocktranspose!(::Val{N}, A::AbstractMatrix{T}) where {N,T}
    Bₗ = MMatrix{N,N,T}(undef)
    Bᵣ = MMatrix{N,N,T}(undef)

    for bⱼ in 1:N:size(A, 2)
        for bᵢ in 1:N:size(A, 1)
            bᵢ > bⱼ && break

            @inbounds Aₗ = @view A[bᵢ:(bᵢ + N - 1), bⱼ:(bⱼ + N - 1)]
            @inbounds Aᵣ = @view A[bⱼ:(bⱼ + N - 1), bᵢ:(bᵢ + N - 1)]

            # fetch blocks
            __blocktranspose_block_fetch!(Aₗ, Bₗ)
            __blocktranspose_block_fetch!(Aᵣ, Bᵣ)

            # swap & transpose blocks
            __blocktranspose_block_transpose!(Bₗ)
            __blocktranspose_block_transpose!(Bᵣ)

            # write blocks swap
            __blocktranspose_block_fetch!(Bᵣ, Aₗ)
            __blocktranspose_block_fetch!(Bₗ, Aᵣ)
        end
    end
end
