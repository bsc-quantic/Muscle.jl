using StaticArrays

@doc raw"""
    blocktranspose!(A, B)

```latex
    B = A^\dagger
```
"""
function blocktranspose! end

function blocktranspose!(A)
    @simd ivdep for I in eachindex(IndexCartesian(), A)
        (i, j) = Tuple(I)
        if i > j # TODO vector mask?
            @inbounds (A[i, j], A[j, i]) = (A[j, i], A[i, j])
        end
    end
end

# NOTE Apple M1: optimal performance for Float64 at N=16
function blocktranspose!(::Val{N}, A::AbstractMatrix{T}) where {N,T}
    bottom = MMatrix{N,N,T}(undef)
    top = MMatrix{N,N,T}(undef)

    for bj in 1:N:size(A, 1)
        for bi in 1:N:size(A, 2)
            bi > bj && break

            # fetch blocks
            for I in eachindex(IndexCartesian(), bottom, top)
                i, j = Tuple(I)
                @inbounds bottom[i, j] = A[bi+i-1, bj+j-1]
                @inbounds top[i, j] = A[bj+i-1, bi+j-1]
            end

            # transpose blocks
            for I in eachindex(IndexCartesian(), bottom, top)
                i, j = Tuple(I)
                i > j && continue
                @inbounds bottom[i, j], bottom[j, i] = bottom[j, i], bottom[i, j]
                @inbounds top[i, j], top[j, i] = top[j, i], top[i, j]
            end

            # swap blocks
            for I in eachindex(IndexCartesian(), bottom, top)
                i, j = Tuple(I)
                @inbounds A[bi+i-1, bj+j-1] = top[i, j]
                @inbounds A[bj+i-1, bi+j-1] = bottom[i, j]
            end
        end
    end
end

# TODO conj to complex
function blocktranspose!(A, B)
    for (colB, rowA) in zip(eachcol(B), eachrow(A))
        @simd for i in eachindex(colB, rowA)
            @inbounds colB[i] = rowA[i]
        end
    end
end