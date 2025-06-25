using Test
using Muscle: Tensor, Index
using CUDA

@testset "matmul" begin
    A = Tensor(CUDA.ones(2, 3), [Index(:i), Index(:j)])
    B = Tensor(CUDA.ones(3, 4), [Index(:j), Index(:k)])

    C = binary_einsum(A, B)
    @test inds(C) == [Index(:i), Index(:k)]
    @test size(C) == (2, 4)
    @test parent(C) == 3 * CUDA.ones(2, 4)

    # specifying output inds
    C = binary_einsum(A, B; out=[Index(:i), Index(:k)])
    @test inds(C) == [Index(:i), Index(:k)]
    @test size(C) == (2, 4)
    @test parent(C) == 3 * CUDA.ones(2, 4)

    # permuting output inds
    C = binary_einsum(A, B; out=[Index(:k), Index(:i)])
    @test inds(C) == [Index(:k), Index(:i)]
    @test size(C) == (4, 2)
    @test parent(C) == 3 * CUDA.ones(4, 2)

    # specifying contracting inds
    C = binary_einsum(A, B; dims=[Index(:j)])
    @test inds(C) == [Index(:i), Index(:k)]
    @test size(C) == (2, 4)
    @test parent(C) == 3 * CUDA.ones(2, 4)
end

@testset "inner product" begin
    A = Tensor(CUDA.ones(3, 4), [Index(:i), Index(:j)])
    B = Tensor(CUDA.ones(4, 3), [Index(:j), Index(:i)])

    C = binary_einsum(A, B)
    @test isempty(inds(C))
    @test size(C) == ()
    @test parent(C) == fill(12)

    # specifying output inds
    C = binary_einsum(A, B; out=Index[])
    @test isempty(inds(C))
    @test size(C) == ()
    @test parent(C) == fill(12)

    # specifying contracting inds
    C = binary_einsum(A, B; dims=[Index(:i), Index(:j)])
    @test isempty(inds(C))
    @test size(C) == ()
    @test parent(C) == fill(12)

    # permutation contracting inds shouldn't matter
    C = binary_einsum(A, B; dims=[Index(:j), Index(:i)])
    @test isempty(inds(C))
    @test size(C) == ()
    @test parent(C) == fill(12)
end

@testset "outer product" begin
    A = Tensor(CUDA.ones(2, 3), [Index(:i), Index(:j)])
    B = Tensor(CUDA.ones(4, 5), [Index(:k), Index(:l)])

    C = binary_einsum(A, B)
    @test inds(C) == [Index(:i), Index(:j), Index(:k), Index(:l)]
    @test size(C) == (2, 3, 4, 5)
    @test parent(C) == fill(1, 2, 3, 4, 5)

    C = binary_einsum(A, B; out=[Index(:i), Index(:j), Index(:k), Index(:l)])
    @test inds(C) == [Index(:i), Index(:j), Index(:k), Index(:l)]
    @test size(C) == (2, 3, 4, 5)
    @test parent(C) == fill(1, 2, 3, 4, 5)

    # try different output permutations
    C = binary_einsum(A, B; out=[Index(:k), Index(:l), Index(:i), Index(:j)])
    @test inds(C) == [Index(:k), Index(:l), Index(:i), Index(:j)]
    @test size(C) == (4, 5, 2, 3)
    @test parent(C) == fill(1, 4, 5, 2, 3)

    C = binary_einsum(A, B; out=[Index(:l), Index(:k), Index(:j), Index(:i)])
    @test inds(C) == [Index(:l), Index(:k), Index(:j), Index(:i)]
    @test size(C) == (5, 4, 3, 2)
    @test parent(C) == fill(1, 5, 4, 3, 2)

    C = binary_einsum(A, B; out=[Index(:l), Index(:i), Index(:k), Index(:j)])
    @test inds(C) == [Index(:l), Index(:i), Index(:k), Index(:j)]
    @test size(C) == (5, 2, 4, 3)
    @test parent(C) == fill(1, 5, 2, 4, 3)

    C = binary_einsum(A, B; out=[Index(:j), Index(:i), Index(:k), Index(:l)])
    @test inds(C) == [Index(:j), Index(:i), Index(:k), Index(:l)]
    @test size(C) == (3, 2, 4, 5)
    @test parent(C) == fill(1, 3, 2, 4, 5)
end

@testset "scale" begin
    A = Tensor(CUDA.ones(2, 3), [Index(:i), Index(:j)])
    α = Tensor(fill(2.0))

    C = binary_einsum(A, α)
    @test inds(C) == [Index(:i), Index(:j)]
    @test size(C) == (2, 3)
    @test parent(C) == α[] .* parent(A)

    C = binary_einsum(α, A)
    @test inds(C) == [Index(:i), Index(:j)]
    @test size(C) == (2, 3)
    @test parent(C) == α[] .* parent(A)

    C = binary_einsum(A, α; out=[Index(:j), Index(:i)])
    @test inds(C) == [Index(:j), Index(:i)]
    @test size(C) == (3, 2)
    @test parent(C) == α[] .* transpose(parent(A))

    C = binary_einsum(α, A; out=[Index(:j), Index(:i)])
    @test inds(C) == [Index(:j), Index(:i)]
    @test size(C) == (3, 2)
    @test parent(C) == α[] .* transpose(parent(A))
end

@testset "batch matmul" begin
    A = Tensor(CUDA.ones(2, 3, 6), [Index(:i), Index(:j), Index(:batch)])
    B = Tensor(CUDA.ones(3, 4, 6), [Index(:j), Index(:k), Index(:batch)])

    # specifying output inds
    C = binary_einsum(A, B; out=[Index(:i), Index(:k), Index(:batch)])
    @test inds(C) == [Index(:i), Index(:k), Index(:batch)]
    @test size(C) == (2, 4, 6)
    @test parent(C) == 3 * CUDA.ones(2, 4, 6)

    C = binary_einsum(A, B; out=[Index(:k), Index(:i), Index(:batch)])
    @test inds(C) == [Index(:k), Index(:i), Index(:batch)]
    @test size(C) == (4, 2, 6)
    @test parent(C) == 3 * CUDA.ones(4, 2, 6)

    C = binary_einsum(A, B; out=[Index(:batch), Index(:i), Index(:k)])
    @test inds(C) == [Index(:batch), Index(:i), Index(:k)]
    @test size(C) == (6, 2, 4)
    @test parent(C) == 3 * CUDA.ones(6, 2, 4)

    # specifying contracting inds
    C = binary_einsum(A, B; dims=[Index(:j)])
    @test inds(C) == [Index(:i), Index(:k), Index(:batch)] broken = true
    @test size(C) == (2, 4, 6) broken = true
    @test parent(C) == 3 * CUDA.ones(2, 4, 6) broken = true
end

@testset "manual" begin
    @testset "eltype = $T" for T in [Float64, ComplexF64]
        A = Tensor(CUDA.ones(T, 2, 3, 4), Index.([:i, :j, :k]))
        B = Tensor(CUDA.ones(T, 4, 5, 3), Index.([:k, :l, :j]))

        # contraction of all common indices
        C = binary_einsum(A, B; dims=[Index(:j), Index(:k)])

        @test inds(C) == [Index(:i), Index(:l)]
        @test size(C) == (2, 5)
        @test parent(C) ≈ begin
            A_mat = reshape(parent(A), 2, 12)
            B_mat = reshape(permutedims(parent(B), [3, 1, 2]), 12, 5)
            A_mat * B_mat
        end

        # contraction of NOT all common indices
        C = binary_einsum(A, B; dims=[Index(:j)])

        @test inds(C) == [Index(:i), Index(:k), Index(:l)]
        @test size(C) == (2, 4, 5)
        @test parent(C) ≈ begin
            C = zeros(2, 4, 5)
            for i in 1:2, j in 1:3, k in 1:4, l in 1:5
                C[i, k, l] += A[i, j, k] * B[k, l, j]
            end
            C
        end
    end
end
