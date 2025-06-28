using Test
using Muscle
using Muscle: BackendStrided
using Strided
using StridedViews

@testset "matmul" begin
    A = Tensor(ones(2, 3), [Index(:i), Index(:j)])
    B = Tensor(ones(3, 4), [Index(:j), Index(:k)])

    # specifying output inds
    C = binary_einsum(BackendStrided(), [Index(:i), Index(:k)], A, B)
    @test inds(C) == [Index(:i), Index(:k)]
    @test size(C) == (2, 4)
    @test parent(C) == 3 * ones(2, 4)

    # permuting output inds
    C = binary_einsum(BackendStrided(), [Index(:k), Index(:i)], A, B)
    @test inds(C) == [Index(:k), Index(:i)]
    @test size(C) == (4, 2)
    @test parent(C) == 3 * ones(4, 2)
end

@testset "inner product" begin
    A = Tensor(ones(3, 4), [Index(:i), Index(:j)])
    B = Tensor(ones(4, 3), [Index(:j), Index(:i)])

    # specifying output inds
    C = binary_einsum(BackendStrided(), Index[], A, B)
    @test isempty(inds(C))
    @test size(C) == ()
    @test parent(C) == fill(12)
end

@testset "outer product" begin
    A = Tensor(ones(2, 3), [Index(:i), Index(:j)])
    B = Tensor(ones(4, 5), [Index(:k), Index(:l)])

    C = binary_einsum(BackendStrided(), [Index(:i), Index(:j), Index(:k), Index(:l)], A, B)
    @test inds(C) == [Index(:i), Index(:j), Index(:k), Index(:l)]
    @test size(C) == (2, 3, 4, 5)
    @test parent(C) == fill(1, 2, 3, 4, 5)

    # try different output permutations
    C = binary_einsum(BackendStrided(), [Index(:k), Index(:l), Index(:i), Index(:j)], A, B)
    @test inds(C) == [Index(:k), Index(:l), Index(:i), Index(:j)]
    @test size(C) == (4, 5, 2, 3)
    @test parent(C) == fill(1, 4, 5, 2, 3)

    C = binary_einsum(BackendStrided(), [Index(:l), Index(:k), Index(:j), Index(:i)], A, B)
    @test inds(C) == [Index(:l), Index(:k), Index(:j), Index(:i)]
    @test size(C) == (5, 4, 3, 2)
    @test parent(C) == fill(1, 5, 4, 3, 2)

    C = binary_einsum(BackendStrided(), [Index(:l), Index(:i), Index(:k), Index(:j)], A, B)
    @test inds(C) == [Index(:l), Index(:i), Index(:k), Index(:j)]
    @test size(C) == (5, 2, 4, 3)
    @test parent(C) == fill(1, 5, 2, 4, 3)

    C = binary_einsum(BackendStrided(), [Index(:j), Index(:i), Index(:k), Index(:l)], A, B)
    @test inds(C) == [Index(:j), Index(:i), Index(:k), Index(:l)]
    @test size(C) == (3, 2, 4, 5)
    @test parent(C) == fill(1, 3, 2, 4, 5)
end

@testset "scale" begin
    A = Tensor(ones(2, 3), [Index(:i), Index(:j)])
    α = Tensor(fill(2.0))

    C = binary_einsum(BackendStrided(), inds(A), A, α)
    @test inds(C) == [Index(:i), Index(:j)]
    @test size(C) == (2, 3)
    @test parent(C) == α[] .* parent(A)

    C = binary_einsum(BackendStrided(), inds(A), α, A)
    @test inds(C) == [Index(:i), Index(:j)]
    @test size(C) == (2, 3)
    @test parent(C) == α[] .* parent(A)

    C = binary_einsum(BackendStrided(), [Index(:j), Index(:i)], A, α)
    @test inds(C) == [Index(:j), Index(:i)]
    @test size(C) == (3, 2)
    @test parent(C) == α[] .* transpose(parent(A))

    C = binary_einsum(BackendStrided(), [Index(:j), Index(:i)], α, A)
    @test inds(C) == [Index(:j), Index(:i)]
    @test size(C) == (3, 2)
    @test parent(C) == α[] .* transpose(parent(A))
end

# hyperindices not supported on BackendStrided
@testset "batch matmul" begin
    A = Tensor(ones(2, 3, 6), [Index(:i), Index(:j), Index(:batch)])
    B = Tensor(ones(3, 4, 6), [Index(:j), Index(:k), Index(:batch)])

    # specifying output inds
    @test_throws ArgumentError binary_einsum(BackendStrided(), [Index(:i), Index(:k), Index(:batch)], A, B)
    @test_throws ArgumentError binary_einsum(BackendStrided(), [Index(:k), Index(:i), Index(:batch)], A, B)
    @test_throws ArgumentError binary_einsum(BackendStrided(), [Index(:batch), Index(:i), Index(:k)], A, B)
end

@testset "manual" begin
    @testset "eltype = $T" for T in [Float64, ComplexF64]
        A = Tensor(ones(T, 2, 3, 4), Index.([:i, :j, :k]))
        B = Tensor(ones(T, 4, 5, 3), Index.([:k, :l, :j]))

        # contraction of all common indices
        C = binary_einsum(BackendStrided(), [Index(:i), Index(:l)], A, B)

        @test inds(C) == [Index(:i), Index(:l)]
        @test size(C) == (2, 5)
        @test parent(C) ≈ begin
            A_mat = reshape(parent(A), 2, 12)
            B_mat = reshape(permutedims(parent(B), [3, 1, 2]), 12, 5)
            A_mat * B_mat
        end

        # contraction of NOT all common indices
        # hyperindices not supported on BackendStrided
        @test_throws ArgumentError binary_einsum(BackendStrided(), [Index(:i), Index(:k), Index(:l)], A, B)
    end
end
