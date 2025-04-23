using Test
using Muscle: Tensor, Index, Operations

# TODO test on NVIDIA GPU

@testset "matmul" begin
    A = Tensor(ones(2, 3), [Index(:i), Index(:j)])
    B = Tensor(ones(3, 4), [Index(:j), Index(:k)])

    C = Operations.binary_einsum(A, B)
    @test inds(C) == [Index(:i), Index(:k)]
    @test size(C) == (2, 4)
    @test parent(C) == 3 * ones(2, 4)

    # specifying output inds
    C = Operations.binary_einsum(A, B; out=[Index(:i), Index(:k)])
    @test inds(C) == [Index(:i), Index(:k)]
    @test size(C) == (2, 4)
    @test parent(C) == 3 * ones(2, 4)

    # permuting output inds
    C = Operations.binary_einsum(A, B; out=[Index(:k), Index(:i)])
    @test inds(C) == [Index(:k), Index(:i)]
    @test size(C) == (4, 2)
    @test parent(C) == 3 * ones(4, 2)

    # specifying contracting inds
    C = Operations.binary_einsum(A, B; dims=[Index(:j)])
    @test inds(C) == [Index(:i), Index(:k)]
    @test size(C) == (2, 4)
    @test parent(C) == 3 * ones(2, 4)
end

@testset "inner product" begin
    A = Tensor(ones(3, 4), [Index(:i), Index(:j)])
    B = Tensor(ones(4, 3), [Index(:j), Index(:i)])

    C = Operations.binary_einsum(A, B)
    @test isempty(inds(C))
    @test size(C) == ()
    @test parent(C) == fill(12)

    # specifying output inds
    C = Operations.binary_einsum(A, B; out=Index[])
    @test isempty(inds(C))
    @test size(C) == ()
    @test parent(C) == fill(12)

    # specifying contracting inds
    C = Operations.binary_einsum(A, B; dims=[Index(:i), Index(:j)])
    @test isempty(inds(C))
    @test size(C) == ()
    @test parent(C) == fill(12)

    # permutation contracting inds shouldn't matter
    C = Operations.binary_einsum(A, B; dims=[Index(:j), Index(:i)])
    @test isempty(inds(C))
    @test size(C) == ()
    @test parent(C) == fill(12)
end

@testset "outer product" begin
    A = Tensor(ones(2, 3), [Index(:i), Index(:j)])
    B = Tensor(ones(4, 5), [Index(:k), Index(:l)])

    C = Operations.binary_einsum(A, B)
    @test inds(C) == [Index(:i), Index(:j), Index(:k), Index(:l)]
    @test size(C) == (2, 3, 4, 5)
    @test parent(C) == fill(1, 2, 3, 4, 5)

    C = Operations.binary_einsum(A, B; out=[Index(:i), Index(:j), Index(:k), Index(:l)])
    @test inds(C) == [Index(:i), Index(:j), Index(:k), Index(:l)]
    @test size(C) == (2, 3, 4, 5)
    @test parent(C) == fill(1, 2, 3, 4, 5)

    # try different output permutations
    C = Operations.binary_einsum(A, B; out=[Index(:k), Index(:l), Index(:i), Index(:j)])
    @test inds(C) == [Index(:k), Index(:l), Index(:i), Index(:j)]
    @test size(C) == (4, 5, 2, 3)
    @test parent(C) == fill(1, 4, 5, 2, 3)

    C = Operations.binary_einsum(A, B; out=[Index(:l), Index(:k), Index(:j), Index(:i)])
    @test inds(C) == [Index(:l), Index(:k), Index(:j), Index(:i)]
    @test size(C) == (5, 4, 3, 2)
    @test parent(C) == fill(1, 5, 4, 3, 2)

    C = Operations.binary_einsum(A, B; out=[Index(:l), Index(:i), Index(:k), Index(:j)])
    @test inds(C) == [Index(:l), Index(:i), Index(:k), Index(:j)]
    @test size(C) == (5, 2, 4, 3)
    @test parent(C) == fill(1, 5, 2, 4, 3)

    C = Operations.binary_einsum(A, B; out=[Index(:j), Index(:i), Index(:k), Index(:l)])
    @test inds(C) == [Index(:j), Index(:i), Index(:k), Index(:l)]
    @test size(C) == (3, 2, 4, 5)
    @test parent(C) == fill(1, 3, 2, 4, 5)
end

@testset "scale" begin
    A = Tensor(ones(2, 3), [Index(:i), Index(:j)])
    α = Tensor(fill(2.0))

    C = Operations.binary_einsum(A, α)
    @test inds(C) == [Index(:i), Index(:j)]
    @test size(C) == (2, 3)
    @test parent(C) == α[] .* parent(A)

    C = Operations.binary_einsum(α, A)
    @test inds(C) == [Index(:i), Index(:j)]
    @test size(C) == (2, 3)
    @test parent(C) == α[] .* parent(A)

    C = Operations.binary_einsum(A, α; out=[Index(:j), Index(:i)])
    @test inds(C) == [Index(:j), Index(:i)]
    @test size(C) == (2, 3)
    @test parent(C) == α[] .* parent(A)

    C = Operations.binary_einsum(α, A; out=[Index(:j), Index(:i)])
    @test inds(C) == [Index(:j), Index(:i)]
    @test size(C) == (2, 3)
    @test parent(C) == α[] .* parent(A)
end

@testset "batch matmul" begin
    A = Tensor(ones(2, 3, 6), [Index(:i), Index(:j), Index(:batch)])
    B = Tensor(ones(3, 4, 6), [Index(:j), Index(:k), Index(:batch)])

    # specifying output inds
    C = Operations.binary_einsum(A, B; out=[Index(:i), Index(:k), Index(:batch)])
    @test inds(C) == [Index(:i), Index(:k), Index(:batch)]
    @test size(C) == (2, 4, 6)
    @test parent(C) == 3 * ones(2, 4, 6)

    C = Operations.binary_einsum(A, B; out=[Index(:k), Index(:i), Index(:batch)])
    @test inds(C) == [Index(:k), Index(:i), Index(:batch)]
    @test size(C) == (4, 2, 6)
    @test parent(C) == 3 * ones(4, 2, 6)

    C = Operations.binary_einsum(A, B; out=[Index(:batch), Index(:i), Index(:k)])
    @test inds(C) == [Index(:batch), Index(:i), Index(:k)]
    @test size(C) == (6, 2, 4)
    @test parent(C) == 3 * ones(6, 2, 4)

    # specifying contracting inds
    C = Operations.binary_einsum(A, B; dims=[Index(:j)])
    @test inds(C) == [Index(:i), Index(:k), Index(:batch)] broken = true
    @test size(C) == (2, 4, 6)
    @test parent(C) == 3 * ones(2, 4, 6) broken = true
end
