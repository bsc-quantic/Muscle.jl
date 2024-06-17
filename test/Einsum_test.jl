@testset "Einsum" begin
    using LinearAlgebra

    @testset "axis sum" begin
        A = rand(2, 3, 4)
        @test einsum("ijk->jk", A) ≈ dropdims(sum(A, dims=1); dims=1)

        A = rand(2, 3, 4)
        @test einsum("ijk->", A) ≈ fill(sum(A))
    end

    @testset "diagonal" begin
        A = rand(2, 2)
        @test einsum("ii->i", A) ≈ diag(A)

        A = rand(2, 3, 2)
        B = similar(A, 2, 3)
        B[1, :] = A[1, :, 1]
        B[2, :] = A[2, :, 2]
        @test einsum("iji->ij", A) ≈ B
    end

    @testset "trace" begin
        A = rand(2, 2)
        @test einsum("ii->", A) ≈ fill(LinearAlgebra.tr(A))

        A = rand(2, 3, 2)
        @test einsum("iji->j", A) ≈ A[1, :, 1] + A[2, :, 2]
    end

    @testset "matrix multiplication" begin
        A = rand(2, 3)
        B = rand(3, 4)
        @test einsum("ij,jk->ik", A, B) ≈ A * B
    end

    @testset "inner product" begin
        A = rand(3, 4)
        B = rand(4, 3)
        @test einsum("ij,ji->", A, B) ≈ fill(LinearAlgebra.dot(A, B'))
    end

    @testset "outer product" begin
        A = rand(2, 2)
        B = rand(2, 2)
        @test einsum("ij,kl->ijkl", A, B) ≈ permutedims(reshape(kron(A, B), 2, 2, 2, 2), (2, 4, 1, 3))
    end

    @testset "scale" begin
        A = rand(2, 2)
        α = 2.0
        @test einsum(",ij->ij", α, A) ≈ α * A broken = true
        @test einsum("ij,->ij", A, α) ≈ α * A broken = true

        A = rand(2, 2)
        α = fill(2.0)
        @test einsum(",ij->ij", α, A) ≈ α .* A
        @test einsum("ij,->ij", A, α) ≈ α .* A
    end
end
