using Test
using Muscle
using Muscle: Tensor, Index, binary_einsum
using OMEinsum

@testset "unary_einsum" begin
    @testset "axis sum" begin
        # @test einsum("ijk->jk", A) ≈ dropdims(sum(A, dims=1); dims=1)
        @testset let A = Tensor(ones(2, 3, 4), [Index(:i), Index(:j), Index(:k)])
            Ar = unary_einsum(A; dims=[Index(:i)])
            @test Ar ≈ dropdims(sum(A; dims=1); dims=1)

            Ar = unary_einsum(A; out=[Index(:j), Index(:k)])
            @test Ar ≈ dropdims(sum(A; dims=1); dims=1)

            B = Tensor(zeros(3, 4), [Index(:j), Index(:k)])
            unary_einsum!(B, A)
            @test B ≈ dropdims(sum(A; dims=1); dims=1)
        end

        # @test einsum("ijk->", A) ≈ fill(sum(A))
        @testset let A = Tensor(ones(2, 3, 4), [Index(:i), Index(:j), Index(:k)])
            Ar = unary_einsum(A; dims=inds(A))
            @test isempty(inds(Ar))
            @test parent(Ar) ≈ fill(sum(A))

            Ar = unary_einsum(A; out=Index[])
            @test isempty(inds(Ar))
            @test parent(Ar) ≈ fill(sum(A))

            B = Tensor(zeros())
            unary_einsum!(B, A)
            @test parent(B) ≈ fill(sum(A))
        end
    end

    @testset "diagonal" begin
        # @test einsum("ii->i", A) ≈ diag(A)
        @testset let A = Tensor(ones(2, 2), [Index(:i), Index(:i)])
            Ar = unary_einsum(A; out=[Index(:i)])
            @test inds(Ar) == [Index(:i)]
            @test parent(Ar) ≈ ones(2)

            B = Tensor(zeros(2), [Index(:i)])
            unary_einsum!(B, A)
            @test parent(B) ≈ ones(2)
        end

        # @test einsum("iji->ij", A) ≈ B
        @testset let A = Tensor(ones(2, 3, 2), [Index(:i), Index(:j), Index(:i)])
            Ar = unary_einsum(A; out=[Index(:i), Index(:j)])
            @test inds(Ar) == [Index(:i), Index(:j)]
            @test parent(Ar) ≈ ones(2, 3)

            B = Tensor(zeros(2, 3), [Index(:i), Index(:j)])
            unary_einsum!(B, A)
            @test parent(B) ≈ ones(2, 3)
        end
    end

    @testset "trace" begin
        # @test einsum("ii->", A) ≈ fill(LinearAlgebra.tr(A))
        @testset let A = Tensor(ones(2, 2), [Index(:i), Index(:i)])
            Ar = unary_einsum(A)
            @test isempty(inds(Ar))
            @test parent(Ar) ≈ 2ones()

            Ar = unary_einsum(A; dims=[Index(:i)])
            @test isempty(inds(Ar))
            @test parent(Ar) ≈ 2ones()

            Ar = unary_einsum(A; out=Index[])
            @test isempty(inds(Ar))
            @test parent(Ar) ≈ 2ones()

            B = Tensor(zeros())
            unary_einsum!(B, A)
            @test parent(B) ≈ 2ones()
        end

        # @test einsum("iji->j", A) ≈ A[1, :, 1] + A[2, :, 2]
        @testset let A = Tensor(ones(2, 3, 2), [Index(:i), Index(:j), Index(:i)])
            Ar = unary_einsum(A)
            @test inds(Ar) == [Index(:j)]
            @test parent(Ar) ≈ 2ones(3)

            Ar = unary_einsum(A; dims=[Index(:i)])
            @test inds(Ar) == [Index(:j)]
            @test parent(Ar) ≈ 2ones(3)

            Ar = unary_einsum(A; out=[Index(:j)])
            @test inds(Ar) == [Index(:j)]
            @test parent(Ar) ≈ 2ones(3)

            B = Tensor(zeros(3), [Index(:j)])
            unary_einsum!(B, A)
            @test parent(B) ≈ 2ones(3)
        end
    end
end

@testset "binary_einsum" begin
    @testset "matmul" begin
        A = Tensor(ones(2, 3), [Index(:i), Index(:j)])
        B = Tensor(ones(3, 4), [Index(:j), Index(:k)])

        # specifying output inds
        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:i), Index(:k)], A, B)
        @test inds(C) == [Index(:i), Index(:k)]
        @test size(C) == (2, 4)
        @test parent(C) == 3 * ones(2, 4)

        # permuting output inds
        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:k), Index(:i)], A, B)
        @test inds(C) == [Index(:k), Index(:i)]
        @test size(C) == (4, 2)
        @test parent(C) == 3 * ones(4, 2)
    end

    @testset "inner product" begin
        A = Tensor(ones(3, 4), [Index(:i), Index(:j)])
        B = Tensor(ones(4, 3), [Index(:j), Index(:i)])

        # specifying output inds
        C = binary_einsum(Muscle.BackendOMEinsum(), Index[], A, B)
        @test isempty(inds(C))
        @test size(C) == ()
        @test parent(C) == fill(12)
    end

    @testset "outer product" begin
        A = Tensor(ones(2, 3), [Index(:i), Index(:j)])
        B = Tensor(ones(4, 5), [Index(:k), Index(:l)])

        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:i), Index(:j), Index(:k), Index(:l)], A, B)
        @test inds(C) == [Index(:i), Index(:j), Index(:k), Index(:l)]
        @test size(C) == (2, 3, 4, 5)
        @test parent(C) == fill(1, 2, 3, 4, 5)

        # try different output permutations
        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:k), Index(:l), Index(:i), Index(:j)], A, B)
        @test inds(C) == [Index(:k), Index(:l), Index(:i), Index(:j)]
        @test size(C) == (4, 5, 2, 3)
        @test parent(C) == fill(1, 4, 5, 2, 3)

        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:l), Index(:k), Index(:j), Index(:i)], A, B)
        @test inds(C) == [Index(:l), Index(:k), Index(:j), Index(:i)]
        @test size(C) == (5, 4, 3, 2)
        @test parent(C) == fill(1, 5, 4, 3, 2)

        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:l), Index(:i), Index(:k), Index(:j)], A, B)
        @test inds(C) == [Index(:l), Index(:i), Index(:k), Index(:j)]
        @test size(C) == (5, 2, 4, 3)
        @test parent(C) == fill(1, 5, 2, 4, 3)

        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:j), Index(:i), Index(:k), Index(:l)], A, B)
        @test inds(C) == [Index(:j), Index(:i), Index(:k), Index(:l)]
        @test size(C) == (3, 2, 4, 5)
        @test parent(C) == fill(1, 3, 2, 4, 5)
    end

    @testset "scale" begin
        # ones() defaults to float32 but fill() to 64 ? and binary einsum complains when mixed 32/64
        A = Tensor(ones(ComplexF64, 2, 3), [Index(:i), Index(:j)])
        α = Tensor(fill(2.0))

        C = binary_einsum(Muscle.BackendOMEinsum(), inds(A), A, α)
        @test inds(C) == [Index(:i), Index(:j)]
        @test size(C) == (2, 3)
        @test parent(C) == α[] .* parent(A)

        C = binary_einsum(Muscle.BackendOMEinsum(), inds(A), α, A)
        @test inds(C) == [Index(:i), Index(:j)]
        @test size(C) == (2, 3)
        @test parent(C) == α[] .* parent(A)

        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:j), Index(:i)], A, α)
        @test inds(C) == [Index(:j), Index(:i)]
        @test size(C) == (3, 2)
        @test parent(C) == α[] .* transpose(parent(A))

        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:j), Index(:i)], α, A)
        @test inds(C) == [Index(:j), Index(:i)]
        @test size(C) == (3, 2)
        @test parent(C) == α[] .* transpose(parent(A))
    end

    @testset "batch matmul" begin
        A = Tensor(ones(2, 3, 6), [Index(:i), Index(:j), Index(:batch)])
        B = Tensor(ones(3, 4, 6), [Index(:j), Index(:k), Index(:batch)])

        # specifying output inds
        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:i), Index(:k), Index(:batch)], A, B)
        @test inds(C) == [Index(:i), Index(:k), Index(:batch)]
        @test size(C) == (2, 4, 6)
        @test parent(C) == 3 * ones(2, 4, 6)

        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:k), Index(:i), Index(:batch)], A, B)
        @test inds(C) == [Index(:k), Index(:i), Index(:batch)]
        @test size(C) == (4, 2, 6)
        @test parent(C) == 3 * ones(4, 2, 6)

        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:batch), Index(:i), Index(:k)], A, B)
        @test inds(C) == [Index(:batch), Index(:i), Index(:k)]
        @test size(C) == (6, 2, 4)
        @test parent(C) == 3 * ones(6, 2, 4)
    end

    @testset "manual" begin
        @testset "eltype = $T" for T in [Float64, ComplexF64]
            A = Tensor(ones(T, 2, 3, 4), Index.([:i, :j, :k]))
            B = Tensor(ones(T, 4, 5, 3), Index.([:k, :l, :j]))

            # contraction of all common indices
            C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:i), Index(:l)], A, B)

            @test inds(C) == [Index(:i), Index(:l)]
            @test size(C) == (2, 5)
            @test parent(C) ≈ begin
                A_mat = reshape(parent(A), 2, 12)
                B_mat = reshape(permutedims(parent(B), [3, 1, 2]), 12, 5)
                A_mat * B_mat
            end

            # contraction of NOT all common indices
            C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:i), Index(:k), Index(:l)], A, B)

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
end
