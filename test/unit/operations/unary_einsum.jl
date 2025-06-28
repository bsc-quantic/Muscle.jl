using Test
using Muscle: Muscle, Tensor, Index, unary_einsum
using LinearAlgebra

# @testset "axis sum" begin
#     # @test einsum("ijk->jk", A) ≈ dropdims(sum(A, dims=1); dims=1)
#     @testset let A = Tensor(ones(2, 3, 4), [Index(:i), Index(:j), Index(:k)])
#         Ar = unary_einsum(A; dims=[Index(:i)])
#         @test Ar ≈ dropdims(sum(A; dims=1); dims=1)

#         Ar = unary_einsum(A; out=[Index(:j), Index(:k)])
#         @test Ar ≈ dropdims(sum(A; dims=1); dims=1)

#         B = Tensor(zeros(3, 4), [Index(:j), Index(:k)])
#         unary_einsum!(B, A)
#         @test B ≈ dropdims(sum(A; dims=1); dims=1)
#     end

#     # @test einsum("ijk->", A) ≈ fill(sum(A))
#     @testset let A = Tensor(ones(2, 3, 4), [Index(:i), Index(:j), Index(:k)])
#         Ar = unary_einsum(A; dims=inds(A))
#         @test isempty(inds(Ar))
#         @test parent(Ar) ≈ fill(sum(A))

#         Ar = unary_einsum(A; out=Index[])
#         @test isempty(inds(Ar))
#         @test parent(Ar) ≈ fill(sum(A))

#         B = Tensor(zeros())
#         unary_einsum!(B, A)
#         @test parent(B) ≈ fill(sum(A))
#     end
# end

# @testset "diagonal" begin
#     # @test einsum("ii->i", A) ≈ diag(A)
#     @testset let A = Tensor(ones(2, 2), [Index(:i), Index(:i)])
#         Ar = unary_einsum(A; out=[Index(:i)])
#         @test inds(Ar) == [Index(:i)]
#         @test parent(Ar) ≈ ones(2)

#         B = Tensor(zeros(2), [Index(:i)])
#         unary_einsum!(B, A)
#         @test parent(B) ≈ ones(2)
#     end

#     # @test einsum("iji->ij", A) ≈ B
#     @testset let A = Tensor(ones(2, 3, 2), [Index(:i), Index(:j), Index(:i)])
#         Ar = unary_einsum(A; out=[Index(:i), Index(:j)])
#         @test inds(Ar) == [Index(:i), Index(:j)]
#         @test parent(Ar) ≈ ones(2, 3)

#         B = Tensor(zeros(2, 3), [Index(:i), Index(:j)])
#         unary_einsum!(B, A)
#         @test parent(B) ≈ ones(2, 3)
#     end
# end

# @testset "trace" begin
#     # @test einsum("ii->", A) ≈ fill(LinearAlgebra.tr(A))
#     @testset let A = Tensor(ones(2, 2), [Index(:i), Index(:i)])
#         Ar = unary_einsum(A)
#         @test isempty(inds(Ar))
#         @test parent(Ar) ≈ 2ones()

#         Ar = unary_einsum(A; dims=[Index(:i)])
#         @test isempty(inds(Ar))
#         @test parent(Ar) ≈ 2ones()

#         Ar = unary_einsum(A; out=Index[])
#         @test isempty(inds(Ar))
#         @test parent(Ar) ≈ 2ones()

#         B = Tensor(zeros())
#         unary_einsum!(B, A)
#         @test parent(B) ≈ 2ones()
#     end

#     # @test einsum("iji->j", A) ≈ A[1, :, 1] + A[2, :, 2]
#     @testset let A = Tensor(ones(2, 3, 2), [Index(:i), Index(:j), Index(:i)])
#         Ar = unary_einsum(A)
#         @test inds(Ar) == [Index(:j)]
#         @test parent(Ar) ≈ 2ones(3)

#         Ar = unary_einsum(A; dims=[Index(:i)])
#         @test inds(Ar) == [Index(:j)]
#         @test parent(Ar) ≈ 2ones(3)

#         Ar = unary_einsum(A; out=[Index(:j)])
#         @test inds(Ar) == [Index(:j)]
#         @test parent(Ar) ≈ 2ones(3)

#         B = Tensor(zeros(3), [Index(:j)])
#         unary_einsum!(B, A)
#         @test parent(B) ≈ 2ones(3)
#     end
# end
