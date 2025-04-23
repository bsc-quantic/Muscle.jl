using Test
using Muscle: Tensor, Index, Operations

A = Tensor(rand(ComplexF64, 2, 4, 6, 8), [Index(:i), Index(:j), Index(:k), Index(:l)])

# throw if left_inds is not provided
@test_throws ArgumentError Operations.tensor_svd_thin(A)

# throw if index is not present
@test_throws ArgumentError Operations.tensor_svd_thin(A; left_inds=[Index(:z)])
@test_throws ArgumentError Operations.tensor_svd_thin(A; right_inds=[Index(:z)])

# throw if no inds left
@test_throws ArgumentError Operations.tensor_svd_thin(A; left_inds=[Index(:i), Index(:j), Index(:k), Index(:l)])
@test_throws ArgumentError Operations.tensor_svd_thin(A; right_inds=[Index(:i), Index(:j), Index(:k), Index(:l)])

# throw if chosen virtual index already present
@test_throws ArgumentError Operations.tensor_svd_thin(A; left_inds=[Index(:i)], virtualind=Index(:j))

U, s, V = Operations.tensor_svd_thin(A; left_inds=[Index(:i), Index(:j)], virtualind=Index(:x))

@test inds(U) == [Index(:i), Index(:j), Index(:x)]
@test inds(s) == [Index(:x)]
@test inds(V) == [Index(:k), Index(:l), Index(:x)]

@test size(U) == (2, 4, 8)
@test size(s) == (8,)
@test size(V) == (6, 8, 8)

@test isapprox(Operations.binary_einsum(Operations.binary_einsum(U, s; dims=Index[]), V), A)
