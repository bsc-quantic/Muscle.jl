using Test
using Muscle: Muscle, Tensor, Index

# TODO numeric test with non-random data
# TODO test on NVIDIA GPU

A = Tensor(rand(ComplexF64, 2, 4, 6, 8), [Index(:i), Index(:j), Index(:k), Index(:l)])

# throw if inds_u is not provided
@test_throws ArgumentError Muscle.tensor_svd_thin(A)

# throw if index is not present
@test_throws ArgumentError Muscle.tensor_svd_thin(A; inds_u=[Index(:z)])
@test_throws ArgumentError Muscle.tensor_svd_thin(A; inds_v=[Index(:z)])

# throw if no inds left
@test_throws ArgumentError Muscle.tensor_svd_thin(A; inds_u=[Index(:i), Index(:j), Index(:k), Index(:l)])
@test_throws ArgumentError Muscle.tensor_svd_thin(A; inds_v=[Index(:i), Index(:j), Index(:k), Index(:l)])

# throw if chosen virtual index already present
@test_throws ArgumentError Muscle.tensor_svd_thin(A; inds_u=[Index(:i)], ind_s=Index(:j))

U, s, V = Muscle.tensor_svd_thin(A; inds_u=[Index(:i), Index(:j)], ind_s=Index(:x))

@test inds(U) == [Index(:i), Index(:j), Index(:x)]
@test inds(s) == [Index(:x)]
@test inds(V) == [Index(:k), Index(:l), Index(:x)]

@test size(U) == (2, 4, 8)
@test size(s) == (8,)
@test size(V) == (6, 8, 8)

@test isapprox(Muscle.binary_einsum(Muscle.hadamard(U, s), V), A)
