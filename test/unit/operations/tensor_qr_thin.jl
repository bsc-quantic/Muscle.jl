using Test
using Muscle: Muscle, Tensor, Index

# TODO numeric test with non-random data
# TODO test on NVIDIA GPU

A = Tensor(rand(ComplexF64, 2, 4, 6, 8), [Index(:i), Index(:j), Index(:k), Index(:l)])
ind_virtual = Index(:x)

# throw if inds_q is not provided
@test_throws ArgumentError Muscle.tensor_qr_thin(A)

# throw if index is not present
@test_throws ArgumentError Muscle.tensor_qr_thin(A, inds_q=[Index(:z)])
@test_throws ArgumentError Muscle.tensor_qr_thin(A, inds_r=[Index(:z)])

# throw if no inds left
@test_throws ArgumentError Muscle.tensor_qr_thin(A, inds_q=[Index(:i), Index(:j), Index(:k), Index(:l)])
@test_throws ArgumentError Muscle.tensor_qr_thin(A, inds_r=[Index(:i), Index(:j), Index(:k), Index(:l)])

# throw if chosen virtual index already present
@test_throws ArgumentError Muscle.tensor_qr_thin(A, inds_q=[Index(:i)], ind_virtual=Index(:j))

Q, R = Muscle.tensor_qr_thin(A; inds_q=[Index(:i), Index(:j)], ind_virtual)

@test inds(Q) == [Index(:i), Index(:j), Index(:x)]
@test inds(R) == [Index(:x), Index(:k), Index(:l)]

@test size(Q) == (2, 4, 8)
@test size(R) == (8, 6, 8)

@test isapprox(Muscle.binary_einsum(Q, R), A)
