using Test
using Muscle
using LinearAlgebra

# TODO numeric test with non-random data
# TODO test on NVIDIA GPU

A = Tensor(rand(ComplexF64, 2, 4, 6, 8), [Index(:i), Index(:j), Index(:k), Index(:l)])

# throw if inds_u is not provided
@test_throws ArgumentError Muscle.tensor_svd_trunc(A)

# throw if index is not present
@test_throws ArgumentError Muscle.tensor_svd_trunc(A; inds_u=[Index(:z)])
@test_throws ArgumentError Muscle.tensor_svd_trunc(A; inds_v=[Index(:z)])

# throw if no inds left
@test_throws ArgumentError Muscle.tensor_svd_trunc(A; inds_u=[Index(:i), Index(:j), Index(:k), Index(:l)])
@test_throws ArgumentError Muscle.tensor_svd_trunc(A; inds_v=[Index(:i), Index(:j), Index(:k), Index(:l)])

# throw if chosen virtual index already present
@test_throws ArgumentError Muscle.tensor_svd_trunc(A; inds_u=[Index(:i)], ind_s=Index(:j))

U, s, Vt = Muscle.tensor_svd_trunc(A; inds_u=[Index(:i), Index(:j)], ind_s=Index(:x))

@test inds(U) == [Index(:i), Index(:j), Index(:x)]
@test inds(s) == [Index(:x)]
@test inds(Vt) == [Index(:k), Index(:l), Index(:x)]

@test size(U) == (2, 4, 8)
@test size(s) == (8,)
@test size(Vt) == (6, 8, 8)

@test isapprox(Muscle.binary_einsum(Muscle.hadamard(U, s), Vt), A)
@test isisometry(U, Index(:x))
@test isisometry(Vt, Index(:x))

A = Tensor(rand(ComplexF64, 200, 100), [Index(:i), Index(:j)])
U, sfull, Vt = Muscle.tensor_svd_trunc(A; inds_u=Index(:i), ind_s=Index(:x))
@test isapprox(Muscle.binary_einsum(Muscle.hadamard(U, sfull), Vt), A)

U, s, Vt = Muscle.tensor_svd_trunc(A; inds_u=Index(:i), ind_s=Index(:x), threshold=1e-3)
U, s, Vt = Muscle.tensor_svd_trunc(A; inds_u=Index(:i), ind_s=Index(:x), maxdim=20)

# check that norm error = sum of discarded SV^2
U, s, Vt = Muscle.tensor_svd_trunc(A; inds_u=Index(:i), ind_s=Index(:x), threshold=1e-3, maxdim=80)
@test norm(Muscle.binary_einsum(Muscle.hadamard(U, s), Vt) - A) ≈ norm(sfull[(length(s) + 1):end])
