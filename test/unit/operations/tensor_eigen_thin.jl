using Test
using Muscle: Muscle, Tensor, Index
using LinearAlgebra

A = rand(4, 4)
L, U = eigen(A)
A * U ≈ U * Diagonal(L)

# TODO numeric test with non-random data
# TODO test on NVIDIA GPU

Ainds = Index(:l1), Index(:l2), Index(:r1), Index(:r2)
A = Tensor(rand(ComplexF64, 8, 4, 8, 4), Ainds)

# throw if inds_u is not provided
@test_throws ArgumentError Muscle.tensor_eigen_thin(A)

# throw if index is not present
@test_throws ArgumentError Muscle.tensor_eigen_thin(A; inds_u=[Index(:z)])
@test_throws ArgumentError Muscle.tensor_eigen_thin(A; inds_uinv=[Index(:z)])

# throw if no inds left
@test_throws ArgumentError Muscle.tensor_eigen_thin(A; inds_u=[Index(:l1), Index(:l2), Index(:r1), Index(:r2)])
@test_throws ArgumentError Muscle.tensor_eigen_thin(A; inds_uinv=[Index(:l1), Index(:l2), Index(:r1), Index(:r2)])

# throw if non-square 
@test_throws DimensionMismatch Muscle.tensor_eigen_thin(A; inds_u=[Index(:l1), Index(:r1)], ind_lambda=Index(:x))

#Now the actual thing
lambdas, U = Muscle.tensor_eigen_thin(A; inds_u=[Index(:l1), Index(:l2)], ind_lambda=Index(:lambda))

@test Muscle.inds(U) == [Index(:l1), Index(:l2), Index(:lambda)]
@test Muscle.inds(lambdas) == [Index(:lambda)]

@test size(U) == (8, 4, 32)
@test size(lambdas) == (32,)

# Test AU ≈ UΛ  (right eigenvectors)
Ut = replace(U, Index(:l1) => Index(:r1), Index(:l2) => Index(:r2))
AU = Muscle.binary_einsum(A, Ut)
UL = Muscle.hadamard(U, lambdas)
@test isapprox(AU, UL)
