using Test
using Muscle: Muscle, Tensor, Index

# TODO numeric test with non-random data
# TODO test on NVIDIA GPU

A = Tensor(rand(ComplexF64, 8, 4, 6, 3), [Index(:i), Index(:j), Index(:k), Index(:l)])

# throw if inds_u is not provided
@test_throws ArgumentError Muscle.tensor_eigen_thin(A)

# throw if index is not present
@test_throws ArgumentError Muscle.tensor_eigen_thin(A; inds_u=[Index(:z)])
@test_throws ArgumentError Muscle.tensor_eigen_thin(A; inds_v=[Index(:z)])

# throw if no inds left
@test_throws ArgumentError Muscle.tensor_eigen_thin(A; inds_u=[Index(:i), Index(:j), Index(:k), Index(:l)])
@test_throws ArgumentError Muscle.tensor_eigen_thin(A; inds_v=[Index(:i), Index(:j), Index(:k), Index(:l)])

# throw if chosen virtual index already present
@test_throws ArgumentError Muscle.tensor_eigen_thin(A; inds_u=[Index(:i)], ind_lambda=Index(:j))

# throw if non-square 
@test_throws DimensionMismatch Muscle.tensor_eigen_thin(A; inds_u=[Index(:i), Index(:j)], ind_lambda=Index(:x))

lambdas, U = Muscle.tensor_eigen_thin(A; inds_u=[Index(:i), Index(:l)], ind_lambda=Index(:x))

@test inds(U) == [Index(:i), Index(:l), Index(:x)]
@test inds(lambdas) == [Index(:x)]

@test size(U) == (24, 8, 3)
@test size(lambdas) == (24,)

#= TODO no backend and also this is sketchy
function inv(
             A::Tensor; inds_left=(), inds_right=() #, inplace=false, kwargs...
       )
         inds_left, inds_right = factorinds(inds(A), inds_left, inds_right)
         @argcheck isdisjoint(inds_left, inds_right)
         @argcheck issetequal(inds_left ∪ inds_right, inds(A))

         # permute array
         left_sizes = map(Base.Fix1(size, A), inds_left)
         right_sizes = map(Base.Fix1(size, A), inds_right)
         Amat = permutedims(A, [inds_left..., inds_right...])
         Amat = reshape(parent(Amat), prod(left_sizes), prod(right_sizes))

         Ainv = LinearAlgebra.inv(Amat)

         # tensorify results
         Ainv = Tensor(reshape(Ainv, right_sizes..., left_sizes...), [inds_right; inds_left])

       end
  =#
  @test isapprox(Muscle.binary_einsum(A, U), Muscle.binary_einsum(lambdas, U; dims=Index[]))
