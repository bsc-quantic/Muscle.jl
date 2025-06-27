using Test
using Muscle
using KrylovKit
using LinearAlgebra

A = rand(ComplexF64, 4, 4)
data = (A + A') / 2 # Make it Hermitian
tensor = Tensor(data, (Index(:i), Index(:j)))

# Perform eigensolve
vals, vecs, info = eigsolve(tensor; left_inds=[Index(:i)], right_inds=[Index(:j)])

@test length(vals) == 4
@test length(vecs) == 4

for vec in vecs
    @test inds(vec) == [Index(:i)]
    @test size(vec) == (4,)
end

# throw if index is not present
@test_throws ArgumentError eigsolve(tensor; left_inds=[Index(:z)])
@test_throws ArgumentError eigsolve(tensor; right_inds=[Index(:z)])

# throw if the resulting matrix is not square
tensor_non_square = Tensor(rand(ComplexF64, 2, 4, 6), (Index(:i), Index(:j), Index(:k)))
@test_throws ArgumentError eigsolve(tensor_non_square; left_inds=[Index(:i), Index(:j)], right_inds=[Index(:k)])
@test_throws ArgumentError eigsolve(tensor_non_square; right_inds=[Index(:j), Index(:k)])

# Convert vecs to matrix form for reconstruction
V_matrix = hcat([reshape(parent(vec), :) for vec in vecs]...)
D_matrix = Diagonal(vals)
reconstructed_matrix = V_matrix * D_matrix * inv(V_matrix)

# Ensure the reconstruction is correct
reconstructed_tensor = Tensor(reconstructed_matrix, (Index(:i), Index(:j)))
@test isapprox(reconstructed_tensor, tensor)

# Test consistency with permuted tensor
vals_perm, vecs_perm, info = eigsolve(tensor; left_inds=[Index(:j)], right_inds=[Index(:i)])

@test length(vals_perm) == 4
@test length(vecs_perm) == 4

# Ensure the eigenvalues are the same
@test isapprox(sort(real.(vals)), sort(real.(vals_perm))) && isapprox(sort(imag.(vals)), sort(imag.(vals_perm)))

V_matrix_perm = hcat([reshape(parent(vec), :) for vec in vecs_perm]...)
D_matrix_perm = Diagonal(vals)
reconstructed_matrix_perm = V_matrix_perm * D_matrix_perm * inv(V_matrix_perm)

# Ensure the reconstruction is correct
reconstructed_tensor_perm = Tensor(reconstructed_matrix_perm, (Index(:j), Index(:i)))
@test isapprox(reconstructed_tensor_perm, transpose(tensor))

@test parent(reconstructed_tensor) ≈ parent(transpose(reconstructed_tensor_perm))

@testset "Lanczos" begin
    @test_throws ArgumentError eigsolve(
        tensor,
        Tensor(rand(ComplexF64, 4), [Index(:j)]),
        1,
        :SR,
        Lanczos(; krylovdim=2, tol=1e-16);
        left_inds=[Index(:i)],
        right_inds=[Index(:j)],
    )

    vals_lanczos, vecs_lanczos, info = eigsolve(
        tensor,
        Tensor(rand(ComplexF64, 4), [Index(:i)]),
        1,
        :SR,
        Lanczos(; krylovdim=2, tol=1e-16);
        left_inds=[Index(:i)],
        right_inds=[Index(:j)],
    )

    @test length(vals_lanczos) == 1
    @test length(vecs_lanczos) == 1

    @test minimum(vals) ≈ first(vals_lanczos)
end

A = rand(ComplexF64, 4, 4)
data = (A + A') / 2 # Make it Hermitian
tensor = Tensor(reshape(data, 2, 2, 2, 2), (Index(:i), Index(:j), Index(:k), Index(:l)))

vals, vecs, info = eigsolve(tensor; left_inds=[Index(:i), Index(:j)], right_inds=[Index(:k), Index(:l)])

# Convert vecs to matrix form for reconstruction
V_matrix = hcat([reshape(parent(vec), :) for vec in vecs]...)
D_matrix = Diagonal(vals)
reconstructed_matrix = V_matrix * D_matrix * inv(V_matrix)

@test isapprox(reconstructed_matrix, data)
