using Test
using Muscle: Tensor, Index, Operations
using LinearAlgebra: LinearAlgebra

# TODO test on NVIDIA GPU

# these two tensors represent a MPS factorization of the |00> + |01> + |10> + |11> state with norm=√2
Γa = Tensor([1.0 0.0; 0.0 1.0], [Index((; site=1, cut=1)), Index((; bond=(1, 2)))])
Γb = Tensor([1.0 0.0; 0.0 1.0], [Index((; site=2, cut=1)), Index((; bond=(1, 2)))])

op_identity = Tensor(
    reshape(Array{Float64}(LinearAlgebra.I(4)), 2, 2, 2, 2),
    [
        Index((; site=1, cut=1)),
        Index((; site=2, cut=1)),
        Index((; site=1, cut=2)),
        Index((; site=2, cut=2)),
    ],
)

op_cx = Tensor(
    reshape([
            1.0 0.0 0.0 0.0;
            0.0 1.0 0.0 0.0;
            0.0 0.0 0.0 1.0;
            0.0 0.0 1.0 0.0
        ], 2, 2, 2, 2),
    [
        Index((; site=1, cut=1)),
        Index((; site=2, cut=1)),
        Index((; site=1, cut=2)),
        Index((; site=2, cut=2)),
    ],
)

@testset "apply identity" begin
    U, s, V = Operations.simple_update(
        Γa, Index((; site=1, cut=1)),
        Γb, Index((; site=2, cut=1)),
        Index((; bond=(1, 2))),
        op_identity, Index((; site=1, cut=2)), Index((; site=2, cut=2))
    )

    @test U ≈ Γa
    @test V ≈ Γb
    @test s == Tensor([1.0, 1.0], [Index((; bond=(1, 2)))])

    ψ = Operations.binary_einsum(Operations.binary_einsum(U, s; dims=Index[]), V)
    @test Operations.binary_einsum(ψ, conj(ψ)) |> only ≈ 2.0
end

@testset "apply cx" begin
    U, s, V = Operations.simple_update(
        Γa, Index((; site=1, cut=1)),
        Γb, Index((; site=2, cut=1)),
        Index((; bond=(1, 2))),
        op_cx, Index((; site=1, cut=2)), Index((; site=2, cut=2))
    )

    @test U ≈ Γa
    @test V ≈ Tensor(1 / √2 * [1 -1; 1 1], [Index((; site=2, cut=1)), Index((; bond=(1, 2)))])
    @test s == Tensor([√2, 0.0], [Index((; bond=(1, 2)))])

    ψ = Operations.binary_einsum(Operations.binary_einsum(U, s; dims=Index[]), V)
    @test Operations.binary_einsum(ψ, conj(ψ)) |> only ≈ 2.0
end

@testset "apply identity, normalize" begin
    U, s, V = Operations.simple_update(
        Γa, Index((; site=1, cut=1)),
        Γb, Index((; site=2, cut=1)),
        Index((; bond=(1, 2))),
        op_identity, Index((; site=1, cut=2)), Index((; site=2, cut=2));
        normalize=true,
    )

    @test U ≈ Γa
    @test V ≈ Γb
    @test s ≈ Tensor([1 / √2, 1 / √2], [Index((; bond=(1, 2)))])

    ψ = Operations.binary_einsum(Operations.binary_einsum(U, s; dims=Index[]), V)
    @test Operations.binary_einsum(ψ, conj(ψ)) |> only ≈ 1.0
end

@testset "apply cx, normalize" begin
    U, s, V = Operations.simple_update(
        Γa, Index((; site=1, cut=1)),
        Γb, Index((; site=2, cut=1)),
        Index((; bond=(1, 2))),
        op_cx, Index((; site=1, cut=2)), Index((; site=2, cut=2));
        normalize=true
    )

    @test U ≈ Γa
    @test V ≈ Tensor(1 / √2 * [1 -1; 1 1], [Index((; site=2, cut=1)), Index((; bond=(1, 2)))])
    @test s == Tensor([1.0, 0.0], [Index((; bond=(1, 2)))])

    ψ = Operations.binary_einsum(Operations.binary_einsum(U, s; dims=Index[]), V)
    @test Operations.binary_einsum(ψ, conj(ψ)) |> only ≈ 1.0
end

@testset "apply identity, truncate to χ=1" begin
    U, s, V = Operations.simple_update(
        Γa, Index((; site=1, cut=1)),
        Γb, Index((; site=2, cut=1)),
        Index((; bond=(1, 2))),
        op_identity, Index((; site=1, cut=2)), Index((; site=2, cut=2));
        maxdim=1,
    )

    @test U ≈ @view Γa[Index((; bond=(1, 2)))=>1:1]
    @test V ≈ @view Γb[Index((; bond=(1, 2)))=>1:1]
    @test s ≈ Tensor([1], [Index((; bond=(1, 2)))])
end

@testset "apply cx, truncate to χ=1" begin
    U, s, V = Operations.simple_update(
        Γa, Index((; site=1, cut=1)),
        Γb, Index((; site=2, cut=1)),
        Index((; bond=(1, 2))),
        op_cx, Index((; site=1, cut=2)), Index((; site=2, cut=2));
        maxdim=1,
    )

    @test U ≈ @view Γa[Index((; bond=(1, 2)))=>1:1]
    @test V ≈ Tensor(1 / √2 * [1; 1;;], [Index((; site=2, cut=1)), Index((; bond=(1, 2)))])
    @test size(V, Index((; bond=(1, 2)))) == 1
    @test s ≈ Tensor([√2], [Index((; bond=(1, 2)))])
end
