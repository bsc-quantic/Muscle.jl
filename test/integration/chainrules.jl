using Test
using Muscle
using ChainRulesTestUtils

# load to enable unary_einsum
using OMEinsum

@testset "Tensor" begin
    test_frule(Tensor, ones(), Index[])
    test_rrule(Tensor, ones(), Index[])

    test_frule(Tensor, ones(2), Index[Index(:i)])
    test_rrule(Tensor, ones(2), Index[Index(:i)])

    test_frule(Tensor, ones(2, 3), Index[Index(:i), Index(:j)])
    test_rrule(Tensor, ones(2, 3), Index[Index(:i), Index(:j)])
end

@testset "conj" begin
    x = Tensor(Float64[1 2; 3 4], Index.([:i, :j]))
    test_frule(Base.conj, x; testset_name="real - frule")
    test_rrule(Base.conj, x; testset_name="real - rrule")

    x = Tensor(ComplexF64[1+1im 2; 3 4-2im], Index.([:i, :j]))
    test_frule(Base.conj, x; testset_name="complex - frule")
    test_rrule(Base.conj, x; testset_name="complex - rrule")
end

@testset "unary_einsum" begin
    @testset "real" begin
        # do nothing
        x = Tensor(fill(1.0))
        test_frule(unary_einsum, x)
        test_rrule(unary_einsum, x; check_inferred=false)

        # do nothing
        x = Tensor(ones(2), Index.([:i]))
        test_frule(unary_einsum, x)
        test_rrule(unary_einsum, x; check_inferred=false)

        # do nothing
        x = Tensor(ones(2, 3), Index.([:i, :j]))
        test_frule(unary_einsum, x)
        test_rrule(unary_einsum, x; check_inferred=false)

        # sum over axis
        x = Tensor(ones(2, 3), Index.([:i, :j]))
        test_frule(unary_einsum, x; fkwargs=(; dims=Index[Index(:j)]))
        test_frule(unary_einsum, x; fkwargs=(; out=Index[Index(:i)]))
        test_rrule(unary_einsum, x; fkwargs=(; dims=Index[Index(:j)]), check_inferred=false)
        test_rrule(unary_einsum, x; fkwargs=(; out=Index[Index(:i)]), check_inferred=false)
    end

    @testset "complex" begin
        # NOTE FiniteDifferences doesn't work on these
        # x = Tensor(fill(1.0 + 1.0im))
        # test_frule(unary_einsum, x)
        # test_rrule(unary_einsum, x; check_inferred=false)

        x = Tensor(fill(1.0 + 1.0im, 2), Index.([:i]))
        test_frule(unary_einsum, x)
        test_rrule(unary_einsum, x; check_inferred=false)

        x = Tensor(fill(1.0 + 1.0im, 2, 3), Index.([:i, :j]))
        test_frule(unary_einsum, x)
        test_rrule(unary_einsum, x; check_inferred=false)

        x = Tensor(fill(1.0 + 1.0im, 2, 3), Index.([:i, :j]))
        test_frule(unary_einsum, x; fkwargs=(; dims=Index[Index(:j)]))
        test_frule(unary_einsum, x; fkwargs=(; out=Index[Index(:i)]))
        test_rrule(unary_einsum, x; fkwargs=(; dims=Index[Index(:j)]), check_inferred=false)
        test_rrule(unary_einsum, x; fkwargs=(; out=Index[Index(:i)]), check_inferred=false)
    end
end

@testset "binary_einsum - real" begin
    # scalar-scalar product
    a = Tensor(ones())
    b = Tensor(2.0 * ones())
    test_frule(binary_einsum, a, b; check_inferred=false, testset_name="scalar-scalar product - frule")
    test_rrule(binary_einsum, a, b; check_inferred=false, testset_name="scalar-scalar product - rrule")

    # vector-vector inner product
    a = Tensor(ones(2), Index.([:i]))
    b = Tensor(2.0 .* ones(2), Index.([:i]))
    test_frule(binary_einsum, a, b; check_inferred=false, testset_name="vector-vector inner product - frule")
    test_rrule(binary_einsum, a, b; check_inferred=false, testset_name="vector-vector inner product - rrule")

    # vector-vector outer product
    a = Tensor(ones(2), Index.([:i]))
    b = Tensor(2.0 .* ones(3), Index.([:j]))
    test_frule(binary_einsum, a, b; check_inferred=false, testset_name="vector-vector outer product - frule")
    test_rrule(binary_einsum, a, b; check_inferred=false, testset_name="vector-vector outer product - rrule")

    # matrix-vector product
    a = Tensor(ones(2, 3), Index.([:i, :j]))
    b = Tensor(2.0 .* ones(3), Index.([:j]))
    test_frule(binary_einsum, a, b; check_inferred=false, testset_name="matrix-vector product - frule")
    test_rrule(binary_einsum, a, b; check_inferred=false, testset_name="matrix-vector product - rrule")

    # matrix-matrix product
    a = Tensor(ones(4, 2), Index.([:i, :j]))
    b = Tensor(2.0 .* ones(2, 3), Index.([:j, :k]))
    test_frule(binary_einsum, a, b; check_inferred=false, testset_name="matrix-matrix product - frule")
    test_rrule(binary_einsum, a, b; check_inferred=false, testset_name="matrix-matrix product - rrule")

    # matrix-matrix inner product
    a = Tensor(ones(3, 4), Index.([:i, :j]))
    b = Tensor(ones(4, 3), Index.([:j, :i]))
    test_frule(binary_einsum, a, b; check_inferred=false, testset_name="matrix-matrix inner product - frule")
    test_rrule(binary_einsum, a, b; check_inferred=false, testset_name="matrix-matrix inner product - rrule")
end

@testset "binary_einsum - complex" begin
    # NOTE FiniteDifferences doesn't work on these
    # scalar-scalar product
    # a = Tensor(fill(1.0 + 1.0im))
    # b = Tensor(2.0 * fill(1.0 + 1.0im))
    # test_frule(binary_einsum, a, b; check_inferred=false, testset_name="scalar-scalar product - frule")
    # test_rrule(binary_einsum, a, b; check_inferred=false, testset_name="scalar-scalar product - rrule")

    # vector-vector inner product
    a = Tensor(fill(1.0 + 1.0im, 2), Index.([:i]))
    b = Tensor(2.0 .* fill(1.0 + 1.0im, 2), Index.([:i]))
    test_frule(binary_einsum, a, b; check_inferred=false, testset_name="vector-vector inner product - frule")
    # test_rrule(binary_einsum, a, b; check_inferred=false, testset_name="vector-vector inner product - rrule")

    # vector-vector outer product
    a = Tensor(fill(1.0 + 1.0im, 2), Index.([:i]))
    b = Tensor(2.0 .* fill(1.0 + 1.0im, 3), Index.([:j]))
    test_frule(binary_einsum, a, b; check_inferred=false, testset_name="vector-vector outer product - frule")
    test_rrule(binary_einsum, a, b; check_inferred=false, testset_name="vector-vector outer product - rrule")

    # matrix-vector product
    a = Tensor(fill(1.0 + 1.0im, 2, 3), Index.([:i, :j]))
    b = Tensor(2.0 .* fill(1.0 + 1.0im, 3), Index.([:j]))
    test_frule(binary_einsum, a, b; check_inferred=false, testset_name="matrix-vector product - frule")
    test_rrule(binary_einsum, a, b; check_inferred=false, testset_name="matrix-vector product - rrule")

    # matrix-matrix product
    a = Tensor(fill(1.0 + 1.0im, 4, 2), Index.([:i, :j]))
    b = Tensor(2.0 .* fill(1.0 + 1.0im, 2, 3), Index.([:j, :k]))
    test_frule(binary_einsum, a, b; check_inferred=false, testset_name="matrix-matrix product - frule")
    test_rrule(binary_einsum, a, b; check_inferred=false, testset_name="matrix-matrix product - rrule")

    # matrix-matrix inner product
    a = Tensor(fill(1.0 + 1.0im, 3, 4), Index.([:i, :j]))
    b = Tensor(fill(1.0 + 1.0im, 4, 3), Index.([:j, :i]))
    test_frule(binary_einsum, a, b; check_inferred=false, testset_name="matrix-matrix inner product - frule")
    # test_rrule(binary_einsum, a, b; check_inferred=false, testset_name="matrix-matrix inner product - rrule")
end
