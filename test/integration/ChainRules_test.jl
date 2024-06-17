@testset "ChainRules" begin
    using ChainRulesTestUtils

    @testset "einsum" begin
        @testset "unary" begin
            @testset "real" begin
                x = fill(1.0)
                test_frule(einsum, Char[], x, Char[])
                test_rrule(einsum, Char[], x, Char[]; check_inferred=false)

                x = ones(2)
                test_frule(einsum, Char['i'], x, Char['i'])
                test_rrule(einsum, Char['i'], x, Char['i']; check_inferred=false)

                x = ones(2, 3)
                test_frule(einsum, Char['i', 'j'], x, Char['i', 'j'])
                test_rrule(einsum, Char['i', 'j'], x, Char['i', 'j']; check_inferred=false)

                x = ones(2, 3)
                test_frule(einsum, Char['j'], x, Char['i', 'j'])
                test_rrule(einsum, Char['j'], x, Char['i', 'j']; check_inferred=false)
            end

            @testset "complex" begin
                x = fill(1.0 + 1.0im)
                test_frule(einsum, Char[], x, Char[])
                test_rrule(einsum, Char[], x, Char[]; check_inferred=false)

                x = fill(1.0 + 1.0im, 2)
                test_frule(einsum, Char['i'], x, Char['i'])
                test_rrule(einsum, Char['i'], x, Char['i']; check_inferred=false)

                x = fill(1.0 + 1.0im, 2, 3)
                test_frule(einsum, Char['i', 'j'], x, Char['i', 'j'])
                test_rrule(einsum, Char['i', 'j'], x, Char['i', 'j']; check_inferred=false)

                x = fill(1.0 + 1.0im, 2, 3)
                test_frule(einsum, Char['j'], x, Char['i', 'j'])
                test_rrule(einsum, Char['j'], x, Char['i', 'j']; check_inferred=false)
            end
        end

        @testset "binary" begin
            @testset "real" begin
                # scalar-scalar product
                a = ones()
                b = 2.0 * ones()
                test_frule(einsum, Char[], a, Char[], b, Char[]; check_inferred=false, testset_name="scalar-scalar product - frule")
                test_rrule(einsum, Char[], a, Char[], b, Char[]; check_inferred=false, testset_name="scalar-scalar product - rrule")

                # vector-vector inner product
                a = ones(2)
                b = 2.0 .* ones(2)
                test_frule(einsum, Char[], a, Char['i'], b, Char['i']; check_inferred=false, testset_name="vector-vector inner product - frule")
                test_rrule(einsum, Char[], a, Char['i'], b, Char['i']; check_inferred=false, testset_name="vector-vector inner product - rrule")

                # vector-vector outer product
                a = ones(2)
                b = 2.0 .* ones(3)
                test_frule(einsum, Char['i', 'j'], a, Char['i'], b, Char['j']; check_inferred=false, testset_name="vector-vector outer product - frule")
                test_rrule(einsum, Char['i', 'j'], a, Char['i'], b, Char['j']; check_inferred=false, testset_name="vector-vector outer product - rrule")

                # matrix-vector product
                a = ones(2, 3)
                b = 2.0 .* ones(3)
                test_frule(einsum, Char['i'], a, Char['i', 'j'], b, Char['j']; check_inferred=false, testset_name="matrix-vector product - frule")
                test_rrule(einsum, Char['i'], a, Char['i', 'j'], b, Char['j']; check_inferred=false, testset_name="matrix-vector product - rrule")

                # matrix-matrix product
                a = ones(4, 2)
                b = 2.0 .* ones(2, 3)
                test_frule(einsum, Char['i', 'k'], a, Char['i', 'j'], b, Char['j', 'k']; check_inferred=false, testset_name="matrix-matrix product - frule")
                test_rrule(einsum, Char['i', 'k'], a, Char['i', 'j'], b, Char['j', 'k']; check_inferred=false, testset_name="matrix-matrix product - rrule")

                # matrix-matrix inner product
                a = ones(3, 4)
                b = ones(4, 3)
                test_frule(einsum, Char[], a, Char['i', 'j'], b, Char['j', 'i']; check_inferred=false, testset_name="matrix-matrix inner product - frule")
                test_rrule(einsum, Char[], a, Char['i', 'j'], b, Char['j', 'i']; check_inferred=false, testset_name="matrix-matrix inner product - rrule")
            end

            @testset "complex" begin
                # scalar-scalar product
                a = fill(1.0 + 1.0im)
                b = 2.0 * fill(1.0 + 1.0im)
                test_frule(einsum, Char[], a, Char[], b, Char[]; check_inferred=false, testset_name="scalar-scalar product - frule")
                test_rrule(einsum, Char[], a, Char[], b, Char[]; check_inferred=false, testset_name="scalar-scalar product - rrule")

                # vector-vector inner product
                a = fill(1.0 + 1.0im, 2)
                b = 2.0 .* fill(1.0 + 1.0im, 2)
                test_frule(einsum, Char[], a, Char['i'], b, Char['i']; check_inferred=false, testset_name="vector-vector inner product - frule")
                test_rrule(einsum, Char[], a, Char['i'], b, Char['i']; check_inferred=false, testset_name="vector-vector inner product - rrule")

                # vector-vector outer product
                a = fill(1.0 + 1.0im, 2)
                b = 2.0 .* fill(1.0 + 1.0im, 3)
                test_frule(einsum, Char['i', 'j'], a, Char['i'], b, Char['j']; check_inferred=false, testset_name="vector-vector outer product - frule")
                test_rrule(einsum, Char['i', 'j'], a, Char['i'], b, Char['j']; check_inferred=false, testset_name="vector-vector outer product - rrule")

                # matrix-vector product
                a = fill(1.0 + 1.0im, 2, 3)
                b = 2.0 .* fill(1.0 + 1.0im, 3)
                test_frule(einsum, Char['i'], a, Char['i', 'j'], b, Char['j']; check_inferred=false, testset_name="matrix-vector product - frule")
                test_rrule(einsum, Char['i'], a, Char['i', 'j'], b, Char['j']; check_inferred=false, testset_name="matrix-vector product - rrule")

                # matrix-matrix product
                a = fill(1.0 + 1.0im, 4, 2)
                b = 2.0 .* fill(1.0 + 1.0im, 2, 3)
                test_frule(einsum, Char['i', 'k'], a, Char['i', 'j'], b, Char['j', 'k']; check_inferred=false, testset_name="matrix-matrix product - frule")
                test_rrule(einsum, Char['i', 'k'], a, Char['i', 'j'], b, Char['j', 'k']; check_inferred=false, testset_name="matrix-matrix product - rrule")

                # matrix-matrix inner product
                a = fill(1.0 + 1.0im, 3, 4)
                b = fill(1.0 + 1.0im, 4, 3)
                test_frule(einsum, Char[], a, Char['i', 'j'], b, Char['j', 'i']; check_inferred=false, testset_name="matrix-matrix inner product - frule")
                test_rrule(einsum, Char[], a, Char['i', 'j'], b, Char['j', 'i']; check_inferred=false, testset_name="matrix-matrix inner product - rrule")
            end
        end
    end
end