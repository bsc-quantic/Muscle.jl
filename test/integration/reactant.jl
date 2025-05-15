using Test
using Muscle
using Reactant
using Adapt
using Enzyme

# TODO test `make_tracer`
# TODO test `create_result`
# TODO test `traced_getfield`

# TODO test unary einsum
# TODO test scalar × tensor
@testset "conj" begin
    A = Tensor(rand(ComplexF64, 2, 3), (:i, :j))
    Are = adapt(ConcreteRArray, A)

    C = conj(A)
    Cre = @jit conj(Are)

    @test Cre ≈ C
end

@testset "binary_einsum" begin
    @testset "matrix multiplication - eltype=$T" for T in [Float64, ComplexF64]
        A = Tensor(rand(T, 2, 3), (:i, :j))
        B = Tensor(rand(T, 3, 4), (:j, :k))
        Are = adapt(ConcreteRArray, A)
        Bre = adapt(ConcreteRArray, B)

        @testset "without permutation" begin
            C = binary_einsum(A, B)
            Cre = @jit binary_einsum(Are, Bre)
            @test Cre ≈ C

            @testset "hybrid" begin
                Cre = @jit binary_einsum(A, Bre)
                @test Cre ≈ C
            end
        end

        @testset "with permutation" begin
            f(a, b) = binary_einsum(a, b; out=Index.([:k, :i]))
            C = f(A, B)
            Cre = @jit f(Are, Bre)
            @test Cre ≈ C

            @testset "hybrid" begin
                Cre = @jit f(A, Bre)
                @test Cre ≈ C
            end
        end
    end

    @testset "inner product - eltype=$T" for T in [Float64, ComplexF64]
        A = Tensor(rand(T, 3, 4), (:i, :j))
        B = Tensor(rand(T, 4, 3), (:j, :i))
        C = binary_einsum(A, B)

        Are = adapt(ConcreteRArray, A)
        Bre = adapt(ConcreteRArray, B)
        Cre = @jit binary_einsum(Are, Bre)

        @test Cre ≈ C
    end

    @testset "outer product - eltype=$T" for T in [Float64, ComplexF64]
        A = Tensor(rand(T, 2, 2), (:i, :j))
        B = Tensor(rand(T, 2, 2), (:k, :l))
        C = binary_einsum(A, B)

        Are = adapt(ConcreteRArray, A)
        Bre = adapt(ConcreteRArray, B)
        Cre = @jit binary_einsum(Are, Bre)

        @test Cre ≈ C
    end

    @testset "manual - eltype=$T" for T in [Float64, ComplexF64]
        A = Tensor(rand(T, 2, 3, 4), (:i, :j, :k))
        B = Tensor(rand(T, 4, 5, 3), (:k, :l, :j))
        Are = adapt(ConcreteRArray, A)
        Bre = adapt(ConcreteRArray, B)

        # binary_einsumion of all common indices
        f1(a, b) = binary_einsum(a, b; dims=[Index(:j), Index(:k)])
        C = f1(A, B)
        Cre = @jit f1(Are, Bre)

        @test Cre ≈ C

        # binary_einsumion of not all common indices
        f2(a, b) = binary_einsum(a, b; dims=[Index(:j)])
        C = f2(A, B)
        Cre = @jit f2(Are, Bre)

        @test Cre ≈ C
    end

    @testset "multiple tensors - eltype=$T" for T in [Float64, ComplexF64]
        A = Tensor(rand(T, 2, 3, 4), (:i, :j, :k))
        B = Tensor(rand(T, 4, 5, 3), (:k, :l, :j))
        C = Tensor(rand(T, 5, 6, 2), (:l, :m, :i))
        D = Tensor(rand(T, 6, 7, 2), (:m, :n, :i))

        Are = adapt(ConcreteRArray, A)
        Bre = adapt(ConcreteRArray, B)
        Cre = adapt(ConcreteRArray, C)
        Dre = adapt(ConcreteRArray, D)

        f3(a, b, c, d) = binary_einsum(binary_einsum(a, b), binary_einsum(c, d; dims=[Index(:m)]))

        X = f3(A, B, C, D)
        Xre = @jit f3(Are, Bre, Cre, Dre)

        @test Xre ≈ X
    end
end

@testset "autodiff" begin
    @testset "inner product" begin
        # inner-product of vectors
        @testset let
            A = Tensor([1.0, 2.0], (:i,))
            B = Tensor([3.0, 4.0], (:i,))
            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)

            grad_f(a, b) = Enzyme.gradient(Reverse, binary_einsum, a, b)
            dAre, dBre = @jit grad_f(Are, Bre)

            @test dAre ≈ B
            @test dBre ≈ A
        end

        # inner-product of matrices
        @testset let
            A = Tensor([1.0 2.0; 3.0 4.0], (:i, :k))
            B = Tensor([5.0 6.0; 7.0 8.0], (:i, :k))
            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)

            grad_f2(a, b) = Enzyme.gradient(Reverse, binary_einsum, a, b)
            dAre, dBre = @jit grad_f2(Are, Bre)

            @test dAre ≈ B
            @test dBre ≈ A
        end

        # inner-product of complex matrices
        @testset let
            A = Tensor(ComplexF64[1.0 2.0; 3.0im 4.0], (:i, :k))
            B = Tensor(ComplexF64[5.0 6.0im; 7.0 8.0], (:i, :k))
            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)

            grad_f3(a, b) = Enzyme.gradient(Reverse, binary_einsum, a, b)
            dAre, dBre = @jit grad_f3(Are, Bre)

            @test dAre ≈ conj(B)
            @test dBre ≈ conj(A)
        end
    end

    # TODO test other einsum cases
end
