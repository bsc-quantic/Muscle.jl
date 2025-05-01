using Muscle
using Test
using SafeTestsets

@testset "Unit" verbose = true begin
    @testset "Tensor" include("unit/tensor.jl")
    @testset "Operations" verbose = true begin
        # @testset "unary_einsum" include("unit/operations/unary_einsum.jl")
        @testset "binary_einsum" include("unit/operations/binary_einsum.jl")
        @testset "tensor_qr_thin" include("unit/operations/tensor_qr_thin.jl")
        @testset "tensor_svd_thin" include("unit/operations/tensor_svd_thin.jl")
        @testset "simple_update" include("unit/operations/simple_update.jl")
    end
end

@testset "Integration" verbose = true begin
    @safetestset "Reactant" begin
        include("integration/reactant.jl")
    end

    #     include("integration/ChainRules_test.jl")
    #     include("integration/Dagger_test.jl")
end

# using Aqua
# Aqua.test_all(Muscle)
