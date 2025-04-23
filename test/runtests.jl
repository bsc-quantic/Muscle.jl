using Muscle
using Test

@testset "Unit" verbose = true begin
    @testset "Operations" verbose = true begin
        # @testset "unary_einsum" include("unit/operations/unary_einsum.jl")
        @testset "binary_einsum" include("unit/operations/binary_einsum.jl")
        @testset "tensor_svd_thin" include("unit/operations/tensor_svd_thin.jl")
    end
end

# @testset "Integration" verbose = true begin
#     include("integration/ChainRules_test.jl")
#     include("integration/Dagger_test.jl")
# end

# using Aqua
# Aqua.test_all(Muscle)
