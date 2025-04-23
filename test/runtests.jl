using Muscle
using Test

@testset "Unit" verbose = true begin
    @testset "Operations" verbose = true begin
        # @testset "Unary Einsum" include("unit/operations/unary_einsum.jl")
        @testset "Binary Einsum" include("unit/operations/binary_einsum.jl")
    end
end

# @testset "Integration" verbose = true begin
#     include("integration/ChainRules_test.jl")
#     include("integration/Dagger_test.jl")
# end

# using Aqua
# Aqua.test_all(Muscle)
