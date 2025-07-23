using Muscle
using Test
using SafeTestsets

@testset "Unit" verbose = true begin
    @testset "Tensor" include("unit/tensor.jl")
    @testset "Operations" verbose = true begin
        @testset "hadamard" include("unit/operations/hadamard.jl")
        @testset "unary_einsum" include("unit/operations/unary_einsum.jl")
        @testset "binary_einsum" include("unit/operations/binary_einsum.jl")
        @testset "tensor_qr_thin" include("unit/operations/tensor_qr_thin.jl")
        @testset "tensor_svd_thin" include("unit/operations/tensor_svd_thin.jl")
        @testset "tensor_eigen_thin" include("unit/operations/tensor_eigen_thin.jl")
        @testset "simple_update" include("unit/operations/simple_update.jl")
    end
end

@testset "Integration" verbose = true begin
    @safetestset "QuantumTags" include("integration/quantumtags.jl")
    @safetestset "OMEinsum" include("integration/omeinsum.jl")
    @safetestset "Strided" include("integration/strided.jl")
    @safetestset "Dagger" include("integration/dagger.jl")

    #     include("integration/ChainRules_test.jl")

    @safetestset "ITensors" include("integration/itensors.jl")

    if !isnothing(get(ENV, "MUSCLE_TEST_CUDA", nothing))
        @safetestset "CUDA" include("integration/cuda.jl")
    end

    if !isnothing(get(ENV, "MUSCLE_TEST_REACTANT", nothing))
        @safetestset "Reactant" include("integration/reactant.jl")
    end
end

# using Aqua
# Aqua.test_all(Muscle)
