using Muscle
using Test

@testset "Unit" verbose = true begin
    include("swap_test.jl")
    include("Einsum_test.jl")
end

@testset "Integration" verbose = true begin
    include("integration/ChainRules_test.jl")
    include("integration/Dagger_test.jl")
end

using Aqua
Aqua.test_all(Muscle)
