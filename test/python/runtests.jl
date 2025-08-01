using Test
using SafeTestsets

@testset "Python integration" verbose = true begin
    @safetestset "qiskit" include("qiskit.jl")
end
