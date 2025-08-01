using Pkg
pkg"instantiate"
pkg"precompile"

using Test
using SafeTestsets
using Muscle
using PythonCall

@testset "Python integration" verbose = true begin
    @safetestset "qiskit" include("qiskit.jl")
end
