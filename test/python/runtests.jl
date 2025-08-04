using Pkg
# force dev install of Muscle.jl on GitHub CI
Pkg.develop(; path=joinpath(@__DIR__, "..", ".."))
pkg"instantiate"
pkg"precompile"

using Test
using SafeTestsets
using Muscle
using PythonCall

@testset "Python integration" verbose = true begin
    @safetestset "qiskit" include("qiskit.jl")
end
