using Muscle
using Test

include("swap_test.jl")

using Aqua
Aqua.test_all(Muscle)
