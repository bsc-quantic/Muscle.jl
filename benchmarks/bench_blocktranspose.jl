using Muscle
using BenchmarkTools
using LinearAlgebra
using Plots
using StatsPlots
using Profile

suite = BenchmarkGroup()

suite["ref"] = @benchmarkable LinearAlgebra.transpose!(B, A) setup = (A = rand(Float32, 2048, 2048); B = similar(A))
suite["copy"] = @benchmarkable copy(transpose(A)) setup = (A = rand(Float32, 2048, 2048))

# @benchmark blocktranspose!(A) setup = (A = rand(Float32, 2048, 2048))
suite["block"] = BenchmarkGroup()
for blocksize in [1, 2, 4, 8, 16, 32]
    suite["block"][blocksize] = @benchmarkable blocktranspose!(Val($blocksize), A) setup = (A = rand(Float32, 2048, 2048))
end

tune!(suite)
results = run(suite)

begin
    plt = Plots.Plot()
    StatsPlots.violin!(plt, results["ref"].times, label="ref", yscale=:log10)
    StatsPlots.violin!(plt, results["copy"].times, label="copy")
    for b in [1, 2, 4, 8, 16, 32]
        StatsPlots.violin!(plt, results["block"][b].times, label="block=$b")
    end
    display(plt)
end

Profile.clear()
A = rand(Float32, 2048, 2048)
@profview for _ in 1:1000
    blocktranspose!(Val(8), A)
end