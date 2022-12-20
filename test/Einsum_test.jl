using Muscle: parse, OuterProduct, InnerProduct, Trace, Permutation

@testset "Einsum - Parsing - InnerProduct" begin
    @test parse(InnerProduct, [:m, :n], [:m, :k], [:k, :n]) == [:k]
    @test parse(InnerProduct, [:n, :m], [:m, :k], [:k, :n]) == [:k]
    @test parse(InnerProduct, [:m, :k, :n], [:m, :k], [:k, :n]) == []
    @test parse(InnerProduct, [:m, :n], [:m], [:n]) == []
end

@testset "Einsum - Parsing - OuterProduct" begin
    @test parse(OuterProduct, [:m, :n], [:m, :k], [:k, :n]) == [:m, :n]
    @test parse(OuterProduct, [:n, :m], [:m, :k], [:k, :n]) == [:m, :n]
    @test parse(OuterProduct, [:m, :k, :n], [:m, :k], [:k, :n]) == [:m, :n]
    @test parse(OuterProduct, [:m, :n], [:m], [:n]) == [:m, :n]
end

@testset "Einsum - Parsing - Trace" begin
    @test parse(Trace, [:i], [:i, :i]) == [:i]
    @test parse(Trace, [:m, :n], [:m, :n]) == []
    @test parse(Trace, [:m, :k, :n], [:m, :k, :n, :k]) == [:k]
end

@testset "Einsum - Parsing - Permutation" begin
    @test parse(Permutation, [:a, :b], [:a, :b]) == ...
    @test parse(Permutation, [:b, :a], [:a, :b]) == ...
end