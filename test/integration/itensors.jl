using Test
using Muscle
using ITensors: ITensor, array
using ITensors: Index as iIndex
using ITensors: inds as iinds
using ITensors: tags as itags

@testset "ITensors" begin
    i = iIndex(2, "i")
    j = iIndex(3, "j")
    k = iIndex(4, "k")

    itensor = ITensor(rand(2, 3, 4), i, j, k)

    tensor = convert(Tensor, itensor)
    @test tensor isa Tensor
    @test size(tensor) == (2, 3, 4)
    @test parent(tensor) == array(itensor)

    tensor = Tensor(rand(2, 3, 4), (:i, :j, :k))
    itensor = convert(ITensor, tensor)
    @test itensor isa ITensor
    @test size(itensor) == (2, 3, 4)
    @test array(itensor) == parent(tensor)
    @test all(splat(==), zip(map(x -> replace(x, "\"" => ""), string.(itags.(iinds(itensor)))), ["i", "j", "k"]))
end
