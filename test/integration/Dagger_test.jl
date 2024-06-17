using Muscle
using Dagger
using Distributed

@testset "Dagger" begin
    addprocs(1)
    @everywhere using Dagger
    @everywhere using Muscle

    try
        @testset "einsum" begin
            @testset "block-block" begin
                data1, data2 = rand(4, 4), rand(4, 4)
                block_array1 = distribute(data1, Dagger.Blocks(2, 2))
                block_array2 = distribute(data2, Dagger.Blocks(2, 2))

                contracted_array = einsum("ij,jk->ik", data1, data2)
                contracted_block_array = einsum("ij,jk->ik", block_array1, block_array2)

                @test contracted_block_array isa DArray
                @test all(==((2, 2)) ∘ size, Dagger.domainchunks(contracted_block_array))
                @test collect(contracted_block_array) ≈ contracted_array
            end
        end
    finally
        rmprocs(workers())
    end
end
