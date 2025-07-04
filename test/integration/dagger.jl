using Muscle
using Dagger
using Distributed

@testset "Dagger" begin
    addprocs(1)
    @everywhere using Dagger
    @everywhere using Muscle

    try
        @testset "binary_einsum" begin
            @testset "block-block" begin
                data1 = Float64[1.0 2.0; 3.0 4.0]
                data2 = Float64[5.0 6.0; 7.0 8.0]

                a = Tensor(data1, [Index(:i), Index(:j)])
                b = Tensor(data2, [Index(:j), Index(:k)])

                block_data1 = distribute(data1, Dagger.Blocks(1, 1))
                block_data2 = distribute(data2, Dagger.Blocks(1, 1))

                block_a = Tensor(block_data1, [Index(:i), Index(:j)])
                block_b = Tensor(block_data2, [Index(:j), Index(:k)])

                c = binary_einsum(a, b)
                block_c = binary_einsum(block_a, block_b)

                @test parent(block_c) isa DArray
                @test all(==((1, 1)) ∘ size, Dagger.domainchunks(block_c))
                @test collect(parent(block_c)) ≈ parent(c)
            end
        end

        # TODO test with other array types
    finally
        rmprocs(workers())
    end
end
