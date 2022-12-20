using Muscle.Kernel: swap!, naiveswap!

@testset "Kernel" verbose = true begin
    @testset "swap!" verbose = true begin
        @testset "naiveswap!" begin
            @test begin
                A, B = Int[], Int[]
                naiveswap!(A, B)
                A == [] && B == []
            end

            @test_throws BoundsError begin
                A, B = [1, 1], [2]
                naiveswap!(A, B)
            end

            @test begin
                A, B = [1], [2]
                naiveswap!(A, B)
                A == [2] && B == [1]
            end

            @test begin
                A, B = fill(1, 8), fill(2, 8)
                naiveswap!(A, B)
                A == fill(2, 8) && B == fill(1, 8)
            end

            @test begin
                A, B = fill(1.0, 8), fill(2.0, 8)
                naiveswap!(A, B)
                A == fill(2.0, 8) && B == fill(1.0, 8)
            end

            @test begin
                A, B = fill(1im, 8), fill(2im, 8)
                naiveswap!(A, B)
                A == fill(2im, 8) && B == fill(1im, 8)
            end

            @test begin
                A, B = fill(1.0im, 8), fill(2.0im, 8)
                naiveswap!(A, B)
                A == fill(2.0im, 8) && B == fill(1.0im, 8)
            end
        end

        @testset "viswap!" begin end
    end

    @testset "mapswap!" verbose = true begin end
end