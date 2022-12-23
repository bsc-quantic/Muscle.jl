using Muscle: mapswap!, Naive, Vectorized

@testset "Kernels" verbose = true begin
    @testset "mapswap!" verbose = true begin
        for impl in [Naive, Vectorized]
            @testset "$impl implementation" begin
                @test begin
                    A, B = Int[], Int[]
                    mapswap!(impl, A, B)
                    A == [] && B == []
                end

                @test_throws DimensionMismatch begin
                    A, B = [1, 1], [2]
                    mapswap!(impl, A, B)
                end

                @test begin
                    A, B = [1], [2]
                    mapswap!(impl, A, B)
                    A == [2] && B == [1]
                end

                @test begin
                    A, B = fill(1, 8), fill(2, 8)
                    mapswap!(impl, A, B)
                    A == fill(2, 8) && B == fill(1, 8)
                end

                @test begin
                    A, B = fill(1.0, 8), fill(2.0, 8)
                    mapswap!(impl, A, B)
                    A == fill(2.0, 8) && B == fill(1.0, 8)
                end

                @test begin
                    A, B = fill(1im, 8), fill(2im, 8)
                    mapswap!(impl, A, B)
                    A == fill(2im, 8) && B == fill(1im, 8)
                end

                @test begin
                    A, B = fill(1.0im, 8), fill(2.0im, 8)
                    mapswap!(impl, A, B)
                    A == fill(2.0im, 8) && B == fill(1.0im, 8)
                end
            end
        end
    end
end