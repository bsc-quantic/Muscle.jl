using Test
using Muscle

@testset "tensor - scalar bcast" begin
    a = Tensor(ones(2, 3, 4), [Index(:i), Index(:j), Index(:k)])
    b = Tensor(fill(2.0))

    let c = hadamard(a, b)
        @test size(c) == (2, 3, 4)
        @test inds(c) == [Index(:i), Index(:j), Index(:k)]
        @test parent(c) ≈ 2.0 .* ones(size(a)...)
    end

    let a = copy(a)
        c = hadamard!(a, a, b)
        @test c === a
        @test parent(a) ≈ 2.0 .* ones(size(a)...)
    end
end

@testset "tensor - vector bcast" begin
    a = Tensor(ones(2, 3, 4), [Index(:i), Index(:j), Index(:k)])

    let b = Tensor([1.0, 2.0], [Index(:i)])
        let c = hadamard(a, b)
            @test size(c) == (2, 3, 4)
            @test inds(c) == [Index(:i), Index(:j), Index(:k)]
            for d in size(c, Index(:i))
                @test all(selectdim(c, Index(:i), d) .≈ d)
            end
        end

        let a = copy(a)
            c = hadamard!(a, a, b)
            @test c === a
            for d in size(c, Index(:i))
                @test all(selectdim(a, Index(:i), d) .≈ d)
            end
        end
    end

    let b = Tensor([1.0, 2.0, 3.0], [Index(:j)])
        let c = hadamard(a, b)
            @test size(c) == (2, 3, 4)
            @test inds(c) == [Index(:i), Index(:j), Index(:k)]
            for d in size(c, Index(:j))
                @test all(selectdim(c, Index(:j), d) .≈ d)
            end
        end

        let a = copy(a)
            c = hadamard!(a, a, b)
            @test c === a
            for d in size(c, Index(:j))
                @test all(selectdim(a, Index(:j), d) .≈ d)
            end
        end
    end

    let b = Tensor([1.0, 2.0, 3.0, 4.0], [Index(:k)])
        let c = hadamard(a, b)
            @test size(c) == (2, 3, 4)
            @test inds(c) == [Index(:i), Index(:j), Index(:k)]
            for d in size(c, Index(:k))
                @test all(selectdim(c, Index(:k), d) .≈ d)
            end
        end

        let a = copy(a)
            c = hadamard!(a, a, b)
            @test c === a
            for d in size(c, Index(:k))
                @test all(selectdim(a, Index(:k), d) .≈ d)
            end
        end
    end
end

@testset "tensor - tensor bcast" begin
    a = Tensor(ones(2, 3, 4), [Index(:i), Index(:j), Index(:k)])
    b = Tensor(Float64[1 2 3; 4 5 6], [Index(:i), Index(:j)])

    let c = hadamard(a, b)
        @test size(c) == (2, 3, 4)
        @test inds(c) == [Index(:i), Index(:j), Index(:k)]

        for coord in eachindex(IndexCartesian(), b)
            view_c = view(c, [ind => i for (ind, i) in zip(inds(b), Tuple(coord))]...)
            @test all(view_c .≈ b[coord])
        end
    end

    let a = copy(a)
        c = hadamard!(a, a, b)
        @test c === a

        for coord in eachindex(IndexCartesian(), b)
            view_a = view(a, [ind => i for (ind, i) in zip(inds(b), Tuple(coord))]...)
            @test all(view_a .≈ b[coord])
        end
    end
end

@testset "tensor - tensor" begin
    a = Tensor(ones(2, 3, 4), [Index(:i), Index(:j), Index(:k)])
    b = Tensor(2ones(2, 3, 4), [Index(:i), Index(:j), Index(:k)])

    let c = hadamard(a, b)
        @test size(c) == (2, 3, 4)
        @test inds(c) == [Index(:i), Index(:j), Index(:k)]
        @test all(c .≈ 2.0)
    end

    let a = copy(a)
        c = hadamard!(a, a, b)
        @test c === a
        @test all(a .≈ 2.0)
    end
end
