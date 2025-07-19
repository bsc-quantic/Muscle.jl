using Test
using Muscle: Tensor, Index, dim
using LinearAlgebra: LinearAlgebra

@testset "Constructors" begin
    @testset "Number" begin
        tensor = Tensor(1.0)
        @test isempty(inds(tensor))
        @test parent(tensor) == fill(1.0)
    end

    @testset "Array" begin
        data = ones(2, 2, 2)
        tensor = Tensor(data, [Index(:i), Index(:j), Index(:k)])

        @test inds(tensor) == [Index(:i), Index(:j), Index(:k)]
        @test parent(tensor) === data

        @test_throws DimensionMismatch Tensor(zeros(2, 3), [Index(:i), Index(:i)])
    end
end

@testset "eltype - $T" for T in [Bool, Int, Float64, Complex{Float64}]
    tensor = Tensor(rand(T, 2), [Index(:i)])
    @test eltype(tensor) == T
end

@testset "elsize - $T" for T in [Bool, Int, Float64, Complex{Float64}]
    tensor = Tensor(rand(T, 2), [Index(:i)])
    @test Base.elsize(tensor) == sizeof(T)
end

@testset "==" begin
    a = Tensor(zeros(2, 2, 2), [Index(:i), Index(:j), Index(:k)])
    b = Tensor(zeros(2, 2, 2), [Index(:i), Index(:j), Index(:k)])
    @test a !== b
    @test a == b

    c = Tensor(zeros(2, 2, 2), [Index(:i), Index(:j), Index(:not_k)])
    @test a != c

    d = Tensor(zeros(2, 2, 4), [Index(:i), Index(:j), Index(:k)])
    @test a != d

    e = Tensor(ones(2, 2, 2), [Index(:i), Index(:j), Index(:k)])
    @test a != e

    f = Tensor(zeros(2, 2, 2, 2), [Index(:i), Index(:j), Index(:k), Index(:l)])
    @test a != f
end

@testset "isequal" begin
    tensor = Tensor(zeros(2, 2, 2), [Index(:i), Index(:j), Index(:k)])
    @test tensor == copy(tensor)
    @test tensor != zeros(size(tensor)...)
    @test zeros(size(tensor)...) != tensor

    @test tensor ∈ [tensor]
    @test copy(tensor) ∈ [tensor]
    @test tensor ∈ [copy(tensor)]
    @test zeros(size(tensor)...) ∉ [tensor]

    @test tensor ∈ Set([tensor])
    @test zeros(size(tensor)...) ∉ Set([tensor])

    @test tensor == permutedims(tensor, (3, 1, 2))
end

@testset "isapprox" begin
    data = rand(2, 3, 4, 5)
    tensor = Tensor(data, [Index(:i), Index(:j), Index(:k), Index(:l)])

    @test tensor ≈ copy(tensor)
    @test tensor ≈ permutedims(tensor, (3, 1, 2, 4))
    @test tensor ≈ permutedims(tensor, (2, 4, 1, 3))
    @test tensor ≈ permutedims(tensor, (4, 3, 2, 1))
    @test tensor ≈ tensor .+ 1e-14

    @test !(tensor ≈ Tensor(data, [Index(:i), Index(:m), Index(:n), Index(:l)]))
    @test !(tensor ≈ Tensor(rand(2, 2, 2), [Index(:i), Index(:j), Index(:k)]))
    @test !(tensor ≈ data)
end

@testset "copy" begin
    tensor = Tensor(zeros(2, 2, 2), [Index(:i), Index(:j), Index(:k)])
    copied = copy(tensor)
    @test tensor !== copied
    @test tensor == copied

    subtensor = view(tensor, Index(:i) => 1)
    copied_subtensor = copy(subtensor)
    @test copied_subtensor isa Tensor{T,N,Array{T,N}} where {T,N}
end

@testset "similar" begin
    tensor = Tensor(zeros(2, 2, 2), [Index(:i), Index(:j), Index(:k)])

    @test eltype(similar(tensor)) == eltype(tensor)
    @test size(similar(tensor)) == size(tensor)
    @test parent(similar(tensor)) !== parent(tensor)
    @test inds(similar(tensor)) == inds(tensor)

    @test inds(similar(tensor; inds=[Index(:a), Index(:b), Index(:c)])) == [Index(:a), Index(:b), Index(:c)]

    @test eltype(similar(tensor, Bool)) == Bool
    @test size(similar(tensor, Bool)) == size(tensor)
    @test inds(similar(tensor, Bool)) == inds(tensor)

    @test eltype(similar(tensor, 2, 2, 4)) == eltype(tensor)
    @test size(similar(tensor, 2, 2, 4)) == (2, 2, 4)
    @test inds(similar(tensor, 2, 2, 4)) == inds(tensor)

    @test eltype(similar(tensor, Bool, 2, 2, 4)) == Bool
    @test size(similar(tensor, Bool, 2, 2, 4)) == (2, 2, 4)
    @test inds(similar(tensor, Bool, 2, 2, 4)) == inds(tensor)

    @test_throws DimensionMismatch similar(tensor, 2, 2)
end

@testset "zero" begin
    tensor = Tensor(ones(2, 2, 2), [Index(:i), Index(:j), Index(:k)])
    @test parent(zero(tensor)) == zeros(size(tensor)...)
    @test inds(zero(tensor)) == inds(tensor)
end

@testset "strides" begin
    tensor = Tensor(zeros(2, 2, 2), [Index(:i), Index(:j), Index(:k)])
    @test strides(tensor) == (1, 2, 4)
    @test stride(tensor, 1) == stride(tensor, Index(:i)) == 1
    @test stride(tensor, 2) == stride(tensor, Index(:j)) == 2
    @test stride(tensor, 3) == stride(tensor, Index(:k)) == 4
end

@testset "unsafe_convert" begin
    tensor = Tensor(zeros(2, 2, 2), [Index(:i), Index(:j), Index(:k)])
    @test Base.unsafe_convert(Ptr{Float64}, tensor) == Base.unsafe_convert(Ptr{Float64}, parent(tensor))
end

@testset "getindex" begin
    data = [1 2; 3 4]
    tensor = Tensor(data, [Index(:i), Index(0)])

    @test tensor[1, 1] == 1
    @test tensor[1, 2] == 2
    @test tensor[2, 1] == 3
    @test tensor[2, 2] == 4

    # indexing with `Index`
    @test tensor[Index(:i) => 1, Index(0) => 1] == 1
    @test tensor[Index(:i) => 1, Index(0) => 2] == 2
    @test tensor[Index(:i) => 2, Index(0) => 1] == 3
    @test tensor[Index(:i) => 2, Index(0) => 2] == 4

    @test tensor[:i => 1, 0 => 1] == 1
    @test tensor[:i => 1, 0 => 2] == 2
    @test tensor[:i => 2, 0 => 1] == 3
    @test tensor[:i => 2, 0 => 2] == 4

    # special case for `Index{Symbol}`
    @testset let tensor = replace(tensor, Index(0) => Index(:j))
        @test tensor[i=1, j=1] == 1
        @test tensor[i=1, j=2] == 2
        @test tensor[i=2, j=1] == 3
        @test tensor[i=2, j=2] == 4
    end

    # partial indexing
    @test tensor[1, :] == [1, 2]
    @test tensor[2, :] == [3, 4]
    @test tensor[:, 1] == [1, 3]
    @test tensor[:, 2] == [2, 4]

    @test tensor[:, :] == data[:, :]
    @test tensor[:] == data[:]

    @test tensor[Index(:i) => 1] == [1, 2]
    @test tensor[Index(:i) => 2] == [3, 4]
    @test tensor[Index(0) => 1] == [1, 3]
    @test tensor[Index(0) => 2] == [2, 4]

    @test tensor[:i => 1] == [1, 2]
    @test tensor[:i => 2] == [3, 4]
    @test tensor[0 => 1] == [1, 3]
    @test tensor[0 => 2] == [2, 4]

    @testset let tensor = replace(tensor, Index(0) => Index(:j))
        @test tensor[i=1] == [1, 2]
        @test tensor[i=2] == [3, 4]
        @test tensor[j=1] == [1, 3]
        @test tensor[j=2] == [2, 4]
    end

    # 0-dim indexing
    @testset let tensor = Tensor(fill(1))
        @test tensor[] == 1
    end
end

@testset "setindex!" begin
    data = [1 2; 3 4]
    tensor = Tensor(data, [Index(:i), Index(0)])

    # assign scalar
    @testset let tensor = copy(tensor)
        tensor[1, 1] = 0
        @test tensor[1, 1] == 0
    end

    @testset let tensor = copy(tensor)
        tensor[Index(:i) => 1, Index(0) => 1] = 0
        @test tensor[1, 1] == 0
    end

    @testset let tensor = copy(tensor)
        tensor[:i => 1, 0 => 1] = 0
        @test tensor[1, 1] == 0
    end

    # assign array
    @testset let tensor = copy(tensor)
        tensor[1, :] = [5, 5]
        @test tensor[1, :] == [5, 5]
    end

    @testset let tensor = copy(tensor)
        tensor[Index(:i) => 1] = [5, 5]
        @test tensor[1, :] == [5, 5]
    end

    @testset let tensor = copy(tensor)
        tensor[:i => 1] = [5, 5]
        @test tensor[1, :] == [5, 5]
    end

    @testset let tensor = copy(tensor)
        tensor[:, 1] = [6, 6]
        @test tensor[:, 1] == [6, 6]
    end

    @testset let tensor = copy(tensor)
        tensor[Index(0) => 1] = [6, 6]
        @test tensor[:, 1] == [6, 6]
    end

    @testset let tensor = copy(tensor)
        tensor[0 => 1] = [6, 6]
        @test tensor[:, 1] == [6, 6]
    end

    @testset let tensor = copy(tensor)
        tensor[:, :] = data * 5
        @test tensor[:, :] == data * 5
    end

    @testset let tensor = copy(tensor)
        tensor[:] = data[:] * 10
        @test tensor[:] == data[:] * 10
    end

    # broadcasting assignment
    @testset let tensor = copy(tensor)
        tensor[1, :] .= 5
        @test tensor[1, :] == [5, 5]
    end

    @testset let tensor = copy(tensor)
        tensor[Index(:i) => 1] .= 5
        @test tensor[1, :] == [5, 5]
    end

    @testset let tensor = copy(tensor)
        tensor[:i => 1] .= 5
        @test tensor[1, :] == [5, 5]
    end

    @testset let tensor = copy(tensor)
        tensor[:, 1] .= 6
        @test tensor[:, 1] == [6, 6]
    end

    @testset let tensor = copy(tensor)
        tensor[Index(0) => 1] .= 6
        @test tensor[:, 1] == [6, 6]
    end

    @testset let tensor = copy(tensor)
        tensor[0 => 1] .= 6
        @test tensor[:, 1] == [6, 6]
    end

    # 0-dim assignment
    @testset let tensor = Tensor(fill(1))
        tensor[] = 2
        @test tensor[] == 2
    end
end

@testset "other indexing methods" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, [Index(:i), Index(:j), Index(:k)])

    @test firstindex(tensor) == 1
    @test lastindex(tensor) == 8
    @test all(firstindex(tensor, i) == 1 for i in 1:ndims(tensor))
    @test all(lastindex(tensor, i) == 2 for i in 1:ndims(tensor))

    @test axes(tensor) == axes(data)
    @test first(tensor) == first(data)
    @test last(tensor) == last(data)

    for i in [0, -1, length(tensor) + 1]
        @test_throws BoundsError tensor[i]
    end
end

@testset "iteration" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, [Index(:i), Index(:j), Index(:k)])

    @test Base.IteratorSize(tensor) == Base.HasShape{3}()
    @test Base.IteratorEltype(tensor) == Base.HasEltype()
    @test all(x -> ==(x...), zip(tensor, data))
end

@testset "Base.replace" begin
    tensor = Tensor(zeros(2, 2, 2), [Index(:i), Index(:j), Index(:k)])
    @test inds(replace(tensor, Index(:i) => Index(:u), Index(:j) => Index(:v), Index(:k) => Index(:w))) ==
        [Index(:u), Index(:v), Index(:w)]
    @test parent(replace(tensor, Index(:i) => Index(:u), Index(:j) => Index(:v), Index(:k) => Index(:w))) ===
        parent(tensor)

    @test inds(replace(tensor, Index(:a) => Index(:u), Index(:b) => Index(:v), Index(:c) => Index(:w))) ==
        [Index(:i), Index(:j), Index(:k)]
    @test parent(replace(tensor, Index(:a) => Index(:u), Index(:b) => Index(:v), Index(:c) => Index(:w))) ===
        parent(tensor)
end

@testset "dim" begin
    tensor = Tensor(zeros(2, 2, 2), [Index(:i), Index(:j), Index(:k)])
    @test dim(tensor, 1) == 1
    for (i, label) in enumerate(inds(tensor))
        @test dim(tensor, label) == i
    end

    @test isnothing(dim(tensor, Index(:_)))
end

@testset "Broadcasting" begin
    data = rand(2, 2, 2)
    @test begin
        tensor = Tensor(data, [Index(:a), Index(:b), Index(:c)])
        tensor = tensor .+ one(eltype(tensor))

        parent(tensor) == data .+ one(eltype(tensor))
    end

    @test begin
        tensor = Tensor(data, [Index(:a), Index(:b), Index(:c)])
        tensor = sin.(tensor)

        parent(tensor) == sin.(data)
    end
end

@testset "selectdim" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, [Index(:i), Index(:j), Index(:k)])

    @test parent(selectdim(tensor, Index(:i), 1)) == selectdim(data, 1, 1)
    @test parent(selectdim(tensor, Index(:j), 2)) == selectdim(data, 2, 2)
    @test issetequal(inds(selectdim(tensor, Index(:i), 1)), [Index(:j), Index(:k)])
    @test issetequal(inds(selectdim(tensor, Index(:i), 1:1)), [Index(:i), Index(:j), Index(:k)])
end

@testset "view" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, [Index(:i), Index(:j), Index(:k)])

    @test parent(view(tensor, 2, :, :)) == view(data, 2, :, :)
    @test parent(view(tensor, Index(:i) => 1)) == view(data, 1, :, :)
    @test parent(view(tensor, Index(:j) => 2)) == view(data, :, 2, :)
    @test parent(view(tensor, Index(:i) => 2, Index(:k) => 1)) == view(data, 2, :, 1)
    @test Index(:i) ∉ inds(view(tensor, Index(:i) => 1))

    @test parent(view(tensor, Index(:i) => 1:1)) == view(data, (1:1), :, :)
    @test Index(:i) ∈ inds(view(tensor, Index(:i) => 1:1))

    data = [1 2; 3 4]
    tensor = Tensor(data, [Index(:i), Index(0)])

    @test @view(tensor[1, 1]) == Tensor(fill(1))
    @test @view(tensor[1, 2]) == Tensor(fill(2))
    @test @view(tensor[2, 1]) == Tensor(fill(3))
    @test @view(tensor[2, 2]) == Tensor(fill(4))

    # indexing with `Index`
    @test @view(tensor[Index(:i) => 1, Index(0) => 1]) == Tensor(fill(1))
    @test @view(tensor[Index(:i) => 1, Index(0) => 2]) == Tensor(fill(2))
    @test @view(tensor[Index(:i) => 2, Index(0) => 1]) == Tensor(fill(3))
    @test @view(tensor[Index(:i) => 2, Index(0) => 2]) == Tensor(fill(4))

    @test @view(tensor[:i => 1, 0 => 1]) == Tensor(fill(1))
    @test @view(tensor[:i => 1, 0 => 2]) == Tensor(fill(2))
    @test @view(tensor[:i => 2, 0 => 1]) == Tensor(fill(3))
    @test @view(tensor[:i => 2, 0 => 2]) == Tensor(fill(4))

    # special case for `Index{Symbol}`
    @testset let tensor = replace(tensor, Index(0) => Index(:j))
        @test @view(tensor[i=1, j=1]) == Tensor(fill(1))
        @test @view(tensor[i=1, j=2]) == Tensor(fill(2))
        @test @view(tensor[i=2, j=1]) == Tensor(fill(3))
        @test @view(tensor[i=2, j=2]) == Tensor(fill(4))
    end

    # partial indexing
    @test @view(tensor[1, :]) == Tensor([1, 2], [Index(0)])
    @test @view(tensor[2, :]) == Tensor([3, 4], [Index(0)])
    @test @view(tensor[:, 1]) == Tensor([1, 3], [Index(:i)])
    @test @view(tensor[:, 2]) == Tensor([2, 4], [Index(:i)])

    @test @view(tensor[:, :]) == Tensor(@view(data[:, :]), [Index(:i), Index(0)])
    @test @view(tensor[:]) == Tensor(@view(data[:]), [Index(:i)])

    @test @view(tensor[Index(:i) => 1]) == Tensor([1, 2], [Index(0)])
    @test @view(tensor[Index(:i) => 2]) == Tensor([3, 4], [Index(0)])
    @test @view(tensor[Index(0) => 1]) == Tensor([1, 3], [Index(:i)])
    @test @view(tensor[Index(0) => 2]) == Tensor([2, 4], [Index(:i)])

    @test @view(tensor[:i => 1]) == Tensor([1, 2], [Index(0)])
    @test @view(tensor[:i => 2]) == Tensor([3, 4], [Index(0)])
    @test @view(tensor[0 => 1]) == Tensor([1, 3], [Index(:i)])
    @test @view(tensor[0 => 2]) == Tensor([2, 4], [Index(:i)])

    @testset let tensor = replace(tensor, Index(0) => Index(:j))
        @test @view(tensor[i=1]) == Tensor([1, 2], [Index(:j)])
        @test @view(tensor[i=2]) == Tensor([3, 4], [Index(:j)])
        @test @view(tensor[j=1]) == Tensor([1, 3], [Index(:i)])
        @test @view(tensor[j=2]) == Tensor([2, 4], [Index(:i)])
    end

    # 0-dim indexing
    @testset let tensor = Tensor(fill(1))
        @test @view(tensor[]) == Tensor(@view(parent(tensor)[]))
        @test @view(tensor[])[] == 1
    end
end

@testset "permutedims" begin
    data = reshape(collect(1:24), 2, 3, 4)
    tensor = Tensor(data, [Index(:i), Index(:j), Index(:k)])

    perm = (3, 1, 2)
    c = permutedims(tensor, perm)
    @test inds(c) == [Index(:k), Index(:i), Index(:j)]
    @test parent(c) == permutedims(data, perm)

    newtensor = Tensor(similar(data, 4, 2, 3), [Index(:a), Index(:b), Index(:c)])
    permutedims!(newtensor, tensor, perm)
    @test parent(newtensor) == parent(c)

    # list of indices as permutator
    perm2 = [Index(:k), Index(:i), Index(:j)]
    c2 = permutedims(tensor, perm2)
    @test inds(c2) == [Index(:k), Index(:i), Index(:j)]
    @test parent(c2) == permutedims(data, perm)
end

@testset "conj/adjoint" begin
    @testset "scalar" begin
        tensor = Tensor(fill(1.0 + 1.0im))

        @test isempty(inds(conj(tensor)))
        @test isapprox(conj(tensor), 1.0 - 1.0im)
        @test adjoint(tensor) == conj(tensor)
    end

    @testset "Vector" begin
        data = fill(1.0 + 1.0im, 2)
        tensor = Tensor(data, [Index(:i)])

        @test inds(conj(tensor)) == [Index(:i)]
        @test all(isapprox.(conj(tensor), fill(1.0 - 1.0im, size(data)...)))
        @test adjoint(tensor) == conj(tensor)
    end

    @testset "Matrix" begin
        data = fill(1.0 + 1.0im, 2, 2)
        tensor = Tensor(data, (Index(:i), Index(:j)))

        @test inds(adjoint(tensor)) == [Index(:i), Index(:j)]
        @test all(isapprox.(conj(tensor), fill(1.0 - 1.0im, size(data)...)))
        @test adjoint(tensor) == conj(tensor)
    end
end

@testset "transpose" begin
    @testset "Vector" begin
        data = rand(Complex{Float64}, 2)
        tensor = Tensor(data, [Index(:i)])

        @test inds(transpose(tensor)) == [Index(:i)]
        @test ndims(transpose(tensor)) == 1
        @test all(isapprox.(transpose(tensor), data))
    end

    @testset "Matrix" begin
        data = rand(Complex{Float64}, 2, 2)
        tensor = Tensor(data, [Index(:i), Index(:j)])

        @test inds(transpose(tensor)) == [Index(:j), Index(:i)]
        @test ndims(transpose(tensor)) == 2
        @test all(isapprox.(transpose(tensor), transpose(data)))
    end
end

@testset "extend" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, [Index(:i), Index(:j), Index(:k)])

    let new = Muscle.extend(tensor; label=Index(:x), axis=1)
        @test inds(new) == [Index(:x), Index(:i), Index(:j), Index(:k)]
        @test size(new, Index(:x)) == 1
        @test selectdim(new, Index(:x), 1) == tensor
    end

    let new = Muscle.extend(tensor; label=Index(:x), axis=3)
        @test inds(new) == [Index(:i), Index(:j), Index(:x), Index(:k)]
        @test size(new, Index(:x)) == 1
        @test selectdim(new, Index(:x), 1) == tensor
    end

    let new = Muscle.extend(tensor; label=Index(:x), axis=1, size=2, method=:zeros)
        @test inds(new) == [Index(:x), Index(:i), Index(:j), Index(:k)]
        @test size(new, Index(:x)) == 2
        @test selectdim(new, Index(:x), 1) == tensor
        @test selectdim(new, Index(:x), 2) == Tensor(zeros(size(data)...), inds(tensor))
    end

    let new = Muscle.extend(tensor; label=Index(:x), axis=1, size=2, method=:repeat)
        @test inds(new) == [Index(:x), Index(:i), Index(:j), Index(:k)]
        @test size(new, Index(:x)) == 2
        @test selectdim(new, Index(:x), 1) == tensor
        @test selectdim(new, Index(:x), 2) == tensor
    end
end

@testset "expand" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, [Index(:i), Index(:j), Index(:k)])

    let new = Muscle.expand(tensor, Index(:i), 3; method=:zeros)
        @test inds(new) == [Index(:i), Index(:j), Index(:k)]
        @test size(new, Index(:i)) == 3
        @test view(new, Index(:i) => 1:2) == tensor
        @test view(new, Index(:i) => 3:3) ≈
            Tensor(zeros(Tuple(ind == Index(:i) ? 1 : size(tensor, ind) for ind in inds(tensor))), inds(tensor))
    end

    let new = Muscle.expand(tensor, Index(:j), 3; method=:zeros)
        @test inds(new) == [Index(:i), Index(:j), Index(:k)]
        @test size(new, Index(:j)) == 3
        @test view(new, Index(:j) => 1:2) == tensor
        @test view(new, Index(:j) => 3:3) ≈
            Tensor(zeros(Tuple(ind == Index(:j) ? 1 : size(tensor, ind) for ind in inds(tensor))), inds(tensor))
    end

    let new = Muscle.expand(tensor, Index(:i), 3; method=:rand)
        @test inds(new) == [Index(:i), Index(:j), Index(:k)]
        @test size(new, Index(:i)) == 3
        @test view(new, Index(:i) => 1:2) == tensor
        @test !(
            view(new, Index(:i) => 3:3) ≈
            Tensor(zeros(Tuple(ind == Index(:i) ? 1 : size(tensor, ind) for ind in inds(tensor))), inds(tensor))
        )
    end
end

@testset "Base.cat" begin
    data = rand(2, 2, 2)
    tensor = Tensor(data, [Index(:i), Index(:j), Index(:k)])

    @testset let pad = Tensor(zeros(3, 2, 2), [Index(:i), Index(:j), Index(:k)])
        new = cat(tensor, pad; dims=Index(:i))
        @test inds(new) == [Index(:i), Index(:j), Index(:k)]
        @test size(new, Index(:i)) == 5
        @test view(new, Index(:i) => 1:2) == tensor
        @test view(new, Index(:i) => 3:5) == pad
    end

    @testset let pad = Tensor(zeros(2, 3, 2), [Index(:i), Index(:j), Index(:k)])
        new = cat(tensor, pad; dims=Index(:j))
        @test inds(new) == [Index(:i), Index(:j), Index(:k)]
        @test size(new, Index(:j)) == 5
        @test view(new, Index(:j) => 1:2) == tensor
        @test view(new, Index(:j) => 3:5) == pad
    end

    @testset let pad = Tensor(zeros(2, 2, 3), [Index(:i), Index(:j), Index(:k)])
        new = cat(tensor, pad; dims=Index(:k))
        @test inds(new) == [Index(:i), Index(:j), Index(:k)]
        @test size(new, Index(:k)) == 5
        @test view(new, Index(:k) => 1:2) == tensor
        @test view(new, Index(:k) => 3:5) == pad
    end

    # TODO test cat on more than 1 dim
end

@testset "fuse" begin
    tensor = Tensor(rand(2, 3), [Index(:i), Index(:j)])
    grouped = Muscle.fuse(tensor, [Index(:i), Index(:j)])
    @test vec(tensor) ≈ parent(grouped)

    grouped = Muscle.fuse(tensor, [Index(:j), Index(:i)])
    @test vec(transpose(parent(tensor))) ≈ parent(grouped)

    tensor = Tensor(rand(2, 3, 4), [Index(:i), Index(:k), Index(:j)])
    grouped = Muscle.fuse(tensor, [Index(:i), Index(:j)])
    @test reshape(permutedims(parent(tensor), [2, 1, 3]), 3, 8) ≈ parent(grouped)

    grouped = Muscle.fuse(tensor, [Index(:j), Index(:i)])
    @test reshape(permutedims(parent(tensor), [2, 3, 1]), 3, 8) ≈ parent(grouped)
end

@testset "sum" begin
    tensor = Tensor(ones(Int, 2, 3, 4), [Index(:i), Index(:j), Index(:k)])

    @test sum(tensor) == Tensor(fill(sum(parent(tensor))))
    @test sum(tensor; dims=Index(:i)) == sum(tensor; dims=1) == Tensor(sum(parent(tensor); dims=1), inds(tensor))
    @test sum(tensor; dims=Index(:j)) == sum(tensor; dims=2) == Tensor(sum(parent(tensor); dims=2), inds(tensor))
    @test sum(tensor; dims=Index(:k)) == sum(tensor; dims=3) == Tensor(sum(parent(tensor); dims=3), inds(tensor))
end

@testset "isisometry" begin
    tensor = Tensor(collect(LinearAlgebra.I(8)), [Index(:i), Index(:j)])
    @test isisometry(tensor, Index(:i))
    @test isisometry(tensor, Index(:j))

    tensor = Tensor(reshape(collect(LinearAlgebra.I(8)), 2, 4, 8), [Index(:i), Index(:j), Index(:k)])
    @test !isisometry(tensor, Index(:i))
    @test !isisometry(tensor, Index(:j))
    @test isisometry(tensor, Index(:k))

    tensor = Tensor(ones(8, 8), [Index(:i), Index(:j)])
    @test !isisometry(tensor, Index(:i))
    @test !isisometry(tensor, Index(:j))
end
