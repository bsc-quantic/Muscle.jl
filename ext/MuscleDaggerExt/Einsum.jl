using Muscle
using Dagger: Dagger, ArrayOp, Context, ArrayDomain, EagerThunk, DArray
using LinearAlgebra

struct Einsum{T,N} <: ArrayOp{T,N}
    ic::Vector{Symbol}
    a::ArrayOp
    ia::Vector{Symbol}
    b::ArrayOp
    ib::Vector{Symbol}

    function Einsum(ic, a, ia, b, ib)
        allunique(ia) || throw(ErrorException("ia must have unique indices"))
        allunique(ib) || throw(ErrorException("ib must have unique indices"))
        allunique(ic) || throw(ErrorException("ic must have unique indices"))
        ic ⊆ ia ∪ ib || throw(ErrorException("ic must be a subset of ia ∪ ib"))
        return new{Base.promote_eltype(a, b),length(ic)}(ic, a, ia, b, ib)
    end
end

function Base.size(@nospecialize(x::Einsum))
    return Tuple(
        Iterators.map(x.ic) do i
            if i ∈ x.ia
                size(x.a, findfirst(==(i), x.ia))
            elseif i ∈ x.ib
                size(x.b, findfirst(==(i), x.ib))
            else
                throw(ErrorException("index $i not found in a nor b"))
            end
        end,
    )
end

function Dagger.Blocks(@nospecialize(x::Einsum))
    return Dagger.Blocks(map(x.ic) do i
        j = findfirst(==(i), x.ia)
        isnothing(j) || return x.a.partitioning.blocksize[j]

        j = findfirst(==(i), x.ib)
        isnothing(j) || return x.b.partitioning.blocksize[j]

        throw(ErrorException("index :$i not found in a nor b"))
    end...)
end

function Dagger.stage(ctx::Context, op::Einsum{T,N}) where {T,N}
    domain = Dagger.ArrayDomain([1:l for l in size(op)])
    partitioning = Dagger.Blocks(op)

    # NOTE careful with ÷ for dividing into partitions
    subdomains = Array{ArrayDomain{N,NTuple{2,UnitRange{Int}}}}(undef, map(÷, size(op), partitioning.blocksize))
    for indices in eachindex(IndexCartesian(), subdomains)
        subdomains[indices] = ArrayDomain(
            map(Tuple(indices), partitioning.blocksize) do i, step
                (i - 1) * step .+ (1:step)
            end,
        )
    end

    suminds = setdiff(op.ia ∪ op.ib, op.ic)
    inner_perm_a = sortperm(map(i -> findfirst(==(i), op.ia), suminds))
    inner_perm_b = sortperm(map(i -> findfirst(==(i), op.ib), suminds))

    mask_a = op.ic .∈ (op.ia,)
    mask_b = op.ic .∈ (op.ib,)
    outer_perm_a = map(i -> findfirst(==(i), op.ia), op.ic[mask_a])
    outer_perm_b = map(i -> findfirst(==(i), op.ib), op.ic[mask_b])

    chunks = similar(subdomains, EagerThunk)
    for indices in eachindex(IndexCartesian(), chunks)
        outer_indices_a = Tuple(indices)[mask_a]
        chunks_a = dropdims(
            reduce(zip(outer_perm_a, outer_indices_a); init=Dagger.chunks(op.a)) do acc, (d, i)
                selectdim(acc, d, i:i)
            end;
            dims=Tuple(outer_perm_a),
        )
        chunks_a = permutedims(chunks_a, inner_perm_a)

        outer_indices_b = Tuple(indices)[mask_b]
        chunks_b = dropdims(
            reduce(zip(outer_perm_b, outer_indices_b); init=Dagger.chunks(op.b)) do acc, (d, i)
                selectdim(acc, d, i:i)
            end;
            dims=Tuple(outer_perm_b),
        )
        chunks_b = permutedims(chunks_b, inner_perm_b)

        chunks[indices] = Dagger.treereduce(
            Dagger.AddComputeOp,
            map(chunks_a, chunks_b) do chunk_a, chunk_b
                # TODO add ThunkOptions: alloc_util, occupancy, ...
                Dagger.@spawn einsum(op.ic, chunk_a, op.ia, chunk_b, op.ib)
            end,
        )
    end

    return DArray(T, domain, subdomains, chunks, partitioning)
end

Muscle.einsum(ic, a::Dagger.DArray, ia, b::Dagger.DArray, ib) = Dagger._to_darray(Einsum(ic, a, ia, b, ib))

Muscle.einsum(a, b::Dagger.DArray; kwargs...) = contract(b, a; kwargs...)
function Muscle.einsum(a::Dagger.DArray, b; kwargs...)
    throw(ArgumentError("einsum on a Dagger.DArray with a non-DArray is not yet supported"))
end
