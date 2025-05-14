struct Index{T}
    tag::T
end

const NamedIndex = Index{Symbol}

# TODO checkout whether this is a good idea
Base.copy(x::Index) = x

Index(name::String) = Index(Symbol(name))

# index management
function findperm(from::AbstractVector{I}, to::AbstractVector{I}) where {I<:Index}
    @assert issetequal(from, to)

    # if there are hyperindices, we remove one by one
    inds_to = collect(Union{Missing,I}, to)

    return map(from) do ind
        i = findfirst(isequal(ind), inds_to)

        # mark element as used
        inds_to[i] = missing

        i
    end
end

# required for `Tensor` constructor
function Base.convert(::Type{ImmutableArray{Index,N}}, x::ImmutableArray{I,N}) where {I<:Index,N}
    return ImmutableArray{Index,N}(x.data)
end

function factorinds(all_inds, left_inds, right_inds)
    isdisjoint(left_inds, right_inds) ||
        throw(ArgumentError("left ($left_inds) and right $(right_inds) indices must be disjoint"))

    left_inds, right_inds = if isempty(left_inds)
        (setdiff(all_inds, right_inds), right_inds)
    elseif isempty(right_inds)
        (left_inds, setdiff(all_inds, left_inds))
    else
        (left_inds, right_inds)
    end

    all(!isempty, (left_inds, right_inds)) || throw(ArgumentError("no right-indices left in factorization"))
    all(∈(all_inds), left_inds ∪ right_inds) || throw(ArgumentError("indices must be in $(all_inds)"))

    return left_inds, right_inds
end
