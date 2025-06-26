struct Index{T}
    tag::T
end

const NamedIndex = Index{Symbol}

# TODO checkout whether this is a good idea
Base.copy(x::Index) = x

Index(name::String) = Index(Symbol(name))

function Base.show(io::IO, ind::Index)
    print(io, "index<")
    print(io, ind.tag)
    print(io, ">")
end

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

function factorinds(all_inds, left_inds::Vector{Index}, right_inds::Vector{Index})
    if !isdisjoint(left_inds, right_inds)
        throw(ArgumentError("left ($left_inds) and right $(right_inds) indices must be disjoint"))
    end

    left_inds, right_inds = if isempty(left_inds)
        (setdiff(all_inds, right_inds), right_inds)
    elseif isempty(right_inds)
        (left_inds, setdiff(all_inds, left_inds))
    else
        (left_inds, right_inds)
    end

    if !all(!isempty, (left_inds, right_inds))
        throw(ArgumentError("no right-indices left in factorization"))
    end

    if !all(∈(all_inds), left_inds ∪ right_inds)
        throw(ArgumentError("indices must be in $(all_inds)"))
    end

    return left_inds, right_inds
end

function factorinds(all_inds, left_inds, right_inds)
    _left_inds = if left_inds isa Index
        Index[left_inds]
    elseif isempty(left_inds)
        Index[]
    elseif left_inds isa Tuple
        Index[ind for ind in left_inds]
    elseif left_inds isa Vector{<:Index} && !(left_inds isa Vector{Index})
        convert(Vector{Index}, left_inds)
    else
        left_inds
    end

    _right_inds = if right_inds isa Index
        Index[right_inds]
    elseif isempty(right_inds)
        Index[]
    elseif right_inds isa Tuple
        Index[ind for ind in right_inds]
    elseif right_inds isa Vector{<:Index} && !(right_inds isa Vector{Index})
        convert(Vector{Index}, right_inds)
    else
        right_inds
    end

    return factorinds(all_inds, _left_inds, _right_inds)
end
