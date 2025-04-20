struct Index{T}
    tag::T
end

const NamedIndex = Index{Symbol}
const TagIndex{T<:Tag} = Index{T}

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
