struct Index{T}
    tag::T
end

# TODO checkout whether this is a good idea
Base.copy(x::Index) = x

Index(name::String) = Index(Symbol(name))

# index management
function findperm(from, to)
    @assert issetequal(from, to)

    # if there are hyperindices, we remove one by one
    inds_to = collect(Union{Missing,Symbol}, to)

    return map(from) do ind
        i = findfirst(isequal(ind), inds_to)

        # mark element as used
        inds_to[i] = missing

        i
    end
end
