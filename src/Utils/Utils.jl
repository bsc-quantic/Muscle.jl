# Utils

# NOTE from https://stackoverflow.com/q/54652787
# TODO can probably be optimized more by not using sets
function nonunique(x)
    # uniqueindexes = indexin(unique(x), collect(x))
    # nonuniqueindexes = setdiff(1:length(x), uniqueindexes)
    # return Tuple(unique(x[nonuniqueindexes]))

    # NOTE `IdSet` is faster than `Set` for this purpose becasue we avoid computing `hash`
    seen = Base.IdSet{eltype(x)}()
    repeated = Base.IdSet{eltype(x)}()

    for xi in x
        if xi in seen
            if !(xi in repeated)
                push!(repeated, xi)
            end
        else
            push!(seen, xi)
        end
    end

    return collect(repeated)
end

include("ImmutableArray.jl")
