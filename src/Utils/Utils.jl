# Utils

# NOTE from https://stackoverflow.com/q/54652787
# TODO can probably be optimized more by not using sets
function nonunique(x)
    # uniqueindexes = indexin(unique(x), collect(x))
    # nonuniqueindexes = setdiff(1:length(x), uniqueindexes)
    # return Tuple(unique(x[nonuniqueindexes]))

    # NOTE `IdSet` is faster than `Set` for this purpose becasue we avoid computing `hash`
    seen = IdSet{eltype(x)}()
    repeated = IdSet{eltype(x)}()

    for xi in x
        if xi in seen
            push!(repeated, xi)
        else
            push!(seen, xi)
        end
    end

    return repeated
end

include("ImmutableArray.jl")
