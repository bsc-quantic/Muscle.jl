# Utils

# NOTE from https://stackoverflow.com/q/54652787
function nonunique(x)
    uniqueindexes = indexin(unique(x), collect(x))
    nonuniqueindexes = setdiff(1:length(x), uniqueindexes)
    return Tuple(unique(x[nonuniqueindexes]))
end

include("ImmutableArray.jl")
