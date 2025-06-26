module MuscleITensorsExt

using Muscle
using ITensors: ITensors, ITensor, Index

function symbolize(index::Index)
    tag = string(ITensors.id(index))

    # NOTE ITensors' Index's tag only has space for 16 characters
    return Symbol(length(tag) > 16 ? tag[(end - 16 + 1):end] : tag)
end

function tagize(index::Symbol)
    tag = string(index)

    # NOTE ITensors' Index's tag only has space for 16 characters
    return length(tag) > 16 ? tag[(end - 16 + 1):end] : tag
end

# TODO customize index names
function Base.convert(::Type{Tensor}, tt::ITensor)
    array = ITensors.array(tt)
    is = map(symbolize, ITensors.inds(tt))
    return Tensor(array, is)
end

function Base.convert(::Type{ITensor}, tt::Tensor; iinds=Dict{Symbol,Index}())
    indices = map(Muscle.inds(tt)) do i
        haskey(iinds, i) ? iinds[i] : Index(size(tt, i), tagize(i.tag)) #TODO check I added a .tag here 
    end
    return ITensor(parent(tt), indices)
end

# Base.convert(::Type{TensorNetwork}, tn::Vector{ITensor}) = TensorNetwork(map(t -> convert(Tensor, t), tn))

# function Base.convert(::Type{Vector{ITensor}}, tn::Tenet.AbstractTensorNetwork; inds=Dict{Symbol,Index}())
#     indices = merge(inds, Dict(
#         map(Iterators.filter(!Base.Fix1(haskey, inds), Tenet.inds(tn))) do i
#             i => Index(size(tn, i), tagize(i))
#         end,
#     ))
#     return map(tensors(tn)) do tensor
#         ITensor(parent(tensor), map(i -> indices[i], Tenet.inds(tensor)))
#     end
# end

end
