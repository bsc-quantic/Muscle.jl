struct ImmutableArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

const ImmutableVector{T} = ImmutableArray{T,1}
const ImmutableMatrix{T} = ImmutableArray{T,2}

# ImmutableArray(data::Array{T,N}) where {T,N} = ImmutableArray{T,N}(data)
ImmutableVector(data::Vector{T}) where {T} = ImmutableVector{T}(data)
ImmutableMatrix(data::Matrix{T}) where {T} = ImmutableMatrix{T}(data)

ImmutableArray(data::Tuple) = ImmutableArray(collect(data))
ImmutableVector(data::Tuple) = ImmutableVector(collect(data))
ImmutableArray{T}(data::Tuple) where {T} = ImmutableArray{T,1}(collect(data))
ImmutableArray{T,N}(data::Tuple) where {T,N} = ImmutableArray{T,N}(collect(data))

# TODO delete `parent` method because it can leak
Base.parent(a::ImmutableArray) = a.data
Base.size(a::ImmutableArray) = size(a.data)
Base.copy(a::ImmutableArray) = ImmutableArray(copy(a.data))

function Base.getindex(a::ImmutableArray, i::Int...)
    return a.data[i...]
end
