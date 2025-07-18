using Base: @propagate_inbounds
using Base.Broadcast: Broadcasted, ArrayStyle
using LinearAlgebra
using Adapt

"""
    Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}

An array-like object with named dimensions (i.e. [`Index`](@ref)).
"""
struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    inds::ImmutableVector{Index}

    function Tensor(data::A, inds::ImmutableVector{I}) where {T,N,A<:AbstractArray{T,N},I<:Index}
        if length(inds) != N
            throw(ArgumentError("ndims(data) [$(ndims(data))] must be equal to length(inds) [$(length(inds))]"))
        end

        if !all(i -> allequal(Iterators.map(dim -> size(data, dim), findall(==(i), inds))), nonunique(inds))
            throw(DimensionMismatch("nonuniform size of repeated indices"))
        end

        return new{T,N,A}(data, inds)
    end

    function Tensor(data::A, inds::AbstractVector{I}) where {T,N,A<:AbstractArray{T,N},I<:Index}
        return Tensor(data, ImmutableVector(inds))
    end
end

"""
    Tensor(data::AbstractArray{T,N}, inds::AbstractVector{Index})
    Tensor(data::AbstractArray{T,N}, inds::NTuple{N,Index}) where {T,N}
    Tensor(data::AbstractArray{T,0}) where {T}
    Tensor(data::Number)

Construct a tensor with the given data and indices.
"""
Tensor(data::A, inds::NTuple{N}) where {T,N,A<:AbstractArray{T,N}} = Tensor{T,N,A}(data, collect(inds))
Tensor(data::AbstractArray{T,0}) where {T} = Tensor(data, Index[])
Tensor(data::Number) = Tensor(fill(data))
Tensor{T,N,A}(data::A, inds::AbstractVector) where {T,N,A<:AbstractArray{T,N}} = Tensor(data, ImmutableVector(inds))

# useful methods
Tensor(data::AbstractArray, inds::Vector{Symbol}) = Tensor(data, map(Index, inds))
function Tensor{T,N,A}(data::A, inds::Vector{Symbol}) where {T,N,A<:AbstractArray{T,N}}
    Tensor(data, ImmutableVector(map(Index, inds)))
end

Tensor(x::Tensor) = x
Tensor{T,N,A}(x::Tensor{T,N,A}) where {T,N,A} = x
function Tensor{T,N,A}(x::Tensor) where {T,N,A}
    throw(ArgumentError("Tensor type mismatch: $(typeof(x)) is not a Tensor{T,N,A}"))
end

Tensor(::Tensor, _) = throw(ArgumentError("Can't wrap a `Tensor` with another `Tensor`"))
function Tensor{T,N,A}(::Tensor, _) where {T,N,A}
    throw(ArgumentError("Can't wrap a `Tensor` with another `Tensor`"))
end

"""
    inds(::Tensor)

Return the indices of the `Tensor`.
"""
inds(x::Tensor) = x.inds

Base.copy(t::Tensor{T,N,<:SubArray{T,N}}) where {T,N} = Tensor(copy(parent(t)), copy(inds(t)))
Adapt.adapt_structure(to, x::Tensor) = Tensor(adapt(to, parent(x)), inds(x))

arraytype(::Type{Tensor{T,N,A}}) where {T,N,A} = A
arraytype(::T) where {T<:Tensor} = arraytype(T)

"""
    Base.similar(::Tensor{T,N}[, S::Type, dims::Base.Dims{N}; inds])

Return a uninitialize tensor of the same size, eltype and [`inds`](@ref) as `tensor`. If `S` is provided, the eltype of the tensor will be `S`. If `dims` is provided, the size of the tensor will be `dims`.
"""
Base.similar(t::Tensor; inds=inds(t)) = Tensor(similar(parent(t)), inds)
Base.similar(t::Tensor, S::Type; inds=inds(t)) = Tensor(similar(parent(t), S), inds)
function Base.similar(t::Tensor{T,N}, S::Type, dims::Base.Dims{N}; inds=inds(t)) where {T,N}
    return Tensor(similar(parent(t), S, dims), inds)
end
function Base.similar(t::Tensor, ::Type, dims::Base.Dims{N}; kwargs...) where {N}
    throw(DimensionMismatch("`dims` needs to be of length $(ndims(t))"))
end
Base.similar(t::Tensor{T,N}, dims::Base.Dims{N}; inds=inds(t)) where {T,N} = Tensor(similar(parent(t), dims), inds)
function Base.similar(t::Tensor, dims::Base.Dims{N}; kwargs...) where {N}
    throw(DimensionMismatch("`dims` needs to be of length $(ndims(t))"))
end

"""
    Base.zero(tensor::Tensor)

Return a tensor of the same size, eltype and [`inds`](@ref) as `tensor` but filled with zeros.
"""
Base.zero(t::Tensor) = Tensor(zero(parent(t)), inds(t))

Base.:(==)(a::AbstractArray, b::Tensor) = isequal(b, a)
Base.:(==)(a::Tensor, b::AbstractArray) = isequal(a, b)
Base.:(==)(a::Tensor, b::Tensor) = isequal(a, b)
Base.isequal(a::AbstractArray, b::Tensor) = false
Base.isequal(a::Tensor, b::AbstractArray) = false
function Base.isequal(a::Tensor, b::Tensor)
    issetequal(inds(a), inds(b)) || return false
    perm = findperm(inds(a), inds(b))
    return isequal(parent(a), PermutedDimsArray(parent(b), perm))
end

Base.isequal(a::Tensor{A,0}, b::Tensor{B,0}) where {A,B} = isequal(only(a), only(b))

Base.isapprox(a::AbstractArray, b::Tensor) = false
Base.isapprox(a::Tensor, b::AbstractArray) = false
function Base.isapprox(a::Tensor, b::Tensor; kwargs...)
    issetequal(inds(a), inds(b)) || return false
    perm = findperm(inds(a), inds(b))
    return isapprox(parent(a), PermutedDimsArray(parent(b), perm); kwargs...)
end

Base.isapprox(a::Tensor{T,0}, b::T; kwargs...) where {T} = isapprox(only(a), b; kwargs...)
Base.isapprox(a::T, b::Tensor{T,0}; kwargs...) where {T} = isapprox(b, a; kwargs...)
Base.isapprox(a::Tensor{A,0}, b::Tensor{B,0}; kwargs...) where {A,B} = isapprox(only(a), only(b); kwargs...)

# NOTE: `replace` does not currenly support cyclic replacements
"""
    Base.replace(::Tensor, old_new::Pair{Index,Index}...)

Replace the indices of the tensor according to the given pairs of old and new indices.

!!! warning

    This method does not support cyclic replacements.
"""
Base.replace(t::Tensor, old_new::Pair...) = Tensor(parent(t), replace(inds(t), old_new...))

"""
    Base.parent(::Tensor)

Return the underlying array of the tensor.
"""
Base.parent(t::Tensor) = t.data
Adapt.parent_type(::Type{Tensor{T,N,A}}) where {T,N,A} = A
Adapt.parent_type(::Type{Tensor{T,N}}) where {T,N} = AbstractArray{T,N}
Adapt.parent_type(::Type{Tensor{T}}) where {T} = AbstractArray{T}
Adapt.parent_type(::Type{Tensor}) = AbstractArray
Adapt.parent_type(::T) where {T<:Tensor} = parent_type(T)

"""
    dim(tensor::Tensor, i)

Return the location of the dimension of `tensor` corresponding to the given index `i`.
"""
dim(::Tensor, i::Number) = i
dim(t::Tensor, i::Symbol) = dim(t, Index(i))
dim(t::Tensor, i::Index) = findfirst(==(i), inds(t))

# Iteration interface
Base.IteratorSize(T::Type{Tensor}) = Iterators.IteratorSize(parent_type(T))
Base.IteratorEltype(T::Type{Tensor}) = Iterators.IteratorEltype(parent_type(T))

Base.isdone(t::Tensor) = Base.isdone(parent(t))
Base.isdone(t::Tensor, state) = Base.isdone(parent(t), state)

# Indexing interface
Base.IndexStyle(T::Type{<:Tensor}) = IndexStyle(parent_type(T))

"""
    Base.getindex(::Tensor, i...)
    Base.getindex(::Tensor; i...)
    (::Tensor)[index=i...]

Return the element of the tensor at the given indices. If kwargs are provided, then it is equivalent to calling [`view`](@ref).
"""
@propagate_inbounds Base.getindex(t::Tensor, i...) = getindex(parent(t), i...)
@propagate_inbounds function Base.getindex(t::Tensor; i...)
    length(i) == 0 && return (getindex ∘ parent)(t)
    # WARN `label.tag` might not be stable
    return getindex(t, [get(i, label.tag, Colon()) for label in inds(t)]...)
end

"""
    Base.setindex!(t::Tensor, v, i...)
    Base.setindex(::Tensor; i...)
    (::Tensor)[index=i...]

Set the element of the tensor at the given indices to `v`. If kwargs are provided, then it is equivalent to calling `.=` on [`view`](@ref).
"""
@propagate_inbounds Base.setindex!(t::Tensor, v, i...) = setindex!(parent(t), v, i...)
@propagate_inbounds function Base.setindex!(t::Tensor, v; i...)
    length(i) == 0 && return setindex!(parent(t), v)
    return setindex!(t, v, [get(i, label, Colon()) for label in inds(t)]...)
end

Base.firstindex(t::Tensor) = firstindex(parent(t))
Base.lastindex(t::Tensor) = lastindex(parent(t))

# AbstractArray interface
Base.eltype(x::Tensor) = eltype(x.data)

"""
    Base.size(::Tensor[, i::Index])

Return the size of the underlying array. If the dimension `i` (specified by `Index` or `Integer`) is specified, then the size of the corresponding dimension is returned.
"""
Base.size(t::Tensor) = size(parent(t))
Base.size(t::Tensor, i) = size(parent(t), dim(t, i))

"""
    Base.length(::Tensor)

Return the length of the underlying array.
"""
Base.length(t::Tensor) = length(parent(t))

Base.axes(t::Tensor) = axes(parent(t))
Base.axes(t::Tensor, d) = axes(parent(t), dim(t, d))

# StridedArrays interface
Base.strides(t::Tensor) = strides(parent(t))
Base.stride(t::Tensor, i) = stride(parent(t), dim(t, i))

# fix ambiguity
Base.stride(t::Tensor, i::Integer) = stride(parent(t), i)

Base.unsafe_convert(::Type{Ptr{T}}, t::Tensor{T}) where {T} = Base.unsafe_convert(Ptr{T}, parent(t))

Base.elsize(T::Type{<:Tensor}) = Base.elsize(parent_type(T))

# Broadcasting
Base.BroadcastStyle(::Type{T}) where {T<:Tensor} = ArrayStyle{T}()

function Base.similar(bc::Broadcasted{ArrayStyle{Tensor{T,N,A}}}, ::Type{ElType}) where {T,N,A,ElType}
    # NOTE already checked if dimension mismatch
    # TODO throw on label mismatch?
    tensor = first(arg for arg in bc.args if arg isa Tensor{T,N,A})
    return similar(tensor, ElType)
end

"""
    Base.selectdim(tensor::Tensor, dim::Index, i)
    Base.selectdim(tensor::Tensor, dim::Integer, i)

Return a view of the tensor where the index for dimension `dim` equals `i`.

!!! note

    This method doesn't return a `SubArray`, but a `Tensor` wrapping a `SubArray`.

See also: [`selectdim`](@ref)
"""
Base.selectdim(t::Tensor, d::Integer, i) = Tensor(selectdim(parent(t), d, i), inds(t))
function Base.selectdim(t::Tensor, d::Integer, i::Integer)
    data = selectdim(parent(t), d, i)
    indices = [label for (i, label) in enumerate(inds(t)) if i != d]
    return Tensor(data, indices)
end

Base.selectdim(t::Tensor, d, i) = selectdim(t, dim(t, d), i)

"""
    Base.permutedims(tensor::Tensor, perm)

Permute the dimensions of `tensor` according to the given permutation `perm`. The [`inds`](@ref) will be permuted accordingly.
"""
function Base.permutedims(t::Tensor, perm)
    _inds = Index[]
    for i in perm
        push!(_inds, inds(t)[i])
    end
    Tensor(permutedims(parent(t), perm), _inds)
end

# shortcut for 0-dimensional tensors
Base.permutedims(t::Tensor{T,0}, _) where {T} = t
Base.permutedims(t::Tensor{T,0}, ::Base.AbstractVecOrTuple{Index}) where {T} = t

Base.permutedims!(dest::Tensor, src::Tensor, perm) = permutedims!(parent(dest), parent(src), perm)

function Base.permutedims(t::Tensor{T}, perm::Base.AbstractVecOrTuple{Index}) where {T}
    perm = map(i -> findfirst(==(i), inds(t)), perm)
    return permutedims(t, perm)
end

"""
    Base.dropdims(tensor::Tensor; dims)

Return a tensor where the dimensions specified by `dims` are removed. `size(tensor, dim) == 1` for each dimension in `dims`.
"""
function Base.dropdims(t::Tensor; dims=tuple(findall(==(1), size(t))...))
    return Tensor(dropdims(parent(t); dims), inds(t)[setdiff(1:ndims(t), dims)])
end

"""
    Base.view(tensor::Tensor, i...)
    Base.view(tensor::Tensor, inds::Pair{<:Index,<:Any}...)

Return a view of the tensor with the given indices. If a `Pair` is given, the index is replaced by the value of the pair.

!!! note

    This method doesn't return a `SubArray`, but a `Tensor` wrapping a `SubArray`.
"""
function Base.view(t::Tensor, i...)
    return Tensor(view(parent(t), i...), [label for (label, j) in zip(inds(t), i) if !(j isa Integer)])
end

function Base.view(t::Tensor, indices::P...) where {I<:Index,P<:Base.Pair{I}}
    indices = map(inds(t)) do ind
        i = findfirst(x -> x == ind, first.(indices))
        !isnothing(i) ? indices[i].second : Colon()
    end

    let data = view(parent(t), indices...),
        indices = [label for (index, label) in zip(indices, inds(t)) if !(index isa Integer)]

        Tensor(data, indices)
    end
end

# NOTE: `conj` is automatically managed because `Tensor` inherits from `AbstractArray`,
# but there is a bug when calling `conj` on `Tensor{T,0}` which makes it return a `Tensor{Tensor{Complex, 0}, 0}`
"""
    Base.conj(::Tensor)

Return the conjugate of the tensor.
"""
Base.conj(x::Tensor{<:Complex,0}) = Tensor(conj(parent(x)), Index[])

"""
    Base.adjoint(::Tensor)

Return the adjoint of the tensor.

!!! note

    This method doesn't transpose the array. It is equivalent to [`conj`](@ref).
"""
Base.adjoint(t::Tensor) = conj(t)

# NOTE: Maybe use transpose for lazy transposition ?
Base.transpose(t::Tensor{T,1,A}) where {T,A<:AbstractArray{T,1}} = copy(t)
Base.transpose(t::Tensor{T,2,A}) where {T,A<:AbstractArray{T,2}} = Tensor(transpose(parent(t)), reverse(inds(t)))

"""
    extend(tensor::Tensor; label[, axis=1, size=1, method=:zeros])

Expand the tensor by adding a new dimension `label` with the given `size` at the specified `axis`.
Currently the supported methods are `:zeros` and `:repeat`.
"""
function extend(tensor::Tensor; label, axis=1, size=1, method=:zeros)
    array = parent(tensor)
    data = if size == 1
        reshape(array, Base.size(array)[1:(axis - 1)]..., 1, Base.size(array)[axis:end]...)
    elseif method === :zeros
        extend_zeros(array, axis, size)
    elseif method === :repeat
        extend_repeat(array, axis, size)
    else
        # method === :identity ? __extend_identity(array, axis, size) :
        throw(ArgumentError("method \"$method\" is not valid"))
    end

    indices = (inds(tensor)[1:(axis - 1)]..., label, inds(tensor)[axis:end]...)

    return Tensor(data, indices)
end

function extend_zeros(array, axis, size)
    new = zeros(eltype(array), Base.size(array)[1:(axis - 1)]..., size, Base.size(array)[axis:end]...)

    view = selectdim(new, axis, 1)
    copy!(view, array)

    return new
end

function extend_repeat(array, axis, size)
    return repeat(
        reshape(array, Base.size(array)[1:(axis - 1)]..., 1, Base.size(array)[axis:end]...);
        outer=(fill(1, axis - 1)..., size, fill(1, ndims(array) - axis + 1)...),
    )
end

# TODO expand on more than 1 axis
"""
    expand(tensor::Tensor, ind::Index, size; method=:zeros)

Pad the tensor along the dimension specified by `ind` to reach new `size`.
Supported methods are `:zeros` and `:rand`.
"""
function expand(tensor::Tensor, ind::Index, _size; method=:zeros)
    @assert size(tensor, ind) <= _size "New size $_size of index $ind must be bigger than or equal to $(size(tensor, ind))"
    size(tensor, ind) == _size && return tensor # TODO return copy instead?
    axis = dim(tensor, ind)

    # TODO use `similar` to do just 1 allocation and set via `views`
    pad_size = [i == axis ? _size - size(tensor, i) : size(tensor, i) for i in 1:ndims(tensor)]
    pad_data = if method === :zeros
        zeros(eltype(tensor), Tuple(pad_size))
    elseif method === :rand
        rand(eltype(tensor), Tuple(pad_size))
    else
        throw(ArgumentError("method \"$method\" is not valid"))
    end

    # TODO expand in more axis?
    new_data = cat(parent(tensor), pad_data; dims=axis)
    return Tensor(new_data, inds(tensor))
end

Base.cat(tensor::Tensor) = tensor

"""
    Base.cat(a::Tensor, b::Tensor; dims)

Concatenate two tensors `a` and `b` along the specified dimensions `dims`.

The indices of the tensors must be equal, otherwise the second tensor will be permuted to match the first one.

!!! note

    `dims` must be a list of `Index`.
"""
function Base.cat(a::Tensor, b::Tensor; dims)
    dims = dims isa Index ? [dims] : dims
    @assert issetequal(inds(a), inds(b)) "Indices of tensors must be equal, got $(inds(a)) and $(inds(b))"
    @assert all(i -> size(a, i) == size(b, i), setdiff(inds(a), dims)) "Sizes of tensors must be equal in all dimensions except for the concatenation dimensions"

    if inds(a) != inds(b)
        b = permutedims(b, inds(a))
    end

    _dims = map(Base.Fix1(dim, a), dims)
    data = cat(parent(a), parent(b); dims=_dims)
    return Tensor(data, inds(a))
end

Base.cat(tensors::Tensor...; kwargs...) = foldl((a, b) -> cat(a, b; kwargs...), tensors)

LinearAlgebra.opnorm(x::Tensor, p::Real) = opnorm(parent(x), p)

# TODO choose a new index name? currently choosing the first index of `parinds`
"""
    fuse(tensor, parinds; ind=first(parinds))

Fuses `parinds`, leaves them on the right-side internally permuted with `permutator` and names it as `ind`.
"""
function fuse(tensor::Tensor, parinds; ind=first(parinds))
    @assert allunique(inds(tensor))
    @assert parinds ⊆ inds(tensor)

    locs = findall(∈(parinds), inds(tensor))
    perm = filter(∉(locs), 1:ndims(tensor))
    append!(perm, map(i -> findfirst(==(i), inds(tensor)), parinds))

    data = perm == 1:ndims(tensor) ? parent(tensor) : permutedims(parent(tensor), perm)
    data = reshape(data, (size(data)[1:(ndims(data) - length(parinds))]..., :))

    newinds = [filter(∉(parinds), inds(tensor))..., ind]
    return Tensor(data, newinds)
end

function Base._mapreduce_dim(f, op, init, tensor::Tensor, ind::Index)
    Base._mapreduce_dim(f, op, init, parent(tensor), dim(tensor, ind))
end
function Base._mapreduce_dim(f, op, init, tensor::Tensor, c::Colon)
    Base._mapreduce_dim(f, op, init, parent(tensor), c)
end
function Base._mapreduce_dim(f, op, init, tensor::Tensor, dims)
    Base._mapreduce_dim(f, op, init, parent(tensor), dim.((tensor,), dims))
end

# fix for ambiguity
function Base._mapreduce_dim(f, op, init::Base._InitialValue, t::Tensor, c::Colon)
    Base._mapreduce_dim(f, op, init, parent(t), c)
end

Base._sum(x::Tensor, ind::Index; kwargs...) = Tensor(Base._sum(parent(x), dim(x, ind); kwargs...), inds(x))
Base._sum(x::Tensor, c::Colon; kwargs...) = Tensor(fill(Base._sum(parent(x), c; kwargs...)))
Base._sum(x::Tensor, dims; kwargs...) = Tensor(Base._sum(parent(x), dim.((x,), dims); kwargs...), inds(x))

function isisometry(tensor::Tensor, ind; atol::Real=1e-12)
    # legacy behavior
    if isnothing(ind)
        return isapprox(parent(binary_einsum(tensor, conj(tensor))), fill(true); atol)
    end

    @assert ind in inds(tensor) "Index $ind is not in the tensor indices $(inds(tensor))"

    inda, indb = Index(gensym(:a)), Index(gensym(:b))
    a = replace(tensor, ind => inda)
    b = replace(conj(tensor), ind => indb)

    n = size(tensor, ind)
    contracted = binary_einsum(a, b)

    return isapprox(contracted, LinearAlgebra.I(n); atol)
end
