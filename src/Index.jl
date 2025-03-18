abstract type Index end

isindex(::Index) = true
isindex(::Any) = false

struct NamedIndex <: Index
    name::Symbol
end

NamedIndex(name::String) = NamedIndex(Symbol(name))

Base.copy(x::NamedIndex) = x

# Site interface
"""
    Site(id)
    Site(i, j, ...)
    lane"i,j,..."

Represents the location of a physical index.

See also: [`Site`](@ref), [`sites`](@ref)
"""
struct Site{N} <: Index
    id::NTuple{N,Int}
end

Site(site::Site) = site
Site(id::Int) = Site((id,))
Site(id::Vararg{Int,N}) where {N} = Site(id)
Site(id::CartesianIndex) = Site(Tuple(id))

Base.copy(x::Site) = x

issite(::Site) = true
issite(::Index) = false

site(x::Site) = x

Base.isless(a::Site, b::Site) = a.id < b.id
Base.isless(a::Index, b::Index) = site(a) < site(b)

Base.ndims(::Site{N}) where {N} = N
Base.ndims(x::Index) = ndims(site(x))

Base.show(io::IO, x::Site) = print(io, "$(x.id)")

"""
    site"i,j,..."

Constructs a [`Site`](@ref) object with the given coordinates. The coordinates are given as a comma-separated list of integers.
"""
macro site_str(str)
    expr = Meta.parse(str)
    @assert Meta.isexpr(expr, :tuple) "Invalid site string"
    return :(Site($(expr.args...)))
end

# Bond interface
struct Bond{N} <: Index
    src::Site{N}
    dst::Site{N}

    Bond{N}(src::Site{N}, dst::Site{N}) where {N} = new{N}(minmax(src, dst)...)
end

Base.copy(x::Bond) = x

isbond(::Bond) = true
isbond(::Index) = false

bond(x::Bond) = x

Base.ndims(::Bond{N}) where {N} = N

function Base.getindex(bond::Bond, i::Int)
    if i == 1
        return bond.src
    elseif i == 2
        return bond.dst
    else
        throw(BoundsError(bond, i))
    end
end

Base.show(io::IO, x::Bond) = print(io, "$(x.src) <=> $(x.dst)")

Pair(e::Bond) = e.src => e.dst
Tuple(e::Bond) = (e.src, e.dst)

function Base.iterate(bond::Bond, state=0)
    if state == 0
        (bond.src, 1)
    elseif state == 1
        (bond.dst, 2)
    else
        nothing
    end
end

Base.IteratorSize(::Type{Bond}) = Base.HasLength()
Base.length(::Bond) = 2
Base.IteratorEltype(::Type{Bond{L}}) where {L} = Base.HasEltype()
Base.eltype(::Bond{L}) where {L} = L
Base.isdone(::Bond, state) = state == 2

hassite(bond::Bond, site::Site) = site == bond.src || site == bond.dst
sites(bond::Bond) = (bond.src, bond.dst)

# Plug interface
"""
    Plug(id[; dual = false])
    Plug(i, j, ...[; dual = false])

Represents a [`Site`](@ref) with an annotation of input or output.
`Site` objects are used to label the indices of tensors in a [`Quantum`](@ref) Tensor Network.

See also: [`Site`](@ref), [`plugs`](@ref), [`isdual`](@ref)
"""
Base.@kwdef struct Plug{N} <: Index
    site::Site{N}
    isdual::Bool = false
end

Plug(id::Int; kwargs...) = Plug(Site(id); kwargs...)
Plug(@nospecialize(id::NTuple{N,Int}); kwargs...) where {N} = Plug(Site(id); kwargs...)
Plug(@nospecialize(id::Vararg{Int,N}); kwargs...) where {N} = Plug(Site(id); kwargs...)
Plug(@nospecialize(id::CartesianIndex); kwargs...) = Plug(Site(id); kwargs...)

Base.copy(x::Plug) = x

isplug(::Plug) = true
isplug(::Index) = false

isdual(x::Plug) = x.isdual
isdual(x::Index) = isdual(plug(x))

plug(x::Plug) = x
Base.adjoint(x::Plug) = Plug(lane(x); isdual=!isdual(x))

Base.show(io::IO, x::Plug) = print(io, "$(lane(x))$(isdual(x) ? "'" : "")")

"""
    plug"i,j,...[']"

Constructs a [`Site`](@ref) object with the given coordinates. The coordinates are given as a comma-separated list of integers. Optionally, a trailing `'` can be added to indicate that the site is a dual site (i.e. an "input").

See also: [`@site_str`](@ref)
"""
macro plug_str(str)
    isdual = endswith(str, '\'')
    str = chopsuffix(str, '\'')
    return :(Plug(site"$str"; isdual=$isdual))
end

# Moment interface
struct Moment{I} <: Index
    index::I
    t::Int
end

# Moment(index::I, t::Int) where {I} = Moment{I}(index, t)

Base.copy(x::Moment) = Moment(copy(x), x.t)

issite(x::Moment) = issite(x.index)
isbond(x::Moment) = isbond(x.index)
isdual(x::Moment) = isdual(x.index)

site(x::Moment) = site(x.index)
bond(x::Moment) = bond(x.index)

Base.show(io::IO, x::Moment) = print(io, "$(x.index) @ t=$(x.t)")
