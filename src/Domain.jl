using Adapt
using ArgCheck

abstract type Domain end

struct DomainHost <: Domain end
struct DomainCUDA <: Domain end
struct DomainReactant <: Domain end
struct DomainDagger <: Domain end

Domain(::T) where {T<:AbstractArray} = Domain(T)
Domain(::Type{<:Array}) = DomainHost()
Domain(::Type{T}) where {T<:WrappedArray} = Domain(Adapt.unwrap_type(T))
Domain(x::Tensor) = Domain(parent_type(x))

# TODO promote memspace
function promote_domain(a, b)
    @argcheck Domain(a) == Domain(b) "Domain must be the same"
    return a, b
end

# Base.promote_rule(::Type{DomainHost}, ::Type{DomainHost}) = DomainHost
# Base.promote_rule(::Type{DomainHost}, ::Type{DomainCUDA}) = DomainCUDA
# Base.promote_rule(::Type{DomainHost}, ::Type{DomainReactant}) = DomainReactant

# promote_domain(::A, ::B) where {A<:Domain,B<:Domain} = promote_type(A, B)()

# promote_domain(a, b, c, args...) = promote_domain(promote_domain(a, b), c, args...)
# function promote_domain(a::AbstractArray, b::AbstractArray)
#     target_memspace = promote_domain(memory_space(a), memory_space(b))
#     return adapt_memspace(target_memspace, a), adapt_memspace(target_memspace, b)
# end

# # TODO promote_domain for Tensor

# adapt_memspace(::DomainHost, x::AbstractArray) = memory_space(x) != DomainHost() ? adapt(Array, x) : x
# adapt_memspace(::DomainCUDA, x::AbstractArray) = memory_space(x) != DomainCUDA() ? adapt(CuArray, x) : x
