module MuscleVectorInterfaceExt

using Muscle
using VectorInterface
using ArgCheck

VectorInterface.scalartype(::Tensor{T}) where {T} = T
VectorInterface.zerovector(t::Tensor, S) = Tensor(zerovector(parent(t), S), inds(t))
VectorInterface.zerovector!(t::Tensor) = t .= zero(eltype(t))
VectorInterface.scale(x::Tensor, α) = Tensor(scale(parent(x), α), inds(x))
VectorInterface.scale!(x::Tensor, α) = x .*= α
VectorInterface.scale!(y::Tensor, x::Tensor, α) = y .= α .* x
VectorInterface.add(y::Tensor, x::Tensor) = y + x
VectorInterface.add(y::Tensor, x::Tensor, α::Number) = y + α * x
VectorInterface.add(y::Tensor, x::Tensor, α::Number, β::Number) = y * β + α * x
VectorInterface.add!(y::Tensor, x::Tensor) = y .= y + x
VectorInterface.add!(y::Tensor, x::Tensor, α::Number) = y .= y + α * x
VectorInterface.add!(y::Tensor, x::Tensor, α::Number, β::Number) = y .= y * β + α * x

function VectorInterface.inner(x::Tensor, y::Tensor)
    @argcheck issetequal(inds(x), inds(y))
    return binary_einsum(x, y)
end

end
