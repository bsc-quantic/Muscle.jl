using ArgCheck
using Adapt
using CUDA
using LinearAlgebra: LinearAlgebra

abstract type Vendor end
struct Intel <: Vendor end
struct AMD <: Vendor end
struct NVIDIA <: Vendor end

abstract type MemorySpace end
struct CPUMemorySpace <: MemorySpace end
struct CUDAMemorySpace <: MemorySpace end
struct ReactantMemorySpace <: MemorySpace end

memory_space(::T) where {T<:AbstractArray} = memory_space(T)
memory_space(::Type{<:Array}) = CPUMemorySpace()
memory_space(::Type{<:CuArray}) = CUDAMemorySpace()
memory_space(::Type{T}) where {T<:WrappedArray} = memory_space(Adapt.unwrap_type(T))
memory_space(x::Tensor) = memory_space(parent_type(x))

abstract type Architecture end
struct CPU <: Architecture end
struct GPU <: Architecture end

arch(x::AbstractArray) = arch(memory_space(x))
arch(::CPUMemorySpace) = CPU()
arch(::CUDAMemorySpace) = GPU()

# TODO promote memspace
function promote_memspace(a, b)
    @argcheck memory_space(a) == memory_space(b) "Memory spaces must be the same"
    return a, b
end
