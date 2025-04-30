using ArgCheck
using CUDA
using LinearAlgebra: LinearAlgebra

abstract type Vendor end
struct Intel <: Vendor end
struct AMD <: Vendor end
struct NVIDIA <: Vendor end

abstract type MemorySpace end
struct DefaultMemorySpace <: MemorySpace end
struct CUDAMemorySpace <: MemorySpace end
struct ReactantMemorySpace <: MemorySpace end

memory_space(::Array) = DefaultMemorySpace()
memory_space(x::Base.ReshapedArray) = memory_space(parent(x))
memory_space(x::LinearAlgebra.Transpose) = memory_space(parent(x))
memory_space(::CuArray) = CUDAMemorySpace()

abstract type Architecture end
struct CPU <: Architecture end
struct GPU <: Architecture end

arch(x::AbstractArray) = arch(memory_space(x))
arch(::DefaultMemorySpace) = CPU()
arch(::CUDAMemorySpace) = GPU()

# TODO promote memspace
function promote_memspace(a, b)
    @argcheck memspace(a) == memspace(b) "Memory spaces must be the same"
    return a, b
end
