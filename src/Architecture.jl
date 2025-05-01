using ArgCheck
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

memory_space(::Array) = CPUMemorySpace()
memory_space(::CuArray) = CUDAMemorySpace()

# dispatches for layouts
memory_space(x::Base.ReshapedArray) = memory_space(parent(x))
memory_space(x::Base.SubArray) = memory_space(parent(x))
memory_space(x::Base.PermutedDimsArray) = memory_space(parent(x))
memory_space(x::LinearAlgebra.Transpose) = memory_space(parent(x))
memory_space(x::LinearAlgebra.Adjoint) = memory_space(parent(x))
memory_space(x::LinearAlgebra.Diagonal) = memory_space(parent(x))
memory_space(x::LinearAlgebra.Bidiagonal) = memory_space(parent(x))
memory_space(x::LinearAlgebra.SymTridiagonal) = memory_space(parent(x))
memory_space(x::LinearAlgebra.Tridiagonal) = memory_space(parent(x))
memory_space(x::LinearAlgebra.Symmetric) = memory_space(parent(x))
memory_space(x::LinearAlgebra.Hermitian) = memory_space(parent(x))
memory_space(x::LinearAlgebra.LowerTriangular) = memory_space(parent(x))
memory_space(x::LinearAlgebra.UpperTriangular) = memory_space(parent(x))
memory_space(x::LinearAlgebra.UnitLowerTriangular) = memory_space(parent(x))
memory_space(x::LinearAlgebra.UnitUpperTriangular) = memory_space(parent(x))
memory_space(x::LinearAlgebra.UpperHessenberg) = memory_space(parent(x))

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
