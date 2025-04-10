using CUDA

abstract type Vendor end
struct Intel <: Vendor end
struct AMD <: Vendor end
struct NVIDIA <: Vendor end

abstract type Architecture end
struct CPU <: Architecture end
struct GPU <: Architecture end

abstract type MemorySpace end
struct DefaultMemorySpace <: MemorySpace end
struct CUDAMemorySpace <: MemorySpace end
struct ReactantMemorySpace <: MemorySpace end

memoryspace(::Array) = DefaultMemorySpace()
memoryspace(::CuArray) = CUDAMemorySpace()
