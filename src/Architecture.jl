using ArgCheck
using Adapt
using LinearAlgebra: LinearAlgebra
using ScopedValues

abstract type Vendor end
struct Intel <: Vendor end
struct AMD <: Vendor end
struct NVIDIA <: Vendor end

abstract type MemorySpace end
struct CPUMemorySpace <: MemorySpace end
struct ReactantMemorySpace <: MemorySpace end
struct CUDAMemorySpace <: MemorySpace end

memory_space(::T) where {T<:AbstractArray} = memory_space(T)
memory_space(::Type{<:Array}) = CPUMemorySpace()
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

# Base.promote_rule(::Type{CPUMemorySpace}, ::Type{CPUMemorySpace}) = CPUMemorySpace
# Base.promote_rule(::Type{CPUMemorySpace}, ::Type{CUDAMemorySpace}) = CUDAMemorySpace
# Base.promote_rule(::Type{CPUMemorySpace}, ::Type{ReactantMemorySpace}) = ReactantMemorySpace

# promote_memspace(::A, ::B) where {A<:MemorySpace,B<:MemorySpace} = promote_type(A, B)()

# promote_memspace(a, b, c, args...) = promote_memspace(promote_memspace(a, b), c, args...)
# function promote_memspace(a::AbstractArray, b::AbstractArray)
#     target_memspace = promote_memspace(memory_space(a), memory_space(b))
#     return adapt_memspace(target_memspace, a), adapt_memspace(target_memspace, b)
# end

# # TODO promote_memspace for Tensor

# adapt_memspace(::CPUMemorySpace, x::AbstractArray) = memory_space(x) != CPUMemorySpace() ? adapt(Array, x) : x
# adapt_memspace(::CUDAMemorySpace, x::AbstractArray) = memory_space(x) != CUDAMemorySpace() ? adapt(CuArray, x) : x

abstract type Backend end

struct BackendBase <: Backend end
struct BackendCustom <: Backend end
struct BackendStrided <: Backend end
struct BackendOMEinsum <: Backend end
struct BackendCUDA <: Backend end
struct BackendCuTENSOR <: Backend end
struct BackendCuTensorNet <: Backend end
struct BackendReactant <: Backend end

# set of loaded backends available for use
const LOADED_BACKENDS = Set{Backend}([BackendBase()])
const loaded_backends_lock = ReentrantLock()
register_backend(backend::Backend) = @lock loaded_backends_lock push!(LOADED_BACKENDS, backend)

# set of backends that are allowed to be used for each operation
const ALLOWED_BACKENDS = Dict{Function,Set{Backend}}()
const allowed_backends_lock = ReentrantLock()

# default backend for each operation
const DEFAULT_BACKEND = Dict{Function,ScopedValue{Backend}}()
const default_backends_lock = ReentrantLock()

function choose_backend end
function choose_backend_rule end
function allowed_backends end

# choose_backend(f::Function, arrays::AbstractArray...) = choose_backend(f, arrays...)
choose_backend(f::Function, tensors::Tensor...) = choose_backend(f, parent.(tensors)...)
function choose_backend(f::Function, arrays::AbstractArray...)
    if hasmethod(choose_backend_rule, Tuple{typeof(f),typeof.(arrays)...})
        return choose_backend_rule(f, arrays...)
    elseif hasmethod(choose_backend_rule, Tuple{typeof(f),[Type{typeof(A)} for A in arrays]...})
        return choose_backend_rule(f, typeof.(arrays)...)
    else
        return choose_backend_rule(f, unwrap_type.(typeof.(arrays))...)
    end
end

# choose_backend(arrays::AbstractArray...) = choose_backend(unwrap_type.(arrays)...)
choose_backend(arrays...) = missing
