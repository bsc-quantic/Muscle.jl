using ArgCheck
using ScopedValues

abstract type Backend end

struct BackendDefault <: Backend end
struct BackendBase <: Backend end
struct BackendStrided <: Backend end
struct BackendOMEinsum <: Backend end
struct BackendCUDA <: Backend end
struct BackendCuTENSOR <: Backend end
struct BackendCuTensorNet <: Backend end
struct BackendReactant <: Backend end
struct BackendDagger <: Backend end

const CURRENT_BACKEND = ScopedValue{Backend}()

with_backend(f, backend::Backend) = with(f, CURRENT_BACKEND => backend)

# set of loaded backends available for use
# const LOADED_BACKENDS = Set{Backend}([BackendBase()])
# const LOADED_BACKENDS_LOCK = ReentrantLock()

function choose_backend end
# function allowed_backends end

# choose_backend(f::Function, arrays::AbstractArray...) = choose_backend(f, arrays...)
choose_backend(f::Function, tensors::Tensor...) = choose_backend(f, parent.(tensors)...)
function choose_backend(f::Function, arrays::AbstractArray...)
    if isassigned(CURRENT_BACKEND)
        return CURRENT_BACKEND[]
    end

    memspaces = Domain.(arrays)
    return choose_backend_rule(f, memspaces...)
end

# choose_backend(arrays::AbstractArray...) = choose_backend(unwrap_type.(arrays)...)
# choose_backend(arrays...) = missing

default_backend(op) = throw(ErrorException("No default backend defined for operation: $op"))

macro default_backend(op, backend)
    sop = Symbol(:current_, op)
    quote
        const $sop = ScopedValue(DefaultBackend())
        default_backend(::typeof($op)) = $backend
    end
end
