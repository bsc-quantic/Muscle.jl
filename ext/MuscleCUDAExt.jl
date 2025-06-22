using CUDA, cuTensor
using cuTensorNet: cuTensorNet


memory_space(::Type{<:CuArray}) = CUDAMemorySpace()

choose_backend_rule(::typeof(binary_einsum), ::Type{<:CuArray}, ::Type{<:CuArray}) = BackendCuTENSOR()
choose_backend_rule(::typeof(binary_einsum!), ::Type{<:CuArray}, ::Type{<:CuArray}) = BackendCuTENSOR()
function choose_backend_rule(::typeof(binary_einsum!), ::Type{<:CuArray}, ::Type{<:CuArray}, ::Type{<:CuArray})
    BackendCuTENSOR()
end

choose_backend_rule(::typeof(unary_einsum), ::Type{<:CuArray}) = BackendOMEinsum()
choose_backend_rule(::typeof(unary_einsum!), ::Type{<:CuArray}, ::Type{<:CuArray}) = BackendOMEinsum()

choose_backend_rule(::typeof(tensor_qr_thin), ::Type{<:CuArray}) = BackendCuTensorNet()
function choose_backend_rule(::typeof(tensor_qr_thin!), ::Type{<:CuArray}, ::Type{<:CuArray}, ::Type{<:CuArray})
    BackendCuTensorNet()
end

function choose_backend_rule(::typeof(simple_update), ::Type{<:CuArray}, ::Type{<:CuArray}, ::Type{<:CuArray})
    BackendCuTensorNet()
end

choose_backend_rule(::typeof(tensor_svd_thin), ::Type{<:CuArray}) = BackendCuTensorNet()
function choose_backend_rule(
    ::typeof(tensor_svd_thin!), ::Type{<:CuArray}, ::Type{<:CuArray}, ::Type{<:CuArray}, ::Type{<:CuArray}
)
    BackendCuTensorNet()
end

