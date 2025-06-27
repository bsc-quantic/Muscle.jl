module MuscleOMEinsumExt

using Muscle
using OMEinsum

# TODO add a preference system for some backends
# function Muscle.choose_backend_rule(::typeof(unary_einsum), ::Type{<:Array})
#     Muscle.BackendOMEinsum()
# end

# function Muscle.choose_backend_rule(::typeof(unary_einsum!), ::Type{<:Array}, ::Type{<:Array})
#     Muscle.BackendOMEinsum()
# end

# function Muscle.choose_backend_rule(::typeof(binary_einsum), ::Type{<:Array}, ::Type{<:Array})
#     Muscle.BackendOMEinsum()
# end

# function Muscle.choose_backend_rule(::typeof(binary_einsum!), ::Type{<:Array}, ::Type{<:Array}, ::Type{<:Array})
#     Muscle.BackendOMEinsum()
# end

function unary_einsum(::BackendOMEinsum, inds_y, x, inds_x)
    size_dict = Dict(inds_x .=> size(x))
    y = similar(x, Tuple(size_dict[i] for i in inds_y))
    unary_einsum!(BackendOMEinsum(), y, inds_y, x, inds_x)
    return y
end

function unary_einsum!(::BackendOMEinsum, y, inds_y, x, inds_x)
    size_dict = Dict(inds_x .=> size(x))
    einsum!((inds_x,), inds_y, (x,), y, true, false, size_dict)
    return y
end

function binary_einsum(::Muscle.BackendOMEinsum, inds_c, a, inds_a, b, inds_b)
    size_dict = Dict{Index,Int}()
    for (ind, ind_size) in Iterators.flatten([inds_a .=> size(a), inds_b .=> size(b)])
        size_dict[ind] = ind_size
    end

    c = OMEinsum.get_output_array((a, b), Int[size_dict[i] for i in inds_c], false)
    OMEinsum.einsum!((inds_a, inds_b), inds_c, (a, b), c, true, false, size_dict)
    return c
end

function binary_einsum!(::Muscle.BackendOMEinsum, c, inds_c, a, inds_a, b, inds_b)
    size_dict = Dict{Index,Int}()
    for (ind, ind_size) in Iterators.flatten([inds_a .=> size(a), inds_b .=> size(b)])
        size_dict[ind] = ind_size
    end

    OMEinsum.einsum!((inds_a, inds_b), inds_c, (a, b), c, true, false, size_dict)
    return c
end

end
