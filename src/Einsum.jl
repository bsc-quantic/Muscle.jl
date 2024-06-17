using LinearAlgebra
using OMEinsum: OMEinsum

function einsum(einstr::String, a, b)
    m = match(r"(?<a>\w*),(?<b>\w*)->(?<c>\w*)", einstr)
    ia = isnothing(m[:a]) ? Char[] : collect(m[:a])
    ib = isnothing(m[:b]) ? Char[] : collect(m[:b])
    ic = isnothing(m[:c]) ? Char[] : collect(m[:c])
    return einsum(ic, a, ia, b, ib)
end

function einsum(einstr::String, a)
    m = match(r"(?<a>\w*)->(?<c>\w*)", einstr)
    ia = collect(m[:a])
    ic = isnothing(m[:c]) ? Char[] : collect(m[:c])
    return einsum(ic, a, ia)
end

function einsum(ic, a, ia, b, ib)
    size_dict = [size(i in ia ? a : b, findfirst(==(i), i in ia ? ia : ib)) for i in ic]
    c = OMEinsum.get_output_array((a, b), size_dict; fillzero=false)
    return einsum!(c, ic, a, ia, b, ib)
end

function einsum(ic, a, ia)
    size_dict = [size(a, findfirst(==(i), ia)) for i in ic]
    c = OMEinsum.get_output_array((a,), size_dict; fillzero=false)
    return einsum!(c, ic, a, ia)
end

function einsum!(c, ic, a, ia, b, ib)
    ixs = (ia, ib)
    iy = ic
    xs = (a, b)
    y = c
    size_dict = Dict{Char,Int}([(ia .=> size(a))..., (ib .=> size(b))...])

    OMEinsum.einsum!(ixs, iy, xs, y, true, false, size_dict)

    return c
end

function einsum!(c, ic, a, ia)
    ixs = (ia,)
    iy = ic
    xs = (a,)
    y = c
    size_dict = Dict(ia .=> size(a))

    OMEinsum.einsum!(ixs, iy, xs, y, true, false, size_dict)

    return c
end
