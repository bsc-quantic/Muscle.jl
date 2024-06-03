using LinearAlgebra
using OMEinsum: OMEinsum

function einsum(einstr::String, a, b)
    m = match(r"(?<a>\w+),(<b>\w+)->(<c>\w+)", einstr)
    return einsum(collect(m[:c]), a, collect(m[:a]), b, collect(m[:b]))
end

function einsum(einstr::String, a)
    m = match(r"(?<a>\w+)->(?<c>\w+)", einstr)
    return einsum(collect(m[:c]), a, collect(m[:a]))
end

function einsum(ic, a, ia, b, ib)
    c = OMEinsum.get_output_array((a, b), [...]; fillzero=false)
    return einsum!(c, ic, a, ia, b, ib)
end

function einsum(ic, a, ia)
    c = OMEinsum.get_output_array((a, b), [...]; fillzero=false)
    return einsum!(c, ic, a, ia)
end

function einsum!(c, ic, a, ia, b, ib)
    ixs = (ia, ib)
    iy = ic
    xs = (a, b)
    y = c
    size_dict = Dict([(ia .=> size(a))..., (ib .=> size(b))...])

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
