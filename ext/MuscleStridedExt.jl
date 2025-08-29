module MuscleStridedExt

using Muscle
using Muscle: BackendStrided, arraytype
using Strided
using StridedViews
using ArgCheck

function Muscle.choose_backend_rule(::typeof(binary_einsum), ::Type{<:StridedView}, ::Type{<:Array})
    BackendStrided()
end

function Muscle.choose_backend_rule(::typeof(binary_einsum), ::Type{<:Array}, ::Type{<:StridedView})
    BackendStrided()
end

function Muscle.choose_backend_rule(::typeof(binary_einsum), ::Type{<:StridedView}, ::Type{<:StridedView})
    BackendStrided()
end

function Muscle.choose_backend_rule(
    ::typeof(binary_einsum!), ::Type{<:StridedView}, ::Type{<:StridedView}, ::Type{<:StridedView}
)
    BackendStrided()
end

function Muscle.binary_einsum(::BackendStrided, inds_c, a::Tensor, b::Tensor; kwargs...)
    binary_einsum(
        BackendStrided(),
        inds_c,
        arraytype(a) isa StridedView ? a : Tensor(StridedView(parent(a)), inds(a)),
        arraytype(b) isa StridedView ? b : Tensor(StridedView(parent(b)), inds(b)),
    )
end

function Muscle.binary_einsum(
    ::BackendStrided, inds_c, a::Tensor{Ta,Na,<:StridedView}, b::Tensor{Tb,Nb,<:StridedView}; kwargs...
) where {Ta,Tb,Na,Nb}
    inds_contract = inds(a) ∩ inds(b)
    inds_left = setdiff(inds(a), inds_contract)
    inds_right = setdiff(inds(b), inds_contract)

    # can't deal with hyperindices
    @argcheck isdisjoint(inds_c, inds_contract)
    @argcheck issetequal(inds_c, symdiff(inds(a), inds(b)))

    sizes_c = map(inds_c) do ind
        if ind in inds_left
            return size(a, ind)
        elseif ind in inds_right
            return size(b, ind)
        else
            throw(ArgumentError("Index $ind not found in either tensor"))
        end
    end

    sa = a
    sb = b
    sc = Tensor(StridedView(zeros(Base.promote_eltype(a, b), Tuple(sizes_c)), Tuple(sizes_c)), inds_c)

    binary_einsum!(BackendStrided(), sc, sa, sb)
    return sc
end

function Muscle.binary_einsum!(
    ::BackendStrided, c::Tensor{Tc,Nc,<:StridedView}, a::Tensor{Ta,Na,<:StridedView}, b::Tensor{Tb,Nb,<:StridedView}
) where {Tc,Nc,Ta,Tb,Na,Nb}
    inds_contract = inds(a) ∩ inds(b)
    inds_left = setdiff(inds(a), inds_contract)
    inds_right = setdiff(inds(b), inds_contract)

    # can't deal with hyperindices
    @argcheck isdisjoint(inds(c), inds_contract)
    @argcheck issetequal(inds(c), symdiff(inds(a), inds(b)))

    sizes_left = map(Base.Fix1(size, a), inds_left)
    sizes_right = map(Base.Fix1(size, b), inds_right)
    sizes_contract = map(Base.Fix1(size, a), inds_contract)

    sa = a
    sb = b
    sc = c

    sa_mat = parent(permutedims(sa, [inds_left; inds_contract]))
    sa_mat = sreshape(sa_mat, Tuple([sizes_left; ones(Int, length(sizes_right)); sizes_contract]))

    sb_mat = parent(permutedims(sb, [inds_right; inds_contract]))
    sb_mat = sreshape(sb_mat, Tuple([ones(Int, length(sizes_left)); sizes_right; sizes_contract]))

    sc_mat = parent(permutedims(sc, [inds_left; inds_right]))
    sc_mat = sreshape(sc_mat, Tuple([sizes_left; sizes_right; ones(Int, length(sizes_contract))]))

    tsize = (sizes_left..., sizes_right..., sizes_contract...)
    Strided._mapreducedim!(*, +, zero, tsize, (sc_mat, sa_mat, sb_mat))

    sc_mat = reshape(sc_mat, Tuple([sizes_left; sizes_right]))
    sc = Tensor(sc_mat, [inds_left; inds_right])
    sc = permutedims(sc, inds(c))
    return sc
end

end
