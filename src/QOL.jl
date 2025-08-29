import Base: *

*(a::Tensor, b::Tensor) = binary_einsum(a,b)


tags(t::Tensor) = [i.tag for i in inds(t)]

function is_iterable(x)
    try
        iterate(x)
        return true
    catch
        return false
    end
end
function to_vec(x)
    x = is_iterable(x) ? x : (x,)
    x = x isa String ? (x,) : x
end



# Prettier(?) printing for Tensors 
function Base.show(io::IO, ::MIME"text/plain", t::Tensor{T}) where T
    if ndims(t) == 0
        println(io, " $(only(t))  - rank 0 Tensor{$T}")
    else
    println(io, "$(ndims(t)) Inds = ", [":$(i.tag)($(size(t, i)))" for i in inds(t)])
    println(io, "Tensor{$T}, first element = $(first(t)) ")
    end
    # println(io, " size  | Index")
    # for i in inds(t)
    #     println(io, "$(lpad(string(size(t, i)),6)) | $(i.tag)")
    # end

    # Optional: print if small tensor ? 
    # if length(t) < 100
    #     # For small tensors we can afford to call the default show
    #     invoke(Base.show, Tuple{IO, Any}, io, t)
    # end
end

function makeinds(n::Int)
    Index.([Symbol(Char(c)) for c in 'i':'z'])[1:n]
end


#sorry
function prime(i::Index; prime_tag::String="*")
    new_tag = string(i.tag)*prime_tag
    Index(new_tag)
end
function prime(t::Tensor; kwargs...)
    t = replace(t, Pair.(inds(t), prime.(inds(t); kwargs...))...)
end
function prime(t::Tensor, i; kwargs...)
    i = Index(i)
    t = replace(t, i => prime(i; kwargs...))
end



function unprime(i::Index; prime_tag::String="*")
    if !endswith(string(i.tag), prime_tag)
        @warn "Index not primed, doing nothing"
        return i 
    end
    
    new_tag = string(i.tag)[1:end-length(prime_tag)]  # avoid chop() and substrings
    i = Index(new_tag)
end
function unprime(t::Tensor; kwargs...)
    t = replace(t, Pair.(inds(t), unprime.(inds(t); kwargs...))...)
end
function unprime(t::Tensor, i; kwargs...)
    i = Index(i)
    t = replace(t, i => unprime(i; kwargs...))
end


""" Contract *only* the indices specified. 
Need to figure out behavior for when repeated indices occur """
function contract(a::Tensor, b::Tensor, contract_inds_a=nothing, contract_inds_b=nothing)

    # try to build indices if we pass symbols or something else - this uses that Index(Index(:x)) == Index(:x)
    contract_inds_a = Index.(to_vec(contract_inds_a))
    contract_inds_b = Index.(to_vec(contract_inds_b))

    # check that inds are where they belong
    @assert issubset(contract_inds_a, inds(a))
    @assert issubset(contract_inds_b, inds(b))

    b = replace(prime(b), Pair.(prime.(contract_inds_b), contract_inds_a)...)

    primed = setdiff(inds(b), contract_inds_a)

    #@show tags(b)

    c = binary_einsum(a, b)

    for ip in primed
        c = unprime(c, ip)
    end
 
    #silly check
    if ndims(c) > ndims(a) + ndims(b) - length(contract_inds_a)
        @warn "Not all inds contracted? "
    end
    return c
end