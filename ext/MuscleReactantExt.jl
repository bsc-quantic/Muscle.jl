module MuscleReactantExt

using Muscle
using Reactant
const MLIR = Reactant.MLIR
const stablehlo = MLIR.Dialects.stablehlo

function Muscle.einsum(ic, @nospecialize(a::Reactant.TracedRArray{Ta,Sa,Na}), ia, @nospecialize(b::Reactant.TracedRArray{Tb,Sb,Nb}), ib) where {Ta,Sa,Na,Tb,Sb,Nb}
    T = promote_type(Ta, Tb)
    mlirty = MLIR.IR.Type(T)

    rsize = Tuple(i ∈ ia ? size(a, findfirst(==(i), ia)) : size(b, findfirst(==(i), ib)) for i in ic)
    result_0 = MLIR.IR.TensorType(rsize, mlirty)
    einsum_config = MLIR.IR.Attribute("$(join(ia)),$(join(ib))->$(join(ic))")

    # TODO replace with `dot_general` + `transpose`
    result = MLIR.IR.result(stablehlo.einsum(a.mlir_data, b.mlir_data; result_0, einsum_config))

    return Reactant.TracedRArray{T,rsize,length(ic)}((), result)
end

function Muscle.einsum(ic, @nospecialize(a::Reactant.TracedRArray{T,S,N}), ia) where {T,S,N}
    mlirty = MLIR.IR.Type(T)

    rsize = Tuple(size(a, findfirst(==(i), ia)) for i in ic)
    result_0 = MLIR.IR.TensorType(rsize, mlirty)
    einsum_config = MLIR.IR.Attribute("$(join(ia))->$(join(ic))")

    # TODO replace with `reduce`
    result = MLIR.IR.result(stablehlo.unary_einsum(a.mlir_data; result_0, einsum_config))

    return Reactant.TracedRArray{T,rsize,length(ic)}((), result)
end

end
