# Contraction

```@setup tensor
using Muscle
a = Tensor(reshape(collect(Float64.(1:4)), 2, 2), [:i, :j])
b = Tensor(reshape(collect(Float64.(10:13)), 2, 2), [:j, :i])
```

Einsum operations are performed automatically by [`unary_einsum`](@ref) and [`binary_einsum`](@ref).

Unlike other tensor libraries, the einsum pattern is not explicitly stated by the user but implicitly inferred from the `Tensor` objects; i.e. repeated indices will be contracted while unique indices will remain.
However, the user might require some flexibility on the output and contracted indices.
That's why [`unary_einsum`](@ref) and [`binary_einsum`](@ref) have two extra keyword arguments: `dims`, which lists the indices to be contracted, and `out`, which lists the resulting indices after the contraction.
Keep in mind that you're not forced to define them: `dims` defaults to the repeated indices and `out` defaults to the unique indices, but it's not recommended to define both.

For example, let's imagine that we want to perform the following operation: A sum over one dimension of a tensor.

```math
X_j = \sum_i A_{ij}
```

[`unary_einsum`](@ref) can act on just one tensor (unary contraction) and the user can write the following operation in two different ways:

```@repl tensor
Muscle.unary_einsum(a; dims=[Index(:i)])
Muscle.unary_einsum(a; out=[Index(:j)])
```

For the case of binary contraction, imagine the following matrix multiplication:

```math
Y_j = \sum_i A_{ij} B_{ji}
```

Then the default would be enough, although you can still define `dims` or `out`.

```@repl tensor
Muscle.binary_einsum(a, b)
Muscle.binary_einsum(a, b; dims=[Index(:i)])
Muscle.binary_einsum(a, b; out=[Index(:j)])
```

But what if instead of contracting index `:i`, we want to perform a Hadamard product (element-wise multiplication)? Then that's a case where implicit inference of the einsum rule is not enough and you need to specify `dims` or `out`.

```@repl tensor
Muscle.binary_einsum(a, b; dims=Index[])
Muscle.binary_einsum(a, b; out=[Index(:i), Index(:j)])
```
