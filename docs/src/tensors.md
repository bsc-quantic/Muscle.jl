# Tensors

```@setup tensor
using Muscle
```

If you have reached here, you probably know what a tensor is, and probably have heard many jokes about _what a tensor is_[^1]. Nevertheless, we are gonna give a brief remainder.

[^1]: For example, recursive definitions like _a tensor is whatever that transforms as a tensor_.

A tensor $T$ of order[^2] $n$ is a multilinear[^3] function between $n$ vector spaces over a field $\mathcal{F}$.

[^2]: The _order_ of a tensor may also be known as _rank_ or _dimensionality_ in other fields. However, these can be missleading, since it has nothing to do with the _rank_ of linear algebra nor with the _dimensionality_ of a vector space. Thus we prefer to use the word _order_.
[^3]: Meaning that the relationships between the output and the inputs, and the inputs between them, are linear.

```math
T : \mathcal{F}^{\dim(1)} \times \dots \times \mathcal{F}^{\dim(n)} \mapsto \mathcal{F}
```

In layman's terms, you can view a tensor as a linear function that maps a set of vectors to a scalar.

```math
T(\mathbf{v}^{(1)}, \dots, \mathbf{v}^{(n)}) = c \in \mathcal{F} \qquad\qquad \forall i, \mathbf{v}^{(i)} \in \mathcal{F}^{\dim(i)}
```

Just like with matrices and vectors, $n$-dimensional arrays of numbers can be used to represent tensors. Furthermore, scalars, vectors and matrices can be viewed as tensors of order 0, 1 and 2, respectively.

The dimensions of the tensors are usually identified with labels and known as tensor indices or just indices. By appropeately fixing the indices in a expression, a lot of different linear algebra operations can be described.

For example, the trace operation...

```math
tr(A) = \sum_i A_{ii}
```

... a tranposition of dimensions...

```math
A_{ji} = A^T_{ij}
```

... or a matrix multiplication.

```math
C_{ik} = \sum_j A_{ij} B_{jk}
```

## The `Tensor` type

In `Muscle`, a tensor is represented by the `Tensor` type, which wraps an array and a list of [`Index`](@ref). An [`Index`](@ref) is wrapper type that can fit anything in it.

```@repl tensor
Index(:i)
Index(1)
Index((1,2))
```

You can create a `Tensor` by passing an `AbstractArray` and a `Vector` or `Tuple` of `Index`s.

```@repl tensor
t = Tensor(rand(3,5,2), [Index(:i), Index(:j), Index(:k)])
```

!!! tip
    For backward compatibility, if all the indices are `Index{Symbol}`s, you can directly pass a list of `Symbols` to the `Tensor` constructor. For example,
    ```@repl tensor
    Tensor(rand(3,2), [:i, :j])
    ```

Use `parent` and `inds` to access the underlying array and indices respectively.

```@repl tensor
parent(t)
inds(t)
```

The _dimensionality_ or size of each index can be consulted using the `size` function.

```@repl tensor
size(t)
size(t, Index(:j))
length(t)
```

Note that these indices are the ones that really define the dimensions of the tensor and not the order of the array dimensions.

```@repl tensor
a = Tensor([1 0; 1 0], (:i, :j))
b = Tensor([1 1; 0 0], (:j, :i))
a ≈ b
```

This is key for interacting with other tensors.

```@repl tensor
c = a + b
parent(a) + parent(b)
```

As such [`adjoint`](@ref) doesn't permute the dimensions; it just conjugates the array.

```@repl tensor
d = Tensor([1im 2im; 3im 4im], (:i, :j))
d'
conj(d)
```

## Indexing

[`Tensor`](@ref), as a subtype of `AbstractArray`, allows direct indexing of the underneath array with [`getindex`](@ref)/[`setindex!`](@ref) or the `[...]` notation.

```@repl tensor
a[1,1] = 3
a[1,:]
```

But like explained above, on [`Tensor`](@ref) you should refer the dimensions by their index label, which Tenet allows in many methods.

```@repl tensor
a[i=1,j=1]
```

Check out that not specifying all the indices is equivalent to using `:` on the non-specified indices.

```@repl tensor
a[i=1]
a[i=1,j=:]
```

Other supported methods are [`permutedims`](@ref), [`selectdim`](@ref) and [`view`](@ref).

```@repl tensor
permutedims(a, [:j, :i])
selectdim(a, :i, 1)
view(a, :i=>1)
```
