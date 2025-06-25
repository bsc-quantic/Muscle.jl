# Singular Value Decomposition

```@setup example
using Muscle
```

```@repl example
t = Tensor(rand(2,2,2,2), [:i,:j,:k,:l]);
u, s, v = tensor_svd_thin(
    t;
    inds_u = Index.([:i, :k]),
    inds_v = Index.([:j, :l]),
    ind_s = Index(:x)
);
u
s
v
```
