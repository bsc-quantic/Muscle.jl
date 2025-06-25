# QR decompostion

```@setup example
using Muscle
```

```@repl example
t = Tensor(rand(2,2,2,2), [:i,:j,:k,:l]);
q, r = tensor_qr_thin(
    t;
    inds_q = Index.([:i, :k]),
    inds_r = Index.([:j, :l]),
    ind_virtual = Index(:x)
);
q
r
```
