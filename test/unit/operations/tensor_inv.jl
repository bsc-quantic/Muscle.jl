function tinv(
             A::Tensor; inds_left=(), inds_right=() #, inplace=false, kwargs...
       )
         inds_left, inds_right = factorinds(inds(A), inds_left, inds_right)
         @argcheck isdisjoint(inds_left, inds_right)
         @argcheck issetequal(inds_left ∪ inds_right, inds(A))

         # permute array
         left_sizes = map(Base.Fix1(size, A), inds_left)
         right_sizes = map(Base.Fix1(size, A), inds_right)
         Amat = permutedims(A, [inds_left..., inds_right...])
         Amat = reshape(parent(Amat), prod(left_sizes), prod(right_sizes))

         Ainv = LinearAlgebra.inv(Amat)

         # tensorify results
         Ainv = Tensor(reshape(Ainv, right_sizes..., left_sizes...), [inds_right; inds_left])

       end
