module MuscleFiniteDifferencesExt

using Muscle
using FiniteDifferences

tensor_0dim_from_vec(x) = Tensor(fill(only(x)))

function FiniteDifferences.to_vec(x::Tensor{T,0}) where {T}
    return T[only(x)], tensor_0dim_from_vec
end

end
