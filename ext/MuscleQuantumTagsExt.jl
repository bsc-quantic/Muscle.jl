module MuscleQuantumTagsExt

using Muscle: Index
using QuantumTags

QuantumTags.isplug(ind::Index) = isplug(ind.tag)

end
