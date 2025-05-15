using Test
using Muscle
using QuantumTags

@test isplug(Index(plug"1"))
@test !isplug(Index(:i))
