using Test
using Muscle
using QuantumTags
using PythonCall
qiskit = pyimport("qiskit")

@testset "pyconvert to Tensor - 1-qubit gate" begin
    circuit = qiskit.QuantumCircuit(1)
    circuit.x(0)

    gate = only(circuit)
    op = pyconvert(Tensor, gate)

    @test inds(op) == Index.([plug"1", plug"1'"])
    @test parent(op) ≈ [0 1; 1 0]
end

@testset "pyconvert to Tensor - 2-qubit gate" begin
    circuit = qiskit.QuantumCircuit(2)
    circuit.cx(0, 1)

    gate = only(circuit)
    op = pyconvert(Tensor, gate)

    @test inds(op) == Index.([plug"1", plug"2", plug"1'", plug"2'"])
    # @test reshape(parent(op), 4, 4) ≈ [
    #     1 0 0 0
    #     0 0 1 0
    #     0 1 0 0
    #     0 0 0 1
    # ]

    state = Tensor(zeros(2, 2), Index.([plug"1'", plug"2'"]))
    state[plug"1'" => 1, plug"2'" => 1] = 1 / sqrt(2)
    state[plug"1'" => 2, plug"2'" => 2] = 1 / sqrt(2)

    evolved_state = binary_einsum(state, op)
    @test evolved_state[plug"1" => 1, plug"2" => 1] == 1 / sqrt(2)
    @test evolved_state[plug"1" => 1, plug"2" => 2] == 0
    @test evolved_state[plug"1" => 2, plug"2" => 1] == 1 / sqrt(2)
    @test evolved_state[plug"1" => 2, plug"2" => 2] == 0
end
