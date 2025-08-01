module MusclePythonCallExt

using Muscle
using PythonCall
using PythonCall.Convert: pyconvert_add_rule, pyconvert_return, pyconvert_unconverted

function pyconvert_rule_qiskit_instruction(T::Type{<:Tensor}, pyobj)
    # NOTE we discard any quantum register information: we only keep qubit index
    # NOTE add 1 to convert from Python's 0-based indexing to Julia's 1-based indexing
    lanes = map(x -> pyconvert(Int, x._index) + 1, instr.qubits)
    _plugs = [[plug"$s" for s in lanes]; [plug"$s'" for s in lanes]]

    matrix = if pyhasattr(instr, Py("matrix"))
        instr.matrix
    else
        instr.operation.to_matrix()
    end

    # if unassigned parameters, throw
    if pyisnone(matrix)
        throw(ArgumentError("Expected parameters already assigned, but got $(pyobj.params)"))
    end

    matrix = pyconvert(Array, matrix)
    array = reshape(matrix, fill(2, length(_plugs))...)

    pyconvert_return(T(array, Index.(_plugs)))
end

function __init__()
    pyconvert_add_rule("qiskit._accelerate.circuit.CircuitInstruction", Tensor, pyconvert_rule_qiskit_instruction)
end

end
