import numpy as np

from nqubitsim import gates
from nqubitsim.simulator import QuantumSimulator


def test_single_qubit_unitary():
    for g in [gates.X, gates.Y, gates.Z, gates.H, gates.S, gates.T]:
        assert gates.is_unitary(g)


def test_two_qubit_unitary():
    for g in [gates.CNOT, gates.CZ, gates.SWAP]:
        assert gates.is_unitary(g)


def test_expand_dimensions():
    op = gates.expand_single_qubit_gate(gates.X, target=0, num_qubits=3)
    assert op.shape == (8, 8)
    op2 = gates.expand_two_qubit_gate(gates.CNOT, control=0, target=2, num_qubits=3)
    assert op2.shape == (8, 8)




# test for single qubit on x-gate. 
def test_x_gate1():
    sim = QuantumSimulator(num_qubits=1)
    
    # Initial state should be |0⟩
    initial_state = sim.state.vector.copy()
    expected_initial = np.array([1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply X gate (Pauli-X flips |0⟩ to |1⟩)
    sim.apply_gate(gates.X, target=0)
    final_state = sim.state.vector
    expected_final = np.array([0.0, 1.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


# test for X gate on 2 qubits
def test_x_gate2():
    sim = QuantumSimulator(num_qubits=2)
    
    # Initial state should be |00⟩
    initial_state = sim.state.vector.copy()
    expected_initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply X gate to qubit 0 (flips |00⟩ to |10⟩)
    sim.apply_gate(gates.X, target=1)
    final_state = sim.state.vector
    expected_final = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)

    print(final_state)


