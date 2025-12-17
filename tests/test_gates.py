import numpy as np

from nqubitsim import gates
from nqubitsim.simulator import QuantumSimulator


def test_expand_dimensions():
    op = gates.expand_single_qubit_gate(gates.X, target=0, num_qubits=3)
    assert op.shape == (8, 8)
    op2 = gates.expand_two_qubit_gate(gates.CNOT, control=0, target=2, num_qubits=3)
    assert op2.shape == (8, 8)


#-------------------------------SINGE QUBIT GATE TESTS------------------------------------
#------------------------------------UNITARY TEST-----------------------------------------
def test_single_qubit_unitary():
    for g in [gates.X, gates.Y, gates.Z, gates.H, gates.S, gates.T]:
        assert gates.is_unitary(g)


#------------------------------------X-GATE TESTS---------------------------------------
# test for single qubit on x-gate. 
def test_x_gate1():
    sim = QuantumSimulator(num_qubits=1)
    
    # Initial state should be |0⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply X gate (Pauli-X flips |0⟩ to |1⟩)
    sim.apply_gate(gates.X, target=0)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 1.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


# test for X gate on 2 qubits
def test_x_gate2():
    sim = QuantumSimulator(num_qubits=2)
    
    # Initial state should be |00⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply X gate to qubit 0 (flips |00⟩ to |01⟩)
    sim.apply_gate(gates.X, target=1)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


# test for X gate on 2 qubits
def test_x_gate3():
    sim = QuantumSimulator(num_qubits=2)
    
    # Initial state should be |00⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply X gate to qubit 0 (flips |00⟩ to |10⟩)
    sim.apply_gate(gates.X, target=0)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)




#------------------------------------H-GATE TESTS---------------------------------------
# test for single qubit on H-gate.
def test_H_gate1():
    sim = QuantumSimulator(num_qubits=1)
    
    # Initial state should be |0⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply H gate (Hadamard transforms |0⟩ to (|1⟩ + |1⟩)/√2)
    sim.apply_gate(gates.H, target=0)
    final_state = sim.state.get_vector()
    expected_final = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    np.testing.assert_array_almost_equal(final_state, expected_final)

# test for H gate on 2 qubits
def test_H_gate2():
    sim = QuantumSimulator(num_qubits=2)
    
    # Initial state should be |00⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply H gate to qubit 0 
    sim.apply_gate(gates.H, target=0)
    final_state = sim.state.get_vector()
    expected_final = np.array([1.0, 0.0, 1.0, 0.0], dtype=complex) / np.sqrt(2)
    np.testing.assert_array_almost_equal(final_state, expected_final)


def test_H_gate3():
    sim = QuantumSimulator(num_qubits=2)
    
    # Initial state should be |00⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply H gate to qubit 1
    sim.apply_gate(gates.H, target=1)
    final_state = sim.state.get_vector()
    expected_final = np.array([1.0, 1.0, 0.0, 0.0], dtype=complex) / np.sqrt(2)
    np.testing.assert_array_almost_equal(final_state, expected_final)


#-------------------------------------Y-GATE TESTS---------------------------------------
# test for single qubit on Y-gate.
def test_Y_gate1():
    sim = QuantumSimulator(num_qubits=1)
    
    # Initial state should be |0⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply Y gate (Pauli-Y flips |0⟩ to i|1⟩)
    sim.apply_gate(gates.Y, target=0)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 1.0j], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


# test for Y gate on 2 qubits
def test_Y_gate2():
    sim = QuantumSimulator(num_qubits=2)
    
    # Initial state should be |00⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply Y gate to qubit 0 (flips |00⟩ to i|10⟩)
    sim.apply_gate(gates.Y, target=0)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 0.0, 1.0j, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)   


# test for Y gate on 2 qubits
def test_Y_gate3():
    sim = QuantumSimulator(num_qubits=2)
    
    # Initial state should be |00⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply Y gate to qubit 1 (flips |00⟩ to i|01⟩)
    sim.apply_gate(gates.Y, target=1)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 1.0j, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


#-------------------------------------Z-GATE TESTS---------------------------------------
# test for single qubit on Z-gate.
def test_Z_gate1():
    sim = QuantumSimulator(num_qubits=1)
    
    # Initial state should be |0⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply Z gate (Pauli-Z leaves |0⟩ unchanged)
    sim.apply_gate(gates.Z, target=0)
    final_state = sim.state.get_vector()
    expected_final = np.array([1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)   


def test_Z_gate4():
    sim = QuantumSimulator(num_qubits=1)
    sim.apply_gate(gates.X, target=0) # Start in |1⟩ by applying X first

    # Initial state should be |1⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([0.0, 1.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply Z gate (Pauli-Z flips |1⟩ to -|1⟩)
    sim.apply_gate(gates.Z, target=0)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, -1.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)   


# test for Z gate on 2 qubits
def test_Z_gate2():
    sim = QuantumSimulator(num_qubits=2)
    
    # Initial state should be |00⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply Z gate to qubit 0 (leaves |00⟩ unchanged)
    sim.apply_gate(gates.Z, target=0)
    final_state = sim.state.get_vector()
    expected_final = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


# test for Z gate on 2 qubits
def test_Z_gate3():
    sim = QuantumSimulator(num_qubits=2)
    
    # Initial state should be |00⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply Z gate to qubit 1 (leaves |00⟩ unchanged)
    sim.apply_gate(gates.Z, target=1)
    final_state = sim.state.get_vector()
    expected_final = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


#-------------------------------------S-GATE TESTS---------------------------------------
# test for single qubit on S-gate.
def test_S_gate1():
    sim = QuantumSimulator(num_qubits=1)
    
    # Initial state should be |0⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply S gate (Phase gate leaves |0⟩ unchanged)
    sim.apply_gate(gates.S)
    final_state = sim.state.get_vector()
    expected_final = np.array([1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


def test_S_gate2():
    sim = QuantumSimulator(num_qubits=1)
    sim.apply_gate(gates.X) # Start in |1⟩ by applying X first

    # Initial state should be |1⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([0.0, 1.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply S gate (Phase gate transforms |1⟩ to i|1⟩)
    sim.apply_gate(gates.S)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 0.0 + 1.0j], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)

def test_S_gate3():
    sim = QuantumSimulator(num_qubits=2)
    sim.apply_gate(gates.X, target=0) # Start in |10⟩ by applying X first
    sim.apply_gate(gates.X, target=1) # Now in |11⟩
    
    # Initial state should be |11⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply S gate to qubit 1 (leaves |11⟩ unchanged)
    sim.apply_gate(gates.S)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 0.0, 0.0, 0.0 + 1.0j], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


#-------------------------------------T-GATE TESTS---------------------------------------
# test for single qubit on T-gate.
def test_T_gate1():
    sim = QuantumSimulator(num_qubits=1)
    
    # Initial state should be |0⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply T gate (π/8 gate leaves |0⟩ unchanged)
    sim.apply_gate(gates.T)
    final_state = sim.state.get_vector()
    expected_final = np.array([1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


def test_T_gate2():
    sim = QuantumSimulator(num_qubits=1)
    sim.apply_gate(gates.X) # Start in |1⟩ by applying X first

    # Initial state should be |1⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([0.0, 1.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply T gate (π/8 gate transforms |1⟩ to exp(iπ/4)|1⟩)
    sim.apply_gate(gates.T)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, np.exp(1j * np.pi / 4)], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


def test_phase_gate_on_zero():
    sim = QuantumSimulator(num_qubits=1)

    theta = np.pi / 2
    sim.apply_gate(gates.phase(theta), target=0)

    # |0> should be unchanged
    expected = np.array([1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(sim.state.get_vector(), expected)
    

def test_phase_gate_on_one():
    sim = QuantumSimulator(num_qubits=1)

    # Start from |1>
    sim.apply_gate(gates.X, target=0)

    # Apply phase gate with theta = 2π/3
    theta = 2 * np.pi / 3
    sim.apply_gate(gates.phase(theta), target=0)

    final_state = sim.state.get_vector()

    # |1> should become e^{iθ} |1>
    expected = np.array([0.0, np.exp(1j * theta)], dtype=complex)

    np.testing.assert_array_almost_equal(final_state, expected)


#------------------------------------two-qubit GATE TESTS---------------------------------------
#------------------------------------UNITARY TEST-----------------------------------------
def test_two_qubit_unitary():
    for g in [gates.CNOT, gates.CZ, gates.SWAP]:
        assert gates.is_unitary(g)

#------------------------------------CNOT-GATE TESTS---------------------------------------
def test_CNOT_gate1():
    sim = QuantumSimulator(num_qubits=2)
    
    # Initial state should be |00⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply CNOT with qubit 0 as control and qubit 1 as target (should leave |00⟩ unchanged)
    sim.apply_controlled_gate(gates.CNOT, control=0, target=1)
    final_state = sim.state.get_vector()
    expected_final = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


def test_CNOT_gate2():
    sim = QuantumSimulator(num_qubits=2)
    sim.apply_gate(gates.X, target=0) # Start in |10⟩ by applying X
    initial_state = sim.state.get_vector().copy() # Initial state should be |10⟩
    expected_initial = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)
    
    # Apply CNOT with qubit 0 as control and qubit 1 as target. should result in |11⟩
    sim.apply_controlled_gate(gates.CNOT, control=0, target=1)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


def test_CNOT_gate3():
    sim = QuantumSimulator(num_qubits=3)
    sim.apply_gate(gates.X, target=0) # Start in |100⟩ by applying X
    initial_state = sim.state.get_vector().copy() # Initial state should be |100⟩
    expected_initial = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)

    # Apply CNOT with qubit 0 as control and qubit 2 as target. should result in |101⟩
    sim.apply_controlled_gate(gates.CNOT, control=0, target=2)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


#------------------------------------CZ-GATE TESTS---------------------------------------
def test_CZ_gate1():
    sim = QuantumSimulator(num_qubits=2)

    # Initial state should be |00⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)

    # Apply CZ with qubit 0 as control and qubit 1 as target (should leave |00⟩ unchanged)
    sim.apply_controlled_gate(gates.CZ, control=0, target=1)
    final_state = sim.state.get_vector()
    expected_final = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


def test_CZ_gate2():
    sim = QuantumSimulator(num_qubits=2)
    sim.apply_gate(gates.X, target=1) # Start in |01⟩ by applying X first

    # Initial state should be |01⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)

    # Apply CZ gate (Controlled-Z flips |11⟩ to -|11⟩)
    sim.apply_controlled_gate(gates.CZ, control=0, target=1)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


def test_CZ_gate2():
    sim = QuantumSimulator(num_qubits=2)
    sim.apply_gate(gates.X, target=0) # Start in |10⟩ by applying X first
    sim.apply_gate(gates.X, target=1) # Start in |11⟩ by applying X first

    # Initial state should be |11⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)

    # Apply CZ gate (Controlled-Z flips |11⟩ to -|11⟩)
    sim.apply_controlled_gate(gates.CZ, control=0, target=1)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 0.0, 0.0, -1.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


#------------------------------------SWAP-GATE TESTS---------------------------------------
def test_SWAP_gate1():
    sim = QuantumSimulator(num_qubits=2)

    # Initial state should be |00⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)

    # Apply SWAP gate (should leave |00⟩ unchanged)
    sim.apply_controlled_gate(gates.SWAP, control=0, target=1)
    final_state = sim.state.get_vector()
    expected_final = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)


def test_SWAP_gate2():
    sim = QuantumSimulator(num_qubits=2)
    sim.apply_gate(gates.X, target=0) # Start in |10⟩ by applying X first

    # Initial state should be |10⟩
    initial_state = sim.state.get_vector().copy()
    expected_initial = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(initial_state, expected_initial)

    # Apply SWAP gate (should result in |01⟩)
    sim.apply_controlled_gate(gates.SWAP, control=0, target=1)
    final_state = sim.state.get_vector()
    expected_final = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_almost_equal(final_state, expected_final)