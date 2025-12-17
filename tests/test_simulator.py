import numpy as np

from nqubitsim import gates
from nqubitsim.simulator import QuantumSimulator


def test_superposition_measurement():
    sim = QuantumSimulator(num_qubits=1, rng=np.random.default_rng(0))
    sim.apply_gate(gates.H, target=0)
    probs = sim.state.probabilities([0])
    assert np.allclose(probs, [0.5, 0.5], atol=1e-2)


def test_cnot_entangles():
    sim = QuantumSimulator(num_qubits=2)
    sim.apply_gate(gates.H, target=0)
    sim.apply_controlled_gate(gates.CNOT, control=0, target=1)
    rho = sim.state.as_density_matrix()
    # Expect (|00>+|11>)/sqrt(2)
    expected = np.zeros((4, 4), dtype=complex)
    bell = (1 / np.sqrt(2)) * np.array([1, 0, 0, 1], dtype=complex)
    expected = np.outer(bell, bell.conj())
    assert np.allclose(rho, expected, atol=1e-8)


def test_noise_bit_flip():
    sim = QuantumSimulator(num_qubits=1, noise={"bit_flip": 1.0}, rng=np.random.default_rng(1))
    sim.apply_gate(gates.I, target=0)
    # With p=1, state flips to |1>
    probs = sim.state.probabilities([0])
    assert probs[1] > 0.99


# test that measuring a qubit collapses the state to the correct post-measurement density matrix
def test_measurement_collapse_pure_and_mixed():
    sim = QuantumSimulator(num_qubits=1, rng=np.random.default_rng(0))
    sim.apply_gate(gates.H, target=0)

    # Measure -> should collapse to |0> or |1>
    outcome, post = sim.measure([0])
    rho = post.as_density_matrix()

    if outcome == 0:
        expected = np.array([[1, 0],
                             [0, 0]], dtype=complex)
    else:
        expected = np.array([[0, 0],
                             [0, 1]], dtype=complex)
    assert np.allclose(rho, expected), "Post-measurement state mismatch"


##------------------------------------Full scenario tests---------------------------------------

def test_bell_state_full_scenario():
    sim = QuantumSimulator(num_qubits=2, rng=np.random.default_rng(42))

    # |00>
    sim.apply_gate(gates.H, target=0)
    sim.apply_controlled_gate(gates.CNOT, control=0, target=1)

    # Expect Bell state (|00> + |11>) / sqrt(2)
    rho = sim.state.as_density_matrix()
    bell = (1 / np.sqrt(2)) * np.array([1, 0, 0, 1], dtype=complex)
    expected = np.outer(bell, bell.conj())

    assert np.allclose(rho, expected, atol=1e-8)

    # Measure both qubits
    outcome, _ = sim.measure([0, 1])

    # Only 00 or 11 are allowed
    assert outcome in [0, 3]


def test_three_qubit_ghz_scenario():
    sim = QuantumSimulator(num_qubits=3)

    sim.apply_gate(gates.H, target=0)
    sim.apply_controlled_gate(gates.CNOT, control=0, target=1)
    sim.apply_controlled_gate(gates.CNOT, control=1, target=2)

    rho = sim.state.as_density_matrix()

    ghz = (1 / np.sqrt(2)) * np.array(
        [1, 0, 0, 0, 0, 0, 0, 1], dtype=complex
    )
    expected = np.outer(ghz, ghz.conj())

    assert np.allclose(rho, expected, atol=1e-8)


def test_hzh_interference_scenario():
    sim = QuantumSimulator(num_qubits=1)

    sim.apply_gate(gates.H, target=0)
    sim.apply_gate(gates.Z, target=0)
    sim.apply_gate(gates.H, target=0)

    # HZH |0> = |1>
    final_state = sim.state.get_vector()
    expected = np.array([0.0, 1.0], dtype=complex)

    np.testing.assert_array_almost_equal(final_state, expected)


def test_bit_flip_invariance_of_plus_state():
    sim = QuantumSimulator(
        num_qubits=1,
        noise={"bit_flip": 1.0},
        rng=np.random.default_rng(0)
    )

    sim.apply_gate(gates.H, target=0)

    probs = sim.state.probabilities([0])

    # |+> is invariant under X
    assert np.allclose(probs, [0.5, 0.5], atol=1e-8)


def test_simulator_reset_and_reuse():
    sim = QuantumSimulator(num_qubits=2)

    sim.apply_gate(gates.X, target=0)
    sim.apply_controlled_gate(gates.CNOT, control=0, target=1)

    # Reset simulator
    sim.reset()

    # Should be back to |00>
    expected = np.array([1, 0, 0, 0], dtype=complex)
    np.testing.assert_array_almost_equal(sim.state.get_vector(), expected)
    assert sim.classical_register == []