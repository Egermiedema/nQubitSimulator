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

