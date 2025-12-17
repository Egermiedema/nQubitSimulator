"""QuantumSimulator orchestrates state, gates, measurement, and noise."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from . import gates
from .measurement import povm_measure, projective_measure
from .noise import apply_bit_flip, apply_depolarizing
from .state import QuantumState


class QuantumSimulator:
    """this class calls all the other functions to simulate a quantum circuit or states.
    It holds the quantum state, applies gates, measurements, noise, reset and POVM measurements.
    """

    # Initialize simulator with number of qubits, optional noise config, and Randon Number Generator.
    def __init__(self, num_qubits: int, noise: dict | None = None, rng=None):
        self.num_qubits = num_qubits
        self.rng = rng or np.random.default_rng()
        self.noise_cfg = noise or {}
        self.state = QuantumState(num_qubits=num_qubits)
        self.classical_register: list[int] = []

    # calls function to apply noise after gates
    def _apply_noise(self):
        if not self.noise_cfg:
            return
        if "bit_flip" in self.noise_cfg:
            apply_bit_flip(self.state, self.noise_cfg["bit_flip"], rng=self.rng)
        if "depolarizing" in self.noise_cfg:
            apply_depolarizing(self.state, self.noise_cfg["depolarizing"])

    # Apply a single-qubit gate to target qubit
    def apply_gate(self, gate: np.ndarray, target=0):
        """Apply a single-qubit gate to target qubit."""
        op = gates.expand_single_qubit_gate(gate, target, self.num_qubits)
        """Apply a single-qubit gate to target qubit (defaults to qubit 0)."""
        self.state.apply_unitary(op)
        self._apply_noise()
        return self

    # Apply a two-qubit gate with given control and target
    def apply_controlled_gate(self, gate: np.ndarray, control: int, target: int):
        """Apply a two-qubit gate with given control and target."""
        op = gates.expand_two_qubit_gate(gate, control, target, self.num_qubits)
        self.state.apply_unitary(op)
        self._apply_noise()
        return self

    # Projective measurement; updates state and records classical outcome.
    def measure(self, qubits: Iterable[int], basis: Sequence[np.ndarray] | None = None):
        """Projective measurement; updates state and records classical outcome."""
        outcome, post_state = projective_measure(self.state, qubits, basis=basis, rng=self.rng)
        self.classical_register.append(int(outcome))
        return outcome, post_state

    # POVM measurement.
    def povm(self, operators: Sequence[np.ndarray], qubits: Iterable[int] | None = None):
        """POVM measurement."""
        outcome, post_state = povm_measure(self.state, operators, qubits=qubits, rng=self.rng)
        self.classical_register.append(int(outcome))
        return outcome, post_state

    # Reset simulator to initial state
    def reset(self):
        """Reset to |0...0> and clear classical register."""
        self.state = QuantumState(num_qubits=self.num_qubits)
        self.classical_register = []
        return self

