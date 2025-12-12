"""QuantumSimulator orchestrates state, gates, measurement, and noise."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from . import gates
from .measurement import povm_measure, projective_measure
from .noise import apply_bit_flip, apply_depolarizing
from .state import QuantumState


class QuantumSimulator:
    def __init__(self, num_qubits: int, sparse_threshold: int = 2**10, noise: dict | None = None, rng=None):
        self.num_qubits = num_qubits
        self.rng = rng or np.random.default_rng()
        self.noise_cfg = noise or {}
        self.state = QuantumState(num_qubits=num_qubits, sparse_threshold=sparse_threshold)
        self.classical_register: list[int] = []

    def _apply_noise(self):
        if not self.noise_cfg:
            return
        if "bit_flip" in self.noise_cfg:
            apply_bit_flip(self.state, self.noise_cfg["bit_flip"], rng=self.rng)
        if "depolarizing" in self.noise_cfg:
            apply_depolarizing(self.state, self.noise_cfg["depolarizing"])

    def apply_gate(self, gate: np.ndarray, target: int):
        """Apply a single-qubit gate to target qubit."""
        op = gates.expand_single_qubit_gate(gate, target, self.num_qubits, use_sparse=self.state.use_sparse)
        self.state.apply_unitary(op)
        self._apply_noise()
        return self

    def apply_controlled_gate(self, gate: np.ndarray, control: int, target: int):
        """Apply a two-qubit gate with given control and target."""
        op = gates.expand_two_qubit_gate(gate, control, target, self.num_qubits, use_sparse=self.state.use_sparse)
        self.state.apply_unitary(op)
        self._apply_noise()
        return self

    def measure(self, qubits: Iterable[int], basis: Sequence[np.ndarray] | None = None):
        """Projective measurement; updates state and records classical outcome."""
        outcome, post_state = projective_measure(self.state, qubits, basis=basis, rng=self.rng)
        self.classical_register.append(int(outcome))
        return outcome, post_state

    def povm(self, operators: Sequence[np.ndarray], qubits: Iterable[int] | None = None):
        """POVM measurement."""
        outcome, post_state = povm_measure(self.state, operators, qubits=qubits, rng=self.rng)
        self.classical_register.append(int(outcome))
        return outcome, post_state

    def reset(self):
        """Reset to |0...0> and clear classical register."""
        self.state = QuantumState(num_qubits=self.num_qubits, sparse_threshold=self.state.sparse_threshold)
        self.classical_register = []
        return self

