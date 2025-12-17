"""Noise channels (bit-flip, depolarizing) and helpers."""

from __future__ import annotations
from typing import Iterable, Sequence
import numpy as np
from . import gates
from .state import QuantumState


def apply_bit_flip(state: QuantumState, p: float, rng=None):
    """Apply bit-flip (Pauli-X) to each qubit with probability p."""
    if p <= 0:
        return state
    
    rng = rng or np.random.default_rng()
    for q in range(state.num_qubits):
        
        if rng.random() < p:
            op = gates.expand_single_qubit_gate(gates.X, q, state.num_qubits)
            state.apply_unitary(op)
    return state


#Apply a depolarizing noise channel:
#E(rho) = (1 - p) * rho + (p / d) * I
#where d = 2^n.
def apply_depolarizing(state: QuantumState, p: float):
    if p <= 0:
        return state

    state.promote_to_density()
    d = 2 ** state.num_qubits

    rho = state.get_density()
    I = np.eye(d, dtype=complex)

    new_rho = (1 - p) * rho + (p / d) * I
    state.set_mixed(new_rho)
    return state