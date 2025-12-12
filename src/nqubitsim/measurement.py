"""Measurement utilities: projective and POVM."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .state import QuantumState


def _expand_operator_on_qubits(op: np.ndarray, qubits: List[int], num_qubits: int) -> np.ndarray:
    """Embed an operator acting on |qubits| into the full Hilbert space."""
    k = len(qubits)
    dim_full = 2 ** num_qubits
    dim_sub = 2 ** k
    if op.shape != (dim_sub, dim_sub):
        raise ValueError(f"Operator shape must be {(dim_sub, dim_sub)}.")

    qubits = list(qubits)
    op_tensor = op.reshape([2] * (2 * k))
    full = np.zeros((dim_full, dim_full), dtype=complex)

    def bits(val: int):
        return [(val >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]

    for row in range(dim_full):
        row_bits = bits(row)
        for col in range(dim_full):
            col_bits = bits(col)
            if any(row_bits[q] != col_bits[q] for q in range(num_qubits) if q not in qubits):
                continue
            r_idx = 0
            c_idx = 0
            for b in qubits:
                r_idx = (r_idx << 1) | row_bits[b]
                c_idx = (c_idx << 1) | col_bits[b]
            full[row, col] = op_tensor.ravel()[r_idx * dim_sub + c_idx]
    return full


def projective_measure(state: QuantumState, qubits: Iterable[int], basis: Sequence[np.ndarray] | None = None, rng=None):
    """Perform projective measurement on selected qubits.

    Args:
        state: QuantumState to measure (updated in-place).
        qubits: iterable of qubit indices.
        basis: optional list of orthonormal vectors spanning measured subspace.
               If None, computational basis is used.
        rng: numpy Generator override.
    Returns:
        (outcome_index, post_state)
    """
    rng = rng or np.random.default_rng()
    qubits = list(qubits)
    k = len(qubits)
    rho = state.as_density_matrix()

    if basis is None:
        basis = []
        for outcome in range(2**k):
            bits = [(outcome >> (k - 1 - i)) & 1 for i in range(k)]
            vec = np.array([1.0], dtype=complex)
            for bit in bits:
                vec = np.kron(vec, np.array([1, 0], dtype=complex) if bit == 0 else np.array([0, 1], dtype=complex))
            basis.append(vec)

    # Build projectors
    projectors = []
    for vec in basis:
        vec = np.asarray(vec, dtype=complex).reshape(-1)
        if vec.shape != (2**k,):
            raise ValueError("Basis vector has wrong dimension.")
        proj = np.outer(vec, vec.conj())
        projectors.append(_expand_operator_on_qubits(proj, qubits, state.num_qubits))

    probs = np.array([np.real(np.trace(P @ rho)) for P in projectors])
    probs = np.clip(probs, 0, 1)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)

    outcome = rng.choice(len(projectors), p=probs)
    M = projectors[outcome]
    new_rho = M @ rho @ M
    p_out = probs[outcome]
    new_rho = new_rho / p_out if p_out > 0 else new_rho

    new_state = QuantumState(state.num_qubits, sparse_threshold=state.sparse_threshold, _density_matrix=new_rho)
    # update original state
    state.set_mixed(new_rho)
    return outcome, new_state


def povm_measure(state: QuantumState, operators: Sequence[np.ndarray], qubits: Iterable[int] | None = None, rng=None):
    """Perform POVM measurement.

    Args:
        state: QuantumState to measure (updated).
        operators: Kraus operators for each outcome. If `qubits` is None, they
                   are assumed to be full-system operators. Otherwise they act
                   on the specified subset and are embedded.
        qubits: optional subset of qubits the operators act on.
        rng: numpy Generator override.
    Returns:
        (outcome_index, post_state)
    """
    rng = rng or np.random.default_rng()
    qubits = list(qubits) if qubits is not None else None
    rho = state.as_density_matrix()

    embedded_ops: List[np.ndarray] = []
    for op in operators:
        if qubits is None:
            embedded_ops.append(np.asarray(op, dtype=complex))
        else:
            embedded_ops.append(_expand_operator_on_qubits(np.asarray(op, dtype=complex), qubits, state.num_qubits))

    probs = np.array([np.real(np.trace(k @ rho @ k.conj().T)) for k in embedded_ops])
    probs = np.clip(probs, 0, 1)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)

    outcome = rng.choice(len(embedded_ops), p=probs)
    K = embedded_ops[outcome]
    new_rho = K @ rho @ K.conj().T
    p_out = probs[outcome]
    new_rho = new_rho / p_out if p_out > 0 else new_rho

    new_state = QuantumState(state.num_qubits, sparse_threshold=state.sparse_threshold, _density_matrix=new_rho)
    state.set_mixed(new_rho)
    return outcome, new_state

