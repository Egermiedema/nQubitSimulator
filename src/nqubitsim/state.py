"""State representation (pure vector or density matrix) with optional sparsity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from scipy import sparse


def _as_sparse_if_needed(arr: np.ndarray, use_sparse: bool):
    if not use_sparse:
        return arr
    if arr.ndim == 1:
        return sparse.csr_matrix(arr.reshape(-1, 1))
    return sparse.csr_matrix(arr)


@dataclass
class QuantumState:
    num_qubits: int
    sparse_threshold: int = 2**10  # switch to sparse when dimension >= threshold
    _state_vector: Optional[np.ndarray] = None
    _density_matrix: Optional[np.ndarray] = None

    #initialize state from vector V density matrix V default(|0...0>)
    def __post_init__(self):
        self.dim = 2 ** self.num_qubits
        self.use_sparse = self.dim >= self.sparse_threshold

        if self._state_vector is not None and self._density_matrix is not None:
            raise ValueError("Provide only one of state_vector or density_matrix.")

        if self._state_vector is not None:
            self.set_pure(self._state_vector)
        elif self._density_matrix is not None:
            self.set_mixed(self._density_matrix)
        else:
            zero = np.zeros(self.dim, dtype=complex)
            zero[0] = 1.0
            self.set_pure(zero)

    @property #TODO: verwijderen
    def is_density(self) -> bool:
        return self._density_matrix is not None
    
    #Make a copy of the "QuantumState"
    def copy(self) -> "QuantumState":
        return QuantumState(
            num_qubits=self.num_qubits,
            sparse_threshold=self.sparse_threshold,
            _state_vector=None if self._state_vector is None else self.vector.copy(),
            _density_matrix=None if self._density_matrix is None else self.density.copy(),
        )

    #Return the vector (only pure)
    @property #TODO verwijderen
    def vector(self) -> np.ndarray:
        if self._state_vector is None:
            raise ValueError("State is not in pure form.")
        return self._state_vector

    #return the density matrix (only mixed)
    @property #TODO: verwijderen
    def density(self):
        if self._density_matrix is None:
            raise ValueError("State is not in density form.")
        return self._density_matrix

    #Make state pure and normalize
    def set_pure(self, vec: np.ndarray):
        if vec.shape != (self.dim,):
            raise ValueError(f"State vector must have shape {(self.dim,)}.")
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("State vector cannot be zero.")
        vec = vec / norm
        self._state_vector = np.array(vec, dtype=complex)
        self._density_matrix = None

    #set state as density matrix and validate Hermiticity and renormalize
    def set_mixed(self, rho: np.ndarray):
        if rho.shape != (self.dim, self.dim):
            raise ValueError(f"Density matrix must have shape {(self.dim, self.dim)}.")
        if not np.allclose(rho, rho.conj().T):
            raise ValueError("Density matrix must be Hermitian.")
        trace = np.trace(rho)
        if not np.isclose(trace, 1.0):
            rho = rho / trace  # renormalize
        self._density_matrix = np.array(rho, dtype=complex)
        self._state_vector = None

    def as_density_matrix(self):
        if self._density_matrix is not None:
            return self._density_matrix
        # |psi><psi|
        vec = self.vector.reshape(-1, 1)
        return vec @ vec.conj().T

    def promote_to_density(self):
        """Convert to density matrix representation in-place."""
        if self._density_matrix is None:
            self._density_matrix = self.as_density_matrix()
            self._state_vector = None
        return self

    # apply a unitary operator
    def apply_unitary(self, operator):
        if self._state_vector is not None:
            # U|ψ⟩ pure state
            self._state_vector = operator @ self.vector
        else:
            #UρU† for mixed states
            rho = self.density
            self._density_matrix = operator @ rho @ operator.conj().T
        return self


    #Return probability distribution over measurement outcomes of selected qubits
    def probabilities(self, qubits: Iterable[int]) -> np.ndarray:
        qubits = list(qubits)
        rho = self.as_density_matrix()
        probs = np.zeros(2 ** len(qubits))
        for outcome in range(2 ** len(qubits)):
            bitstring = [(outcome >> (len(qubits) - 1 - i)) & 1 for i in range(len(qubits))]
            projector = self._projector(qubits, bitstring)
            probs[outcome] = np.real(np.trace(projector @ rho))
        return probs

    def _projector(self, qubits: List[int], bits: List[int]):
        """Build projector onto given bitstring for specified qubits."""
        op = None
        for q in range(self.num_qubits):
            if q in qubits:
                idx = qubits.index(q)
                proj = np.array([[1, 0], [0, 0]], dtype=complex) if bits[idx] == 0 else np.array(
                    [[0, 0], [0, 1]], dtype=complex
                )
            else:
                proj = np.eye(2, dtype=complex)
            op = proj if op is None else np.kron(op, proj)
        return op

    def maybe_sparse(self, arr: np.ndarray):
        """Convert array to sparse if threshold exceeded."""
        return _as_sparse_if_needed(arr, self.use_sparse)