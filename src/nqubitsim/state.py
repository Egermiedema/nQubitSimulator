"""State representation (pure vector or density matrix)."""

from __future__ import annotations
from typing import Iterable, List, Optional

import numpy as np

class QuantumState:
    def __init__(self, num_qubits: int,
                 _state_vector: Optional[np.ndarray] = None,
                 _density_matrix: Optional[np.ndarray] = None):

        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits  # Hilbert space dimension

        if _state_vector is not None and _density_matrix is not None:
            raise ValueError("Provide only one of state_vector or density_matrix.")

        if _state_vector is not None:
            self.set_pure(_state_vector)
        elif _density_matrix is not None:
            self.set_mixed(_density_matrix)
        else:
            zero = np.zeros(self.dim, dtype=complex)
            zero[0] = 1.0
            self.set_pure(zero)  # Default to |0...0>
    
    #Make a copy of the "QuantumState" saving stuff such as num_qubits and representation (pure vs mixed)
    def copy(self) -> "QuantumState":
        return QuantumState(
            num_qubits=self.num_qubits,
            _state_vector=None if self._state_vector is None else self.vector.copy(),
            _density_matrix=None if self._density_matrix is None else self.density.copy(),
        )

    #Return the vector (only pure)
    @property
    def vector(self) -> np.ndarray:
        if self._state_vector is None:
            raise ValueError("State is not in pure form.")
        return self._state_vector

    #return the density matrix (only mixed)
    @property 
    def density(self) -> np.ndarray:
        if self._density_matrix is None:
            raise ValueError("State is not in density form.")
        return self._density_matrix

    #Make state pure and normalize
    def set_pure(self, vec: np.ndarray):
        if vec.shape != (self.dim,):
            raise ValueError(f"State vector must have shape {(self.dim,)}.")
        norm = np.linalg.norm(vec) # numpy function to normalize vectors

        if norm < 1e-15:
            raise ValueError("State vector cannot be (near) zero.") 
        # floating point numbers almost never give exactl zero. 
        # passing vector of extremely small numbers is meaningless due to floating point precision limits.
        
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
        return vec @ vec.conj().T # @ is matrix multiplication operator in numpy. you learn new stuff every day!


    def promote_to_density(self): # promote pure state to density matrix
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
        probs = np.zeros(2 ** len(qubits)) # Initialize probability array (with zeros)
        for outcome in range(2 ** len(qubits)):
            bitstring = [(outcome >> (len(qubits) - 1 - i)) & 1 for i in range(len(qubits))]
            projector = self._projector(qubits, bitstring) # Build projector for this outcome
            probs[outcome] = np.real(np.trace(projector @ rho)) # Compute probability p = Tr(Pρ)
        return probs

    def _projector(self, qubits: List[int], bits: List[int]):
        """Build projector onto given bitstring for specified qubits."""
        op = None
        for q in range(self.num_qubits):
            if q in qubits:
                idx = qubits.index(q)
                proj = np.array([[1, 0], [0, 0]], dtype=complex) if bits[idx] == 0 else np.array(
                    [[0, 0], [0, 1]], dtype=complex)
            else:
                proj = np.eye(2, dtype=complex)
            op = proj if op is None else np.kron(op, proj)
        return op
