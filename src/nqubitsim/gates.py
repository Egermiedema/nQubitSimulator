"""Common quantum gates and expansion utilities."""

from __future__ import annotations

import numpy as np
from scipy import sparse

# Base single-qubit gates (2x2)
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def phase(theta: float) -> np.ndarray:
    """Return general phase gate P(theta)."""
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)

# Two-qubit gates (4x4)
CNOT = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=complex,
)

CZ = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ],
    dtype=complex,
)

SWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=complex,
)


def expand_single_qubit_gate(gate: np.ndarray, target: int, num_qubits: int, use_sparse: bool = False):
    """Kronecker-expand a 2x2 gate.
     function takes a single-qubit quantum gate (represented as a 2×2 NumPy array) 
     and expands it into a full operator that acts on an n-qubit quantum system."""
    
    # Checks that gate is exactly 2×2.
    if gate.shape != (2, 2):
        raise ValueError("Single-qubit gate must be 2x2.")
    if target < 0 or target >= num_qubits:
        raise ValueError("Target out of range.")

    # Creates a list factors where each element is either the input gate 
    # (for the target qubit) or the identity matrix I
    factors = []
    for qubit in range(num_qubits):
        factors.append(gate if qubit == target else I)


    # Iteratively applies the Kronecker product (np.kron or sparse.kron) with each subsequent factor.
    # Result: A full operator like I ⊗ I ⊗ ... ⊗ gate ⊗ ... ⊗ I, 
    # where the gate is positioned at the target qubit.
    # example for 3 qubits and target 1: I ⊗ gate ⊗ I. 
    # produces size 8x8 matrix. (2^3 x 2^3)
    op = factors[0]
    for factor in factors[1:]:
        op = np.kron(op, factor) if not use_sparse else sparse.kron(op, factor, format="csr")
    return op


def expand_two_qubit_gate(gate: np.ndarray, control: int, target: int, num_qubits: int, use_sparse: bool = False):
    """Expand a 4x4 two-qubit gate to an n-qubit operator.

    This implementation is correctness-first and keeps ordering consistent
    with expand_single_qubit_gate (qubit 0 is the leftmost factor).
    """
    if gate.shape != (4, 4):
        raise ValueError("Two-qubit gate must be 4x4.")
    if control == target:
        raise ValueError("Control and target must differ.")
    if min(control, target) < 0 or max(control, target) >= num_qubits:
        raise ValueError("Qubit index out of range.")

    targets = (control, target)
    dim = 2 ** num_qubits
    op = sparse.dok_matrix((dim, dim), dtype=complex) if use_sparse else np.zeros((dim, dim), dtype=complex)

    def bits(val: int):
        return [(val >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]

    gate_tensor = gate.reshape(2, 2, 2, 2)
    for row in range(dim):
        row_bits = bits(row)
        for col in range(dim):
            col_bits = bits(col)
            # unaffected qubits must match
            if any(row_bits[q] != col_bits[q] for q in range(num_qubits) if q not in targets):
                continue
            r_sub = row_bits[control] * 2 + row_bits[target]
            c_sub = col_bits[control] * 2 + col_bits[target]
            val = gate_tensor.ravel()[r_sub * 4 + c_sub]
            if val == 0:
                continue
            op[row, col] = val

    return op.tocsr() if use_sparse else op


#Check if a matrix is unitary: U*U† = I.
def is_unitary(matrix: np.ndarray, atol: float = 1e-8) -> bool:
    conjugate_transpose = matrix.conj().T   #conjugate transpose
    product = conjugate_transpose @ matrix  #multiply U† * U
    identity = np.eye(matrix.shape[0])      #create an identity matrix of the same size    
    is_close = np.allclose(product, identity, atol=atol) #check if product is close to identity
    return is_close
