import numpy as np

from nqubitsim import gates


def test_single_qubit_unitary():
    for g in [gates.X, gates.Y, gates.Z, gates.H, gates.S, gates.T]:
        assert gates.is_unitary(g)


def test_two_qubit_unitary():
    for g in [gates.CNOT, gates.CZ, gates.SWAP]:
        assert gates.is_unitary(g)


def test_expand_dimensions():
    op = gates.expand_single_qubit_gate(gates.X, target=0, num_qubits=3)
    assert op.shape == (8, 8)
    op2 = gates.expand_two_qubit_gate(gates.CNOT, control=0, target=2, num_qubits=3)
    assert op2.shape == (8, 8)

