# System Requirements

This document presents the process of collecting, documenting, and managing the
requirements that define the features and functionalities of the n-qubit quantum
simulator. The requirements are derived from the project specification of the
course *Quantum Information & Algorithms*.

---

## 1. System Overview

The system is a Python-based n-qubit quantum simulator. It is designed to model
the behavior of quantum systems by representing quantum states, applying quantum
gates, simulating noise, and performing measurements.

The simulator must support both pure and mixed quantum states and follow the
mathematical principles of quantum mechanics.

---

## 2. Functional Requirements

### 2.1 State Representation

- The system shall initialize an n-qubit quantum system.
- The system shall support **pure states** represented as state vectors  
  $$|\psi\rangle \in \mathbb{C}^{2^n}$$
- The system shall support **mixed states** represented as density matrices  
  $$\rho \in \mathbb{C}^{2^n \times 2^n}$$
- The system shall allow conversion from pure states to mixed states using  
  $$\rho = |\psi\rangle\langle\psi|$$
- All state operations (evolution, measurement) shall correctly handle both
  representations.

---

### 2.2 Quantum Gate Implementation

- The system shall implement single-qubit gates:
  - Pauli-X, Pauli-Y, Pauli-Z
  - Hadamard (H)
  - Phase gates (S, T, and general phase P(θ))
- The system shall implement multi-qubit gates:
  - CNOT
  - CZ
  - SWAP
- Gates shall be represented as unitary matrices satisfying  
  $$U^\dagger U = I$$
- The system shall support applying gates to **any qubit index** in an n-qubit
  system.

---

### 2.3 State Evolution

- For pure states, gate application shall follow:
  $$|\psi'\rangle = U|\psi\rangle$$
- For mixed states, gate application shall follow:
  $$\rho' = U\rho U^\dagger$$

---

### 2.4 Measurement

- The system shall support **projective measurements** for both pure and mixed
  states.
- Measurements shall not be limited to the computational basis.
- The system shall support **POVM measurements** using arbitrary Kraus operators.
- A measurement shall return:
  - the classical measurement outcome
  - the corresponding post-measurement quantum state
- The internal quantum state shall be updated after measurement.
- Measurement outcomes shall be stored in a classical register.

---

### 2.5 Noise Simulation

- The system shall support configurable noise models.
- The system shall implement:
  - Bit-flip noise: applying Pauli-X with probability \( p \)
  - Depolarizing noise:
    $$\rho' = (1 - p)\rho + \frac{p}{2^n}I$$
- Noise shall be optionally enabled or disabled.

---

## 3. Non-Functional Requirements

### 3.1 Modularity and Maintainability

- The system shall be modular and object-oriented.
- Each module shall have a clear responsibility (state, gates, noise,
  measurement, simulator).
- The code shall follow PEP 8 conventions.
- Functions and classes shall be documented using docstrings.

---

### 3.2 Extensibility

- The system architecture shall allow easy addition of:
  - new quantum gates
  - new noise models
  - additional measurement schemes
- Extensions shall not require major refactoring of existing code.

---

### 3.3 Testing and Documentation

- The system shall include unit tests for core functionality.
- Tests shall verify:
  - unitarity of gates
  - correct state evolution
  - correct measurement probabilities
- Documentation shall include:
  - system architecture
  - key design choices
  - explanation of pure vs. mixed state handling

---

## 4. Constraints

- The simulator shall be implemented in Python.
- Numerical computations shall use NumPy (and optionally SciPy).
- Existing quantum computing libraries (e.g., Qiskit, Cirq) shall not be used.

---

## 5. Assumptions

- The simulator is intended for educational and experimental use.
- Performance optimization for large n is secondary to correctness and clarity.
- Users interact with the simulator programmatically (no GUI required).

