# Quantum Simulation Theory

This document summarizes the theoretical foundations of the quantum simulation
framework. The system models quantum states, unitary evolution, noise channels,
and measurements using standard quantum mechanics.

---

## 1. Quantum State Representation

An n-qubit quantum system lives in a Hilbert space of dimension

$$
d = 2^n
$$

A quantum state is represented either as:
- a **state vector** $( |\psi\rangle \in \mathbb{C}^{2^n} \)$ (pure state), or
- a **density matrix** $( \rho \in \mathbb{C}^{2^n \times 2^n} \)$ (mixed state)

The density matrix satisfies:

$$
\rho = \rho^\dagger, \quad \rho \succeq 0, \quad \mathrm{Tr}(\rho) = 1
$$

Pure states are related to density matrices via:

$$
\rho = |\psi\rangle \langle \psi|
$$

---

## 2. Unitary Evolution and Quantum Gates

Quantum gates are represented by **unitary matrices** \( U \) satisfying:

$$
U^\dagger U = I
$$

State evolution is given by:
- Pure state:
$|\psi'\rangle = U |\psi\rangle$
- Mixed state:
$\rho' = U \rho U^\dagger$

Single-qubit and two-qubit gates are expanded to the full system size using
Kronecker products, e.g.:

$$
U_{\text{full}} = I \otimes \cdots \otimes U \otimes \cdots \otimes I
$$

with qubit 0 as the leftmost factor.

---

## 3. Noise Channels

Noise is modeled as quantum channels acting on the density matrix.

### Bit-flip noise  
Each qubit undergoes a Pauli-X operation with probability \( p \).

### Depolarizing noise  
The depolarizing channel is defined as:

$$
\rho' = (1 - p)\rho + \frac{p}{d} I
$$

where \( d = 2^n \).

Noise is applied after gate operations.

---

## 4. Projective Measurements

Projective measurements are defined by projectors:

$$
P_i = |\phi_i\rangle \langle \phi_i|, \quad \sum_i P_i = I
$$

The probability of outcome \( i \) is:

$$
p_i = \mathrm{Tr}(P_i \rho)
$$

After observing outcome \( i \), the state collapses to:

$$
\rho' = \frac{P_i \rho P_i}{p_i}
$$

Measurements may act on a subset of qubits, with projectors embedded into the
full Hilbert space.

---

## 5. POVM Measurements

Generalized measurements (POVMs) are defined by Kraus operators \( K_i \)
satisfying:

$$
\sum_i K_i^\dagger K_i = I
$$

The outcome probability is:

$$
p_i = \mathrm{Tr}(K_i \rho K_i^\dagger)
$$

The post-measurement state is:

$$
\rho' = \frac{K_i \rho K_i^\dagger}{p_i}
$$

POVMs allow modeling noisy or non-ideal measurements.

---

## 6. Simulation Flow

The quantum simulator maintains:
- a quantum state
- an optional noise configuration
- a classical register for measurement outcomes

A typical simulation step consists of:
1. Applying unitary gates
2. Applying noise (if enabled)
3. Performing measurements and updating the classical register

The simulator supports reset, projective measurements, and POVM measurements.

---

## 7. Summary

- States are represented using vectors or density matrices
- Gates are unitary and expanded to full-system operators
- Noise is modeled via quantum channels
- Measurements follow the Born rule
- The simulator orchestrates state evolution, noise, and measurement
