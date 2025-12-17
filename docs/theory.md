# Quantum Measurement Theory

This document describes the theoretical background of the measurement
implementations used in `measurement.py`. Measurements are implemented using
the density matrix formalism, which supports both pure and mixed quantum states.

---

## 1. Quantum Measurement

Quantum measurement is a probabilistic process that extracts classical
information from a quantum state while simultaneously altering that state.

All measurements follow three steps:
1. Compute outcome probabilities
2. Sample an outcome
3. Update (collapse) the quantum state

Outcome probabilities are governed by the Born rule.

---

## 2. Density Matrix Formalism

An n-qubit quantum state is represented by a density matrix

$$
\rho \in \mathbb{C}^{2^n \times 2^n}
$$

with the properties:
- $$ \rho = \rho^\dagger $$ (Hermitian)
- $$ \rho \succeq 0 $$ (positive semidefinite)
- $$ \mathrm{Tr}(\rho) = 1 $$

This representation naturally supports mixed states, entanglement, and partial
measurements.

---

## 3. Projective Measurements

A projective measurement is defined by a set of orthogonal projectors

$$
P_i = | \phi_i \rangle \langle \phi_i |
$$

satisfying

$$
\sum_i P_i = I
$$

The probability of outcome \( i \) is

$$
p_i = \mathrm{Tr}(P_i \rho)
$$

After observing outcome \( i \), the post-measurement state is

$$
\rho' = \frac{P_i \rho P_i}{p_i}
$$

---

## 4. Measurements on Subsystems

Measurements can be applied to a subset of qubits. The corresponding operator
is embedded into the full Hilbert space as

$$
P_i^{(\text{full})} = P_i \otimes I_{\text{rest}}
$$

This ensures that measuring part of the system can collapse the full quantum
state.

---

## 5. POVM Measurements

Generalized measurements (POVMs) are defined by Kraus operators \( K_i \)
satisfying

$$
\sum_i K_i^\dagger K_i = I
$$

The probability of outcome \( i \) is

$$
p_i = \mathrm{Tr}(K_i \rho K_i^\dagger)
$$

The post-measurement state is

$$
\rho' = \frac{K_i \rho K_i^\dagger}{p_i}
$$

POVMs allow modeling noisy, imperfect, or weak measurements.

---

## 6. State Update

After a measurement outcome is sampled, the quantum state is updated in-place
to reflect the collapse. The post-measurement state is also returned as a new
QuantumState object.
