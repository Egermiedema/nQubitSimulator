# Quantum Measurement Theory

This document describes the theoretical background of the measurement
implementations used in `measurement.py`. Measurements are implemented using
the density matrix formalism, which supports both pure and mixed quantum states.

---

## 1. Quantum Measurement

In quantum mechanics, measurement is a probabilistic process that extracts
classical information from a quantum state while simultaneously altering that
state.

All measurements in this project follow the same structure:
1. Compute outcome probabilities
2. Sample an outcome according to these probabilities
3. Update (collapse) the quantum state

The probabilities are governed by the Born rule.

---

## 2. Density Matrix Formalism

An n-qubit quantum state is represented by a density matrix rho:

rho ∈ C^(2^n × 2^n)

The density matrix satisfies:
- rho = rho† (Hermitian)
- rho ≥ 0  (positive semidefinite)
- Tr(rho) = 1

This formalism naturally supports mixed states, entanglement, and partial
measurements.

---

## 3. Projective Measurements

A projective measurement is defined by a set of orthogonal projectors {P_i}:

P_i = |φ_i⟩⟨φ_i|

with:
Σ_i P_i = I

The probability of outcome i is:

p_i = Tr(P_i · rho)

After observing outcome i, the post-measurement state is:

rho' = (P_i · rho · P_i) / p_i

---

## 4. Measurements on Subsystems

Measurements can be applied to a subset of qubits within a larger system.
In this case, the measurement operator is embedded into the full Hilbert space:

P_i(full) = P_i ⊗ I_rest

This ensures that:
- Measurement outcomes depend on entanglement
- Measuring part of the system can collapse the full state

---

## 5. POVM Measurements

Generalized measurements are implemented using POVMs, defined by Kraus
operators {K_i} satisfying:

Σ_i (K_i† · K_i) = I

The probability of outcome i is:

p_i = Tr(K_i · rho · K_i†)

The post-measurement state is:

rho' = (K_i · rho · K_i†) / p_i

POVMs allow modeling noisy, imperfect, or weak measurements.

---

## 6. State Update

After a measurement outcome is sampled, the quantum state is updated in-place
to reflect the collapse. The post-measurement state is also returned as a new
QuantumState object, allowing further processing.
