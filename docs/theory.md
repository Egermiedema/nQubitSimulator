# Quantum Measurement Theory

This document describes the theoretical background of the measurement
implementations used in `measurement.py`. Measurements are formulated using
the density matrix formalism, allowing both pure and mixed quantum states to be
treated in a unified and physically correct manner.

---

## 1. Quantum Measurement

In quantum mechanics, measurement is a probabilistic process that converts a
quantum state into classical information while simultaneously altering the
state itself. Unlike classical systems, quantum measurements are generally
**state-disturbing**.

Given a quantum state represented by a density matrix \( \rho \), the outcome
of a measurement is not deterministic but governed by the Born rule.

All measurements implemented in this project follow three fundamental steps:

1. Compute the probability of each measurement outcome.
2. Randomly sample an outcome according to these probabilities.
3. Update (collapse) the quantum state conditioned on the observed outcome.

---

## 2. Density Matrix Formalism

A quantum state of an \( n \)-qubit system is represented by a density matrix

\[
\rho \in \mathbb{C}^{2^n \times 2^n},
\]

which satisfies:
- \( \rho = \rho^\dagger \) (Hermitian)
- \( \rho \succeq 0 \) (positive semidefinite)
- \( \mathrm{Tr}(\rho) = 1 \)

This representation allows mixed states, entanglement, and partial measurements
to be modeled naturally.

---

## 3. Projective Measurements

### 3.1 Definition

A projective measurement is defined by a set of orthogonal projectors

\[
\{ P_i \}, \quad P_i = | \phi_i \rangle \langle \phi_i |,
\]

satisfying

\[
\sum_i P_i = I.
\]

The probability of obtaining outcome \( i \) is

\[
p_i = \mathrm{Tr}(P_i \rho).
\]

After measurement, the quantum state collapses to

\[
\rho' = \frac{P_i \rho P_i}{p_i}.
\]

---

### 3.2 Computational Basis Measurement

If no custom basis is specified, measurements are performed in the
**computational basis**. For \( k \) measured qubits, this basis consists of
\( 2^k \) orthonormal basis vectors corresponding to all binary bitstrings.

Each basis vector defines a projector, which is then used to evaluate outcome
probabilities.

---

## 4. Measurements on Subsystems

Measurements can be applied to a **subset of qubits** within a larger quantum
system. In this case, the measurement operator must be embedded into the full
Hilbert space:

\[
P_i^{(\text{full})} = P_i \otimes I_{\text{rest}},
\]

where \( I_{\text{rest}} \) acts on all unmeasured qubits.

This embedding ensures that:
- Measurement outcomes depend on entanglement with other qubits.
- Measuring one qubit can collapse the state of the entire system.

---

## 5. POVM Measurements

### 5.1 Generalized Measurements

Projective measurements are a special case of **Positive Operator-Valued
Measures (POVMs)**. A POVM is defined by a set of Kraus operators

\[
\{ K_i \},
\]

satisfying the completeness relation

\[
\sum_i K_i^\dagger K_i = I.
\]

The probability of outcome \( i \) is

\[
p_i = \mathrm{Tr}(K_i \rho K_i^\dagger),
\]

and the post-measurement state is

\[
\rho' = \frac{K_i \rho K_i^\dagger}{p_i}.
\]

---

### 5.2 Use Cases

POVMs allow modeling of:
- Noisy or imperfect measurements
- Weak or partial measurements
- Effective measurements arising from system–environment interactions

In the implementation, Kraus operators may act on the full system or on a
specific subset of qubits.

---

## 6. State Update and Collapse

After a measurement outcome is sampled, the quantum state is updated
**in-place** to reflect the measurement-induced collapse. A new quantum state
object representing the post-measurement state is also returned.

This approach allows:
- Sequential measurements
- Simulation of adaptive measurement strategies
- Clear separation between pre- and post-measurement states

---

## 7. Summary

- Measurements are implemented using the density matrix formalism.
- Projective measurements use orthogonal projectors derived from a basis.
- POVMs use Kraus operators for generalized measurement modeling.
- Measurements can act on subsets of qubits via operator embedding.
- All measurements follow the Born rule and correctly update the quantum state.
