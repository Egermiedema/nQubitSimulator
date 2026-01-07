## Architecture overview

- `QuantumState`: holds pure vectors or density matrices, automatically normalizes and can promote to density form for noise/POVMs. Switches to sparse storage when Hilbert space dimension exceeds `sparse_threshold`.
- `gates`: defines standard gates (`X`, `Y`, `Z`, `H`, `S`, `T`, `phase`, `CNOT`, `CZ`, `SWAP`) plus expansion helpers to lift 1- and 2-qubit gates to `n` qubits.
- `QuantumSimulator`: orchestrates gate application, measurement, and optional noise (`bit_flip`, `depolarizing`). Maintains a classical register of outcomes.
- `measurement`: projective measurement in computational or custom bases, plus general POVM measurement. Both return the post-measurement state and update the simulator state.
- `noise`: basic channels and a hook to add custom Kraus channels.

### Data flow (sequence sketch)
1. Initialize `QuantumSimulator(num_qubits, noise=...)` → creates `QuantumState` in `|0…0⟩`.
2. `apply_gate` / `apply_controlled_gate` expands gate to full dimension → `QuantumState.apply_unitary` updates state → optional noise hook mutates state.
3. `measure` or `povm` builds projectors/Kraus ops on the chosen qubits, samples outcome, renormalizes the post-state, and records the classical result.

### Extension points
- Add gates by defining the 2x2 or 4x4 matrix in `gates.py`; reuse `expand_*` helpers.
- Add noise by extending `noise.py` and wiring into `_apply_noise`.
- Add algorithms by composing gate/measurement calls from `QuantumSimulator`.

### Testing strategy
- Gate matrices: unitary checks.
- State evolution: compare against analytic results on small systems.
- Measurements: probability distributions sum to 1 and collapse matches projector logic.

### sequence diagram
Sequence Diagrams are interaction diagrams that detail how operations are carried out. They capture the interaction between objects in the context of a collaboration. Sequence Diagrams are time focus and they show the order of the interaction visually by using the vertical axis of the diagram to represent time what messages are sent and when. In our sequence diagram the user performs the actions in the example python script "simulation.py".

<p align="center">
  <img src="pictures/sequence_diagram.png" alt="Project Banner" width="80%">
</p>

