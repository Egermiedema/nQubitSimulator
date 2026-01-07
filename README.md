# n-Qubit Quantum Simulator ✅

A compact Python package for simulating pure and mixed n-qubit states with gates, projective and POVM measurements, and simple noise channels (bit-flip, depolarizing). This repository was developed for the "Quantum Information & Algorithms" coursework and is suitable as a lightweight educational simulator.

---

## Quick start ⚡

Requirements
- Python 3.10+
- Required runtime packages: `numpy`, `scipy`
- Optional for development/testing: `pytest`
- Optional for the GUI demo: `matplotlib` (Tkinter is typically included with standard Python installs)

Install
```bash
# Create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install runtime deps
pip install -r requirements.txt

# (optional) Install editable for development
pip install -e .
```

Run tests
```bash
# Run the full test suite
python run_tests.py
# or
python -m pytest
```

---

## Project layout 🔧

- `src/nqubitsim/` — main package
  - `gates.py` — common single- and two-qubit gates and utilities to expand them to an n-qubit operator
  - `state.py` — `QuantumState`: pure / density-matrix representations and helpers (probabilities, conversions)
  - `noise.py` — simple noise channels: `apply_bit_flip`, `apply_depolarizing`
  - `measurement.py` — `projective_measure` and `povm_measure` helpers
  - `simulator.py` — `QuantumSimulator` class tying everything together
- `tests/` — unit tests for gates, simulator behavior, measurements
- `run_simulator.py` / `run_simulatorVisualBellState.py` — example scripts (command-line and GUI demo)
- `docs/` — design notes and architecture diagrams

---

## Quick usage example 📌

```python
from nqubitsim.simulator import QuantumSimulator
from nqubitsim import gates

# Create simulator with 2 qubits and optional noise
sim = QuantumSimulator(num_qubits=2, noise={"bit_flip": 0.02, "depolarizing": 0.05})

# Prepare a Bell pair
sim.apply_gate(gates.H, target=0)
sim.apply_controlled_gate(gates.CNOT, control=0, target=1)

# Get probabilities and measure both qubits
probs = sim.state.probabilities([0, 1])
outcome, post_state = sim.measure([0, 1])
print("Probabilities:", probs)
print("Measured outcome:", outcome)
print("Post-measurement density matrix:\n", post_state.get_density())
```

Notes
- The example scripts add `src/` to `sys.path` so you can run them directly from the repository root.
- For reproducible randomness pass an RNG instance: `rng = numpy.random.default_rng(seed)` to `QuantumSimulator(..., rng=rng)`.

---

## API summary 🧭

- `QuantumSimulator(num_qubits: int, noise: dict|None = None, rng=None)`
  - Fields: `state` (QuantumState), `classical_register` (list of measurement outcomes)
  - Methods:
    - `apply_gate(gate: np.ndarray, target: int = 0)` — apply a single-qubit gate (2×2) at `target`
    - `apply_controlled_gate(gate: np.ndarray, control: int, target: int)` — expand and apply a 4×4 two-qubit gate
    - `measure(qubits: Iterable[int], basis: Sequence[np.ndarray] | None = None)` — projective measurement; returns `(outcome_index, post_state)` and appends outcome to `classical_register`
    - `povm(operators: Sequence[np.ndarray], qubits: Iterable[int] | None = None)` — POVM measurement; returns `(outcome_index, post_state)`
    - `reset()` — reset to |0...0> and clear the classical register

- `QuantumState` helpers
  - `get_vector()`, `get_density()`
  - `set_pure(vec)`, `set_mixed(rho)`, `promote_to_density()`
  - `probabilities(qubits: Iterable[int]) -> ndarray`

- Noise config keys
  - `"bit_flip"`: per-qubit bit-flip probability (applied independently)
  - `"depolarizing"`: global depolarizing strength applied to the density matrix

---

## Examples / Demos 🎛️

- `run_simulator.py` — command-line demo that prepares a Bell pair, measures it, and shows sampling counts
- `run_simulatorVisualBellState.py` — simple Tkinter + matplotlib GUI showing the density matrix and measurement output (requires `matplotlib`)

---

## Contributing & development 🤝

- Tests live in `tests/`. Please add tests for any new behavior or bugfixes.
- Follow the existing lightweight style (simple, readable, well-tested functions).

---

If anything in this README is unclear or you'd like a more elaborate example (e.g., multi-qubit circuits or custom POVMs), open an issue or request an example and I’ll add one. ✨


