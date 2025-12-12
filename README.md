## n-Qubit Quantum Simulator

Python package that simulates pure and mixed \(n\)-qubit states with gates, measurements (projective + POVM), and optional noise channels (bit-flip, depolarizing). Organized for the course project “Quantum Information & Algorithms”.

### Quick start
- Requires Python 3.10+ with `numpy` and `scipy`.
- Create and activate a virtual env, then install deps:  
  - `python -m venv .venv && .venv\Scripts\activate` (Windows)  
  - `pip install -r requirements.txt`
- Run unit tests: `python -m pytest`

### Repo layout
- `src/nqubitsim/`: simulator package  
  - `gates.py`: common single/multi-qubit gates and utilities to lift them to \(n\) qubits.  
  - `state.py`: pure/mixed state containers with dense/sparse handling.  
  - `noise.py`: bit-flip and depolarizing channels; generic Kraus application.  
  - `measurement.py`: projective + POVM measurement helpers.  
  - `simulator.py`: `QuantumSimulator` orchestrating states, gates, measurement, and noise.  
- `tests/`: minimal correctness checks (gates unitary, state evolution, measurements).
- `docs/architecture.md`: design notes and UML-style overview.

### Usage snippet
```python
from nqubitsim.simulator import QuantumSimulator
from nqubitsim import gates

sim = QuantumSimulator(num_qubits=2, noise=None)
sim.apply_gate(gates.H, target=0)
sim.apply_controlled_gate(gates.CNOT, control=0, target=1)
outcome, post = sim.measure([0, 1])
print("outcome", outcome)
print("state", post)
```

### Notes
- Noise is optional; toggle via `noise={"bit_flip": p, "depolarizing": p}` or `None`.
- Sparse switching triggers when the Hilbert space dimension crosses `sparse_threshold` (configurable).
- POVM measurement accepts any list of Kraus operators; validation included.


