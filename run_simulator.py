import pathlib
import sys
import numpy as np

# Ensure src/ is on path when running directly from repo
# Added because the import of the nqubitsim woulf fail sometimes
ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nqubitsim import gates
from nqubitsim.simulator import QuantumSimulator


def prepare_bell_pair(sim: QuantumSimulator):
    """Create (|00> + |11>) / sqrt(2) on qubits 0 and 1."""
    sim.apply_gate(gates.H, target=0)
    sim.apply_controlled_gate(gates.CNOT, control=0, target=1)


def main():
    # Reproducible RNG for deterministic output
    rng = np.random.default_rng(42)

    #sim = QuantumSimulator(num_qubits=2, noise={"bit_flip": 0.02, "depolarizing": 0.05}, rng=rng)
    sim = QuantumSimulator(num_qubits=2, noise= None, rng=rng)
    prepare_bell_pair(sim)

    # Show state probabilities before measurement
    probs = sim.state.probabilities([0, 1])
    print("Probabilities for |00>,|01>,|10>,|11>:", probs)

    # Single projective measurement of both qubits
    outcome, post = sim.measure([0, 1])
    print(f"Measured outcome (integer index): {outcome}")
    print("Classical register:", sim.classical_register)
    print("Post-measurement density matrix:\n", post.density)

    # Reset and run multiple shots to illustrate sampling
    sim.reset()
    prepare_bell_pair(sim)
    shots = 10
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(shots):
        outcome, _ = sim.measure([0, 1])
        counts[outcome] += 1
        sim.reset()
        prepare_bell_pair(sim)
    print(f"Sampling over {shots} shots (0=|00>, 3=|11>):", counts)


if __name__ == "__main__":
    main()