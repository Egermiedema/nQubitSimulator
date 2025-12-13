import pathlib
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Add src/ to path
ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nqubitsim import gates
from nqubitsim.simulator import QuantumSimulator


# -----------------------------------------------------------
#  MATRIX TABLE (scrollable)
# -----------------------------------------------------------

def show_matrix_as_table(rho, frame):
    """Scrollable matrix on the RIGHT side of the UI."""

    canvas = tk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Fill table cells
    for r, row in enumerate(rho):
        for c, val in enumerate(row):
            ttk.Label(
                scroll_frame,
                text=f"{val.real:.3f}{val.imag:+.3f}j",
                font=("Segoe UI", 11),
                borderwidth=1,
                relief="solid",
                padding=4
            ).grid(row=r, column=c, sticky="nsew")

    # Make columns stretch
    for c in range(len(rho)):
        scroll_frame.grid_columnconfigure(c, weight=1)


# -----------------------------------------------------------
#  GUI WINDOW (LEFT+RIGHT layout)
# -----------------------------------------------------------

def show_results_window(sim, probs, outcome, classical_reg, rho, seed):
    window = tk.Tk()
    window.title("Quantum Simulator — Bell State Results")
    window.geometry("1200x800")   # Wider window for right-side matrix

    # Split screen into left + right
    main_frame = ttk.Frame(window)
    main_frame.pack(fill="both", expand=True)

    left_side = ttk.Frame(main_frame)
    right_side = ttk.Frame(main_frame)

    left_side.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    right_side.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    # ---------- TITLE ----------
    title = ttk.Label(left_side, text="Quantum Simulator — Bell State Demo",
                      font=("Segoe UI", 22, "bold"))
    title.pack(pady=10)

    # ---------- LEFT SIDE CONTENT ----------
    input_frame = ttk.LabelFrame(left_side, text="Input Configuration", padding=15)
    input_frame.pack(fill="x", pady=10)

    ttk.Label(input_frame, text=f"Number of qubits: {sim.num_qubits}",
              font=("Segoe UI", 12)).pack(anchor="w")

    ttk.Label(input_frame, text=f"Noise configuration: {sim.noise_cfg}",
              font=("Segoe UI", 12)).pack(anchor="w")

    ttk.Label(input_frame, text=f"Random seed: {seed}",
              font=("Segoe UI", 12)).pack(anchor="w")

    ttk.Label(input_frame, text="Applied operations: H(0) → CNOT(0→1)",
              font=("Segoe UI", 12)).pack(anchor="w")

    # Probabilities
    prob_frame = ttk.LabelFrame(left_side, text="Output: Probabilities", padding=15)
    prob_frame.pack(fill="x", pady=10)

    labels = ["|00>", "|01>", "|10>", "|11>"]
    for lbl, p in zip(labels, probs):
        ttk.Label(prob_frame, text=f"{lbl}: {p:.6f}",
                  font=("Segoe UI", 12)).pack(anchor="w")

    # Measurement
    meas_frame = ttk.LabelFrame(left_side, text="Output: Measurement", padding=15)
    meas_frame.pack(fill="x", pady=10)

    ttk.Label(meas_frame, text=f"Outcome (integer): {outcome}",
              font=("Segoe UI", 12)).pack(anchor="w")

    ttk.Label(meas_frame, text=f"Classical register: {classical_reg}",
              font=("Segoe UI", 12)).pack(anchor="w")

    # ---------- RIGHT SIDE CONTENT (Density Matrix) ----------
    matrix_frame = ttk.LabelFrame(right_side, text="Density Matrix ρ", padding=10)
    matrix_frame.pack(fill="both", expand=True)

    show_matrix_as_table(rho, matrix_frame)

    window.mainloop()


# -----------------------------------------------------------
#  QUANTUM PROGRAM
# -----------------------------------------------------------

def prepare_bell_pair(sim: QuantumSimulator):
    sim.apply_gate(gates.H, target=0)
    sim.apply_controlled_gate(gates.CNOT, control=0, target=1)


def main():
    seed = 42
    rng = np.random.default_rng(seed)
    #de RNG start in exact dezelfde toestand elke keer dat je runt "seed" (voor debuggen)

    sim = QuantumSimulator(
        num_qubits=2,
        noise={"bit_flip": 0.02, "depolarizing": 0.05},
        rng=rng
    )

    prepare_bell_pair(sim)

    probs = sim.state.probabilities([0, 1])
    outcome, post_state = sim.measure([0, 1])

    show_results_window(
        sim=sim,
        probs=probs,
        outcome=outcome,
        classical_reg=sim.classical_register,
        rho=post_state.density,
        seed=seed
    )


if __name__ == "__main__":
    main()
