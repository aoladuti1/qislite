from itertools import product
from typing import Iterable
import numpy as np
from qiskit import ClassicalRegister, qpy, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import (
    DensityMatrix, Operator, partial_trace, Statevector)
from qiskit.result import Result
from qiskit.circuit import InstructionSet
from qiskit_aer import AerSimulator
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Author: Antonio Oladuti
# Min Python version: 3.10

__all__ = [
    "Ket",
    "Bra",
    "I",
    "cos",
    "sin",
    "uy_to_ry_angle",
    "root_of_unity",
    "basis_states",
    "darkmode",
    "lightmode",
    "defaultmode",
    "QFT",
    "run_circuits",
    "get_all_counts",
    "plot_histograms",
    "get_final_states",
    "multi_measure",
    "measure_all",
    "is_entangled",
    "save_circuit",
    "load_circuit",
    "density_matrix",
    "quickdraw",
    "Figure", # From Matplotlib
    "Operator", # From Qiskit
    "Result", # From Qiskit
    "InstructionSet", # From Qiskit
    "ClassicalRegister",  # From Qiskit
    "QuantumRegister",  # From Qiskit
    "QuantumCircuit"  # From Qiskit
]

class Ket(Statevector):
    """ Statevector subclass which uses a smart __mul__ (*) function 
        to contextually decide whether to return a Ket or Operator. 
        Note, you can construct a Ket by a label string 
        (e.g. `Ket("0") == Ket(Statevector.from_label("0")`)"""
    def __init__(self, data: Iterable | str, dims=None):
        if isinstance(data, str):
            gen = Statevector.from_label(data)
            super().__init__(gen.data, gen.dims())
        else:
            super().__init__(data, dims)

    def dagger(self) -> "Bra":
        """ Return the complex conjugate of this `Ket` as a `Bra`. """
        return Bra(self.data.conj().T)
    
    def evolve(self, operators: Operator | list[Operator]) -> "Ket":
        """Evolve the quantum state by one or more operators.

        Args:
            operators (Operator | list[Operator]): the operator(s)

        Returns:
            Ket: the evolved quantum state
        """
        ret = self
        ops = operators
        for op in ([ops] if isinstance(ops, Operator) else ops):
            ret = Statevector.evolve(ret, op)
        return Ket(ret)

    def __str__(self) -> str:
        ret = ""
        for col in self.data.T:
            ret += str(col.reshape(-1, 1)[0]) + "\n"
        return ret
    
    def __repr__(self) -> str:
        return "Ket" + Statevector.__repr__(self)[11:]

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return Operator(np.outer(self.data, other))
        elif isinstance(other, Statevector):
            return Ket(self.tensor(other))
        else:
            return Ket(super().__mul__(other))
        
    def __rmul__(self, other):
        return Ket(super().__mul__(other))

    def __add__(self, other: "Statevector | Ket"):
        return Ket(super().__add__(other))
    
    def __sub__(self, other: "Statevector | Ket"):
        return Ket(super().__sub__(other))

class Bra(np.ndarray):
    """ NumPy array subclass which uses a smart __mul__ (*) function 
        to contextually decide whether to return a Bra or dot product. 
        Raises ValueError if Bra is not a row vector. """
    def __new__(cls, data) -> 'Bra':
        if isinstance(data, str):
            data = Statevector.from_label(data).data.T
        obj = np.asarray(data).view(cls)
        if len(obj.shape) > 1:
            raise ValueError("Bra must be a row vector.")
        return obj
    
    def dagger(self) -> Ket:
        """ Return the complex conjugate of this `Bra` as a `Ket`. """
        return Ket(self.conj().T)
    
    def __mul__(self, other: "Bra"):
        if isinstance(other, Statevector):
            return self.dot(other.data)
        elif isinstance(other, np.ndarray):
            return Bra(np.multiply(self, other))
        else:
            return Bra(super().__mul__(other))
        
    def __rmul__(self, other):
        if isinstance(other, Statevector):
            return Operator(np.outer(other, self))
        return Bra(super().__mul__(other))
        
    def __add__(self, other: "Bra"):
        return Bra(super().__add__(other))
    
    def __sub__(self, other: "Bra"):
        return Bra(super().__sub__(other))
    
def I(n_dim: int = 2) -> Operator:
    """Return the identity matrix of the given dimension.

    Args:
        n_dim (int, optional): number of dimensions. Defaults to 2.

    Returns:
        Operator: the identity matrix
    """
    return Operator(np.eye(n_dim))

def cos(val):
    """Return cosine element-wise. See `numpy.cos()` for details."""
    return np.cos(val)

def sin(val):
    """Return trigonometric sine element-wise. 
       See `numpy.sin()` for details."""
    return np.sin(val)

def uy_to_ry_angle(theta: float) -> float:
    """Convert a IBM CHSH game U(y) gate angle to the Qiskit Ry gate angle.

    Args:
        theta (float): angle

    Returns:
        float: converted angle
    """
    return -2 * theta

def root_of_unity(idx: int, num_roots: int) -> float:
    """Return the `root_num`th root of unity (1-indexed).

    Args:
        root_num (int): the (1-indexed) target root of unity position

    Returns:
        float: target root of unity
    """
    return np.exp(2j * np.pi * idx / num_roots)

def basis_states(num_qubits: int) -> list[Ket]:
    """Generate a list of Ket basis states for a given number of qubits.

    Args:
        num_qubits (int): Number of qubits in the system.

    Returns:
        list[Ket]: A list of Ket basis states for a given number of qubits.
    """
    return [Ket(''.join(bits)) for bits in product("01", repeat=num_qubits)]


def darkmode():
    """ Set mpl output to 'dark_background'"""
    plt.style.use("dark_background")

def lightmode():
    """ Set mpl graph style to 'seaborn-v0_8-pastel'"""
    plt.style.use("seaborn-v0_8-pastel")

def defaultmode():
    """ Set mpl graph style to 'default'"""
    plt.style.use("default")

def QFT(target_idx: int, all_states: Iterable[Ket | Statevector]) -> Ket:
    """Perform the Quantum Fourier Transform (QFT) on a target index state.

    Args:
        target_idx (int): The index of the target state for the QFT.
        all_states (Iterable[Ket | Statevector]): 
            An iterable of all basis states.

    Returns:
        Ket: The resulting state after applying the QFT, 
             normalized by the square root of the number of possible states.
    """
    n_dim = all_states[0].data.shape[-1]
    ret = Ket(all_states[0]) * root_of_unity(0, n_dim)
    for i in range(1, n_dim):
        ret += all_states[i] * root_of_unity(target_idx * i, n_dim)
    return ret * (1/(np.sqrt(n_dim)))

def run_circuits(
        circuits: list[QuantumCircuit] | QuantumCircuit,
        statevector_sim: bool = False,
        shots: int = 1024) -> Result:
    """Run the circuit or circuits with a default `AerSimulator`
    if statevector_sim is False, else run the circuit 
    with `method="statevector"`.

    Args:
        circuits (list[QuantumCircuit] | QuantumCircuit): the circuit/circuits

        statevector_sim (bool, optional): 
            if True, run with `method="statevector"`. Defaults to False.
        
        shots (int, optional): how many times to run the circuits. 
            Defaults to 1024.

    Returns:
        Result: the result of running the circuit
    """
    if isinstance(circuits, QuantumCircuit):
        circlist = [circuits]
    else:
        circlist = circuits
    if statevector_sim:
        return AerSimulator(
            method="statevector", shots=shots).run(circlist).result()
    else:
        return AerSimulator().run(circlist, shots=shots).result()

def get_all_counts(
        result: Result, titles: list[str] | None = None) -> dict[dict]:
    """Retrieve the measurement counts for all circuits in the result object.
    The returned dict is intended for direct use the first argument for the
    `plot_histograms` function.

    Args:
        result (Result): The result object containing measurement 
            outcomes for multiple circuits.
        titles (list[str] | None, optional): Custom titles for each circuit. 
            Defaults to the circuit names in result.

    Returns:
        dict[dict]: A dictionary where each key is a circuit title 
                    and each value is a dictionary of measurement counts.
    """
    n_circuits = len(result.results)
    if titles is None:
        titles = [res.header.name for res in result.results]
    return {titles[i] : result.get_counts(i) for i in range(n_circuits)}

def _format_hist_data(hist_data: dict) -> dict:
    if isinstance(hist_data, list):
        return {f"Circuit {i+1}": cts for i, cts in enumerate(hist_data)}
    elif isinstance(list(hist_data.items())[0][1], int):
        return {"Results" : hist_data}
    else:
        return hist_data

def plot_histograms(hist_data: dict, show: bool = True) -> Figure:
    """Plot histograms for given measurement counts. `hist_data` will normally
    be the result of a call to `get_all_counts()`, 
    but works with `Result.get_counts()` calls too.

    Args:
        hist_data (dict): Dictionary of measurement counts to plot.
        show (bool, optional): If True, display the plot immediately. 
            Defaults to False.

    Returns:
        Figure: A matplotlib figure object 
                           containing the histograms.
    """
    data = _format_hist_data(hist_data)
    num_subplots = len(data)
    fig, axs = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))
    if num_subplots == 1:  # For single subplot case
        axs = [axs]
    for ax, (outer_key, sub_dict) in zip(axs, data.items()):
        bars = ax.bar(sub_dict.keys(), sub_dict.values())
        ax.set_title(f"{outer_key}")
        ax.set_xlabel("States")
        ax.set_ylabel("Counts")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, 
                    height, 
                    f"{height}", 
                    ha="center", 
                    va="bottom")
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def get_final_states(result: Result) -> list[str]:
    """Retrieve the final states from the measurement counts.

    Args:
        result (Result): The result object containing measurement outcomes.

    Returns:
        list[str]: A list of the final states measured in the circuit.
    """
    return list(result.get_counts().keys())

def multi_measure(
        circuit: QuantumCircuit, *qubit_cbit_tuples) -> list[InstructionSet]:
    """Measure multiple qubit-classical bit pairs within a quantum circuit.
       Bits can be indices or registers. Qubit measurements are placed in
       the classical bits.

    Args:
        circuit (QuantumCircuit): The quantum circuit to measure.
        *qubit_cbit_tuples: (Qubit, Clbit) tuples to be measured. 
            A list of tuples may be passed instead of typing out each pair.

    Returns:
        list[InstructionSet]: A list of measurement instructions 
                              applied to the circuit.
    """
    ret = []
    if isinstance(qubit_cbit_tuples[0], list):
        qubit_cbit_tuples = qubit_cbit_tuples[0]
    for qubit_cbit in qubit_cbit_tuples:
        ret.append(circuit.measure(qubit_cbit[0], qubit_cbit[1]))
    return ret

def measure_all(circuits: QuantumCircuit | list[QuantumCircuit]
               ) -> list[QuantumCircuit]:
    """Apply measurement to all qubits in one or multiple circuits.

    Args:
        circuits (QuantumCircuit | list[QuantumCircuit]): 
            A single quantum circuit or a list of circuits.

    Returns:
        list[QuantumCircuit]: A list of quantum circuits with 
            measurements applied to all qubits.
    """
    if isinstance(circuits, QuantumCircuit):
        circlist = [circuits]
    else:
        circlist = circuits
    ret = []
    for circuit in circlist:
        ret.append(circuit.measure_all())
    return ret

def is_entangled(
        state: Statevector, qubit_indices: Iterable[int] = [0]) -> bool:
    """Check if a subset of qubits in a state is entangled.

    Args:
        state (Statevector): The quantum state to check for entanglement.
        qubit_indices (Iterable[int], optional): 
            Indices of qubits to trace out. Defaults to the first qubit `[0]`.

    Returns:
        bool: True if the subset of qubits are entangled; False otherwise.
    """
    reduced_state = partial_trace(state, qubit_indices)
    return reduced_state.purity() < 1

def save_circuit(circuit: QuantumCircuit, filepath: str):
    """Save a quantum circuit to a file.

    Args:
        circuit (QuantumCircuit): The quantum circuit to save.
        filepath (str): The file path to save the circuit as a binary file.
    """
    with open(filepath, "wb") as file:
        qpy.dump(circuit, file)

def load_circuit(filepath: str) -> QuantumCircuit:
    """Load a quantum circuit from a file.

    Args:
        filepath (str): The file path to load the circuit from.

    Returns:
        QuantumCircuit: The loaded quantum circuit, 
                        or None if no circuit is loaded.
    """
    with open(filepath, "rb") as file:
        circuits = qpy.load(file)
    return circuits[0] if circuits else None

def density_matrix(state: Ket | Statevector) -> DensityMatrix:
    """Convert a quantum state into its density matrix representation.

    Args:
        state (Ket | Statevector): The quantum state to convert.

    Returns:
        DensityMatrix: The density matrix representation of the state.
    """
    return DensityMatrix(state)

def quickdraw(circuit: QuantumCircuit, as_text: bool = False) -> str | Figure:
    """Draw a quantum state or circuit as text or a matplotlib Figure.

    Args:
        circuit: The circuit to draw.
        as_text (bool, optional): If True, draw as text; 
            otherwise, use matplotlib. Defaults to False.

    Returns:
        str or Figure: The drawn representation in 
            text format or as a matplotlib figure.
    """
    if as_text:
        ret = circuit.draw(output="text")
        print(ret)
    else:
        ret = circuit.draw(output="mpl")
    return ret