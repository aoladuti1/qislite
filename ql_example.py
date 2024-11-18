from qislite import *

if __name__ == '__main__':
    lightmode()
    X = QuantumRegister(1, "X")
    Y = QuantumRegister(1, "Y")
    A = ClassicalRegister(1, "A")
    B = ClassicalRegister(1, "B")
    circuit = QuantumCircuit(Y, X, B, A)
    circuit.h(Y)
    circuit.cx(Y, X)
    multi_measure(circuit, (Y, B), (X, A)) # == circuit.measure_all()
    quickdraw(circuit)
    results = get_all_counts(run_circuits(circuit, True))
    plot_histograms(results, show=True)