"""Trotterised transverse-field-Ising circuit for the tubulin chain, plus Pauli tomography to recover rho."""
from __future__ import annotations
import itertools, numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli


def trotter_circuit(n: int, J: float, h: float, dt: float, steps: int) -> QuantumCircuit:
    """exp(-i H dt)^steps with H/hbar = -J Σ Z Z - h Σ X, first-order Trotter."""
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for i in range(n - 1):
            qc.rzz(-2 * J * dt, i, i + 1)
        for i in range(n):
            qc.rx(-2 * h * dt, i)
    return qc


def tomography_circuits(base: QuantumCircuit):
    n = base.num_qubits
    circs, bases = [], []
    for basis in itertools.product("XYZ", repeat=n):
        qc = base.copy()
        for q, b in enumerate(basis):
            if b == "X": qc.h(q)
            elif b == "Y": qc.sdg(q); qc.h(q)
        qc.measure_all()
        circs.append(qc); bases.append(basis)
    return circs, bases


def reconstruct(counts_list, bases, n):
    """Linear-inversion tomography from counts in all 3^n Pauli bases (qubit 0 = rightmost bit in keys)."""
    dim = 2 ** n
    rho = np.zeros((dim, dim), dtype=complex)
    expect = {}
    for counts, basis in zip(counts_list, bases):
        total = sum(counts.values())
        for mask in itertools.product([0, 1], repeat=n):          # which qubits included (else identity)
            label = "".join(basis[q] if mask[q] else "I" for q in range(n))
            if label in expect: continue
            val = 0.0
            for bitstr, c in counts.items():
                bits = bitstr.replace(" ", "")[::-1]               # bits[q] for qubit q
                s = 1
                for q in range(n):
                    if mask[q] and bits[q] == "1": s = -s
                val += s * c
            expect[label] = val / total
    for label, val in expect.items():
        # Pauli(label) uses qiskit little-endian ordering: label[0] is qubit n-1
        rho += val * Pauli(label[::-1]).to_matrix()
    rho /= dim
    # project onto the PSD cone and renormalise (Smolin et al. 2012 would be better; clipping is adequate here)
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, 0, None); w /= w.sum()
    return (v * w) @ v.conj().T


def run_tomography(base: QuantumCircuit, backend, shots: int = 2000, optimization_level: int = 1):
    circs, bases = tomography_circuits(base)
    tcircs = transpile(circs, backend, optimization_level=optimization_level, seed_transpiler=7)
    job = backend.run(tcircs, shots=shots)
    result = job.result()
    counts = [result.get_counts(i) for i in range(len(tcircs))]
    return reconstruct(counts, bases, base.num_qubits), tcircs
