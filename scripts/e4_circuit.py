"""E4: Trotter circuit on ideal and noisy (fake IBM Heron) simulators; recover M via tomography."""
import json, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino
from orchq import TubulinChain, Physics
from orchq.circuit import trotter_circuit, run_tomography
from qiskit.quantum_info import DensityMatrix

P = Physics()
fake = FakeTorino()
# median T1/T2 from the fake backend's calibration data
t1s, t2s = [], []
for q in range(fake.num_qubits):
    try:
        props = fake.target.qubit_properties[q]
        if props.t1: t1s.append(props.t1)
        if props.t2: t2s.append(props.t2)
    except Exception:
        pass
T1, T2 = float(np.median(t1s)), float(np.median(t2s))
ideal = AerSimulator()
noisy = AerSimulator.from_backend(fake)
J = h = 2 * np.pi * 400.0
dt_trotter = 2.5e-4            # s of simulated time per Trotter step (angle 2 J dt = 0.63 rad)
res = {"backend": fake.name, "T1_median_s": T1, "T2_median_s": T2, "gamma_hw_1_over_T2": 1 / T2, "J_rad_s": J, "h_rad_s": h,
       "dt_trotter_s": dt_trotter, "runs": {}}
for n in [3, 4]:
    chain = TubulinChain(n=n, representation_exponent=np.log10(P.calibration_tubulins / 8))
    for steps in [1, 2, 4, 8, 12]:
        qc = trotter_circuit(n, J, h, dt_trotter, steps)
        exact = DensityMatrix(qc).data
        M_exact = chain.superposed_count(exact)
        rho_i, _ = run_tomography(qc, ideal, shots=2000)
        rho_n, tc = run_tomography(qc, noisy, shots=2000)
        depth = int(np.median([c.depth() for c in tc])); twoq = int(np.median([sum(1 for inst in c.data if inst.operation.num_qubits == 2) for c in tc]))
        fid = float(np.real(np.trace(rho_n @ exact)))
        r = {"n": n, "steps": steps, "sim_time_ms": steps * dt_trotter * 1e3, "M_exact": M_exact, "M_ideal_tomo": chain.superposed_count(rho_i),
             "M_noisy_tomo": chain.superposed_count(rho_n), "fidelity_noisy": fid, "transpiled_depth": depth, "two_qubit_gates": twoq}
        res["runs"][f"{n}_{steps}"] = r
        print(r, flush=True)
json.dump(res, open("results/e4_circuit.json", "w"), indent=1)
print("T1", T1, "T2", T2, "gamma_hw", 1 / T2)
