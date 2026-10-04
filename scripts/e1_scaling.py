"""E1: OR interval vs number of tubulin qubits, jump unravelling, no environmental decoherence."""
import json, numpy as np, sys
sys.path.insert(0, '.')
from orchq import TubulinChain, run_ensemble, Physics
P = Physics()
rep = np.log10(P.calibration_tubulins / 8)   # 8 qubits <-> 2e10 tubulins
res = {}
for n in range(2, 9):
    dt = 1e-5 if n <= 6 else 2e-5
    ens = run_ensemble(lambda s: TubulinChain(n=n, dt=dt, representation_exponent=rep, seed=s), t_max=0.16, n_traj=5 if n < 8 else 4)
    iv = ens["intervals"]; ft = ens["first_times"]
    pred = P.calibration_tau * 8 / n
    res[n] = {"mean_interval_ms": float(iv.mean()*1e3) if len(iv) else None, "std_interval_ms": float(iv.std()*1e3) if len(iv) else None,
              "n_intervals": int(len(iv)), "mean_first_ms": float(ft.mean()*1e3) if len(ft) else None,
              "predicted_tau_ms": pred*1e3, "M_mean_max": float(ens["M_mean"].max()), "t": ens["t"].tolist()[::20], "M_mean": ens["M_mean"].tolist()[::20]}
    print(n, {k: v for k, v in res[n].items() if k not in ("t", "M_mean")}, flush=True)
json.dump(res, open("results/e1_scaling.json", "w"), indent=1)
