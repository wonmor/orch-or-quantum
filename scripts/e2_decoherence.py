"""E2: event rate and coherence vs environmental dephasing rate, N=6."""
import json, numpy as np, sys
sys.path.insert(0, '.')
from orchq import TubulinChain, run_ensemble, Physics
P = Physics(); n = 6; rep = np.log10(P.calibration_tubulins / 8)
gammas = [0, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5, 1e6, 1e13]
res = {}
for g in gammas:
    t_max = 0.3
    ens = run_ensemble(lambda s: TubulinChain(n=n, dt=1e-5, representation_exponent=rep, gamma_env=g, seed=s), t_max=t_max, n_traj=3)
    iv = ens["intervals"]; nev = len(ens["first_times"]) + len(iv)
    rate = nev / (3 * t_max)
    res[str(g)] = {"gamma": g, "event_rate_hz": rate, "mean_interval_ms": float(iv.mean()*1e3) if len(iv) else None,
                   "M_mean_timeavg": float(ens["M_mean"].mean()), "M_mean_max": float(ens["M_mean"].max())}
    print(res[str(g)], flush=True)
json.dump(res, open("results/e2_decoherence.json", "w"), indent=1)
