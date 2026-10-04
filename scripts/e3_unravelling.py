"""E3: jump (Penrose) vs continuous (Diósi) unravelling, ensemble-averaged M(t), N=6."""
import json, numpy as np, sys
sys.path.insert(0, '.')
from orchq import TubulinChain, run_ensemble, run_trajectory, Physics
P = Physics(); n = 6; rep = np.log10(P.calibration_tubulins / 8)
jump = run_ensemble(lambda s: TubulinChain(n=n, dt=1e-5, representation_exponent=rep, seed=s), t_max=0.2, n_traj=12, mode="jump")
cont = run_trajectory(TubulinChain(n=n, dt=1e-5, representation_exponent=rep, seed=0), 0.2, mode="continuous")
none = run_trajectory(TubulinChain(n=n, dt=1e-5, representation_exponent=rep, seed=0), 0.2, mode="none")
out = {"t": jump["t"].tolist()[::10], "M_jump_mean": jump["M_mean"].tolist()[::10], "M_jump_std": jump["M_std"].tolist()[::10],
       "M_cont": cont["M"].tolist()[::10], "M_none": none["M"].tolist()[::10],
       "jump_mean_interval_ms": float(jump["intervals"].mean()*1e3), "jump_std_interval_ms": float(jump["intervals"].std()*1e3),
       "cont_M_timeavg": float(cont["M"].mean()), "jump_M_timeavg": float(jump["M_mean"].mean()), "none_M_timeavg": float(none["M"].mean())}
print({k: v for k, v in out.items() if not isinstance(v, list)})
json.dump(out, open("results/e3_unravelling.json", "w"), indent=1)
