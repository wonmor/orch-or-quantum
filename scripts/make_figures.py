"""Build figures and results/macros.tex from the experiment JSON files. Every number in the paper comes from here."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
e1 = json.load(open("results/e1_scaling.json")); e2 = json.load(open("results/e2_decoherence.json"))
e3 = json.load(open("results/e3_unravelling.json")); e4 = json.load(open("results/e4_circuit.json"))
macros = {}
def m(name, val): macros[name] = val

# ---------- Fig 1: scaling ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
Ns = sorted(int(k) for k in e1)
for n in Ns:
    r = e1[str(n)]; ax[0].plot(np.array(r["t"]) * 1e3, r["M_mean"], lw=1, label=f"N={n}")
ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("tubulins in coherent superposition, M"); ax[0].set_ylim(0, 9.5); ax[0].legend(fontsize=6, ncol=4, frameon=False, loc="upper center")
iv = [e1[str(n)]["mean_interval_ms"] if e1[str(n)]["mean_interval_ms"] is not None else np.nan for n in Ns]
sd = [e1[str(n)]["std_interval_ms"] if e1[str(n)]["std_interval_ms"] is not None else 0.0 for n in Ns]
pred = [e1[str(n)]["predicted_tau_ms"] for n in Ns]
ft = [e1[str(n)]["mean_first_ms"] for n in Ns]
ax[1].errorbar(Ns, iv, yerr=sd, fmt="o", ms=4, label="simulated OR interval")
ax[1].plot(Ns, ft, "^", ms=4, color="tab:orange", label="time to first event")
ax[1].plot(Ns, pred, "k--", lw=1, label=r"$\tau=\hbar/E_G$ at full superposition")
ax[1].set_xlabel("number of tubulin qubits N"); ax[1].set_ylabel("interval between OR events (ms)"); ax[1].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig("figures/fig1_scaling.pdf"); plt.close(fig)
def _iv(n):
    r = e1[str(n)]
    return "--" if r["mean_interval_ms"] is None else f"{r['mean_interval_ms']:.1f} $\\pm$ {r['std_interval_ms']:.1f}"
rows = "".join(f"{n} & {e1[str(n)]['predicted_tau_ms']:.1f} & {_iv(n)} & {e1[str(n)]['mean_first_ms']:.1f} & {e1[str(n)]['M_mean_max']:.2f} \\\\\n" for n in Ns)
m("ScalingRows", rows)
excess = [(e1[str(n)]["mean_interval_ms"] - e1[str(n)]["predicted_tau_ms"]) for n in Ns if e1[str(n)]["mean_interval_ms"] is not None]
m("ExcessMinMs", f"{min(excess):.1f}"); m("ExcessMaxMs", f"{max(excess):.1f}")
m("Nmax", str(max(Ns))); m("IntervalNeight", f"{e1[str(max(Ns))]['mean_interval_ms']:.1f}"); m("PredNeight", f"{e1[str(max(Ns))]['predicted_tau_ms']:.1f}")

# ---------- Fig 2: decoherence ----------
g = np.array([v["gamma"] for v in e2.values()]); rate = np.array([v["event_rate_hz"] for v in e2.values()]); Mavg = np.array([v["M_mean_timeavg"] for v in e2.values()])
order = np.argsort(g); g, rate, Mavg = g[order], rate[order], Mavg[order]
gpos = np.where(g > 0, g, 1.0)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for a, y, lab in [(ax[0], rate, "OR event rate (Hz)"), (ax[1], Mavg, "time-averaged M")]:
    a.semilogx(gpos, y, "o-", ms=4, lw=1); a.set_xlabel(r"environmental dephasing rate $\gamma$ (s$^{-1}$)"); a.set_ylabel(lab)
    ymax = a.get_ylim()[1]
    for x, lab2, c, yf in [(1e4, "Hagan et al. 2002", "tab:red", 0.95), (1e13, "Tegmark 2000", "tab:red", 0.95), (e4["gamma_hw_1_over_T2"], r"$1/T_2$ (IBM Heron)", "tab:gray", 0.50)]:
        a.axvline(x, color=c, lw=0.8, ls=":"); a.text(x * (0.22 if "T_2" in lab2 else 2.2), ymax * yf, lab2, rotation=90, va="top", ha="center", fontsize=6, color=c)
ax[0].axhline(40, color="tab:green", lw=0.8, ls="--"); ax[0].text(1.3, 41, "40 Hz", fontsize=6, color="tab:green")
fig.tight_layout(); fig.savefig("figures/fig2_decoherence.pdf"); plt.close(fig)
r0 = rate[g == 0][0]
# gamma at which the rate halves (log interpolation on the gamma>0 points)
gp, rp = g[g > 0], rate[g > 0]
gstar = None
for i in range(len(gp) - 1):
    if rp[i] >= r0 / 2 > rp[i + 1]:
        f = (rp[i] - r0 / 2) / (rp[i] - rp[i + 1]); gstar = 10 ** (np.log10(gp[i]) + f * (np.log10(gp[i + 1]) - np.log10(gp[i]))); break
if gstar is None and rp[0] < r0 / 2: gstar = gp[0]
m("RateZero", f"{r0:.1f}"); m("GammaStar", f"{gstar:.0f}" if gstar else "n/a"); m("TdecStarMs", f"{1e3/gstar:.0f}" if gstar else "n/a")
m("GapHagan", f"{1e4/gstar:.0f}" if gstar else "n/a"); m("GapTegmarkOrders", f"{np.log10(1e13/gstar):.0f}" if gstar else "n/a")
rows2 = "".join(f"{'0' if gg == 0 else f'{gg:.0e}'.replace('e+0','e').replace('e+','e')} & {rr:.1f} & {mm:.2f} \\\\\n" for gg, rr, mm in zip(g, rate, Mavg))
m("DecoherenceRows", rows2)
m("RateHagan", f"{rate[g == 1e4][0]:.1f}"); m("MHagan", f"{Mavg[g == 1e4][0]:.2f}"); m("MTegmark", f"{Mavg[g == 1e13][0]:.3f}")

# ---------- Fig 3: unravellings ----------
t = np.array(e3["t"]) * 1e3
fig, ax = plt.subplots(figsize=(5.0, 2.7))
ax.plot(t, e3["M_none"], color="tab:gray", lw=1, label="unitary only (no reduction)")
ax.plot(t, e3["M_cont"], color="tab:blue", lw=1.2, label="continuous (Diósi master equation)")
mj = np.array(e3["M_jump_mean"]); sj = np.array(e3["M_jump_std"])
ax.plot(t, mj, color="tab:purple", lw=1.2, label="jump (Penrose OR), ensemble mean")
ax.fill_between(t, mj - sj, mj + sj, color="tab:purple", alpha=0.15)
ax.set_xlabel("time (ms)"); ax.set_ylabel("M"); ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig("figures/fig3_unravelling.pdf"); plt.close(fig)
m("JumpInterval", f"{e3['jump_mean_interval_ms']:.1f}"); m("JumpIntervalSd", f"{e3['jump_std_interval_ms']:.1f}")
m("ContMavg", f"{e3['cont_M_timeavg']:.2f}"); m("JumpMavg", f"{e3['jump_M_timeavg']:.2f}"); m("NoneMavg", f"{e3['none_M_timeavg']:.2f}")

# ---------- Fig 4: circuit ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for n, c in [(3, "tab:blue"), (4, "tab:purple")]:
    runs = sorted([r for r in e4["runs"].values() if r["n"] == n], key=lambda r: r["steps"])
    st = [r["steps"] for r in runs]
    ax[0].plot(st, [r["M_exact"] for r in runs], color=c, lw=1, ls="--", label=f"N={n} exact")
    ax[0].plot(st, [r["M_ideal_tomo"] for r in runs], color=c, lw=1, marker="o", ms=3, label=f"N={n} ideal simulator + tomography")
    ax[0].plot(st, [r["M_noisy_tomo"] for r in runs], color=c, lw=1, marker="s", ms=3, ls=":", label=f"N={n} {e4['backend']} noise model")
    ax[1].plot(st, [r["fidelity_noisy"] for r in runs], color=c, marker="s", ms=3, lw=1, label=f"N={n}")
ax[0].set_xlabel("Trotter steps"); ax[0].set_ylabel("M"); ax[0].legend(fontsize=6, frameon=False)
ax[1].set_xlabel("Trotter steps"); ax[1].set_ylabel("state fidelity under noise"); ax[1].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig("figures/fig4_circuit.pdf"); plt.close(fig)
rows4 = "".join(f"{r['n']} & {r['steps']} & {r['sim_time_ms']:.2f} & {r['two_qubit_gates']} & {r['M_exact']:.2f} & {r['M_ideal_tomo']:.2f} & {r['M_noisy_tomo']:.2f} & {r['fidelity_noisy']:.2f} \\\\\n" for r in sorted(e4["runs"].values(), key=lambda r: (r["n"], r["steps"])))
m("CircuitRows", rows4); m("Backend", e4["backend"].replace("_", "\\_")); m("TtwoMedianUs", f"{e4['T2_median_s']*1e6:.0f}"); m("ToneMedianUs", f"{e4['T1_median_s']*1e6:.0f}")
m("GammaHW", f"{e4['gamma_hw_1_over_T2']:.1e}".replace("e+0", "\\times10^{").replace("e+", "\\times10^{") + "}")
m("GammaHWplain", f"{e4['gamma_hw_1_over_T2']:.0f}")
m("DtTrotterUs", f"{e4['dt_trotter_s']*1e6:.0f}")
worst = max(e4["runs"].values(), key=lambda r: r["steps"] if r["n"] == 4 else -1)
m("FidelityNfourLast", f"{worst['fidelity_noisy']:.2f}"); m("MnoisyNfourLast", f"{worst['M_noisy_tomo']:.2f}"); m("MexactNfourLast", f"{worst['M_exact']:.2f}"); m("StepsLast", str(worst["steps"]))
m("GammaHWoverStar", f"{e4['gamma_hw_1_over_T2']/gstar:.0f}" if gstar else "n/a")

with open("results/macros.tex", "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("figures + macros written:", list(macros)[:8], "...")
