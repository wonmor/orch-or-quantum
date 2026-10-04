"""
Density-matrix model of Orch-OR-style dynamics on a chain of tubulin qubits.

Each qubit is one (representative) tubulin dimer: |0> = conformation A, |1> = conformation B.
Dynamics per time step dt:
  1. Unitary transverse-field Ising evolution  H = -J Σ Z_i Z_{i+1} - h Σ X_i   (orchestration).
  2. Environmental pure dephasing in the conformation basis at rate gamma per qubit:
         rho_ab <- rho_ab * exp(-gamma * d_H(a,b) * dt)
  3. Diósi–Penrose gravitational self-energy of the coherent superposition:
         E_G(rho) = eps_q * M(rho),   M(rho) = Σ_i 2 || rho^{(i)}_{01} ||_1
     where rho^{(i)}_{01} is the off-diagonal block of rho between the bit-i = 0 and bit-i = 1 subspaces
     (trace norm). M counts the number of tubulins in coherent superposition: it is N for both
     |+>^N and GHZ_N, 0 for any classical mixture, and unaffected by decoherence of *other* qubits.
  4a. Jump unravelling ("Penrose OR"): accumulate S = ∫ E_G dt; when S >= hbar, project onto a
      conformation bitstring sampled from diag(rho), log an event, reset S.
  4b. Continuous unravelling ("Diósi master equation"): rho_ab <- rho_ab * exp(-(eps_q/hbar) d_H(a,b) dt).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.linalg import expm


@dataclass(frozen=True)
class Physics:
    hbar: float = 1.054_571_817e-34          # J s
    calibration_tubulins: float = 2.0e10     # Hameroff & Penrose 2014
    calibration_tau: float = 0.025           # s
    tegmark_gamma: float = 1e13              # 1/s  (Tegmark 2000: t_dec ~ 1e-13 s)
    hagan_gamma: float = 1e4                 # 1/s  (Hagan et al. 2002: t_dec ~ 1e-4 s)

    @property
    def eps_tubulin(self) -> float:
        """Gravitational self-energy per tubulin in superposition, J."""
        return self.hbar / (self.calibration_tau * self.calibration_tubulins)


def hamming_matrix(n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    x = idx[:, None] ^ idx[None, :]
    d = np.zeros_like(x)
    for b in range(n):
        d += (x >> b) & 1
    return d.astype(float)


def pauli_chain_hamiltonian(n: int, J: float, h: float, periodic: bool = False) -> np.ndarray:
    """H/hbar in rad/s: -J Σ Z Z - h Σ X (J, h in rad/s)."""
    I = np.eye(2); X = np.array([[0, 1], [1, 0]], dtype=complex); Z = np.diag([1.0, -1.0]).astype(complex)

    def op(single, pos):
        out = np.array([[1.0 + 0j]])
        for q in range(n):
            out = np.kron(out, single if q == pos else I)
        return out

    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    pairs = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if periodic and n > 2 else [])
    for i, j in pairs:
        H -= J * op(Z, i) @ op(Z, j)
    for i in range(n):
        H -= h * op(X, i)
    return H


@dataclass
class TubulinChain:
    n: int
    J: float = 2 * np.pi * 400.0        # rad/s, orchestration coupling
    h: float = 2 * np.pi * 400.0        # rad/s, transverse drive (MAP / dipole field)
    dt: float = 1e-5                    # s
    representation_exponent: float = np.log10(2.0e10 / 8)   # tubulins per qubit (default: 8 qubits ↔ 2e10)
    gamma_env: float = 0.0              # 1/s environmental dephasing per qubit
    periodic: bool = False
    physics: Physics = field(default_factory=Physics)
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.dim = 2 ** self.n
        self.H = pauli_chain_hamiltonian(self.n, self.J, self.h, self.periodic)
        self.U = expm(-1j * self.H * self.dt)
        self.Ud = self.U.conj().T
        self.D = hamming_matrix(self.n)
        self.eps_q = self.physics.eps_tubulin * 10 ** self.representation_exponent
        self.env_factor = np.exp(-self.gamma_env * self.D * self.dt)
        self.dp_factor = np.exp(-(self.eps_q / self.physics.hbar) * self.D * self.dt)
        # masks for the bit-i blocks
        idx = np.arange(self.dim)
        self.bit0 = [np.where(((idx >> (self.n - 1 - i)) & 1) == 0)[0] for i in range(self.n)]
        self.bit1 = [np.where(((idx >> (self.n - 1 - i)) & 1) == 1)[0] for i in range(self.n)]

    def initial_state(self) -> np.ndarray:
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        rho[0, 0] = 1.0                      # all tubulins in conformation A
        return rho

    def superposed_count(self, rho: np.ndarray) -> float:
        """M(rho): number of tubulins in coherent superposition (trace-norm of off-diagonal bit blocks)."""
        m = 0.0
        for i in range(self.n):
            block = rho[np.ix_(self.bit0[i], self.bit1[i])]
            m += 2.0 * np.linalg.svd(block, compute_uv=False).sum()
        return float(m)

    def coherence_l1(self, rho: np.ndarray) -> float:
        return float(np.abs(rho).sum() - np.trace(np.abs(rho)).real)

    def self_energy(self, rho: np.ndarray) -> float:
        return self.eps_q * self.superposed_count(rho)

    def tau(self, rho: np.ndarray) -> float:
        e = self.self_energy(rho)
        return self.physics.hbar / e if e > 0 else np.inf

    def step_unitary(self, rho):
        return self.U @ rho @ self.Ud

    def step_env(self, rho):
        return rho * self.env_factor if self.gamma_env > 0 else rho

    def step_dp_continuous(self, rho):
        return rho * self.dp_factor

    def collapse(self, rho):
        p = np.real(np.diag(rho)).clip(0)
        p /= p.sum()
        a = self.rng.choice(self.dim, p=p)
        out = np.zeros_like(rho)
        out[a, a] = 1.0
        return out, a


def run_trajectory(chain: TubulinChain, t_max: float, mode: str = "jump", record_every: int = 10):
    """Evolve one trajectory. mode: 'jump' (Penrose OR events), 'continuous' (Diósi dephasing), 'none'."""
    rho = chain.initial_state()
    S = 0.0
    t = 0.0
    times, M, C, Sacc = [], [], [], []
    events = []
    steps = int(round(t_max / chain.dt))
    for k in range(steps):
        rho = chain.step_unitary(rho)
        rho = chain.step_env(rho)
        if mode == "continuous":
            rho = chain.step_dp_continuous(rho)
        m = chain.superposed_count(rho)
        if mode == "jump":
            S += chain.eps_q * m * chain.dt
            if S >= chain.physics.hbar:
                rho, a = chain.collapse(rho)
                events.append({"t": t, "M_at_collapse": m, "outcome": int(a)})
                S = 0.0
                m = 0.0
        t += chain.dt
        if k % record_every == 0:
            times.append(t); M.append(m); C.append(chain.coherence_l1(rho)); Sacc.append(S / chain.physics.hbar)
    return {"t": np.array(times), "M": np.array(M), "C_l1": np.array(C), "S_over_hbar": np.array(Sacc), "events": events}


def run_ensemble(make_chain, t_max: float, n_traj: int, mode: str = "jump"):
    """Average M(t) and collect inter-event intervals over independent trajectories."""
    Ms, intervals, first_times = [], [], []
    tgrid = None
    for s in range(n_traj):
        chain = make_chain(s)
        out = run_trajectory(chain, t_max, mode)
        tgrid = out["t"]
        Ms.append(out["M"])
        ev = [e["t"] for e in out["events"]]
        if ev:
            first_times.append(ev[0])
            intervals.extend(np.diff(ev).tolist())
    return {"t": tgrid, "M_mean": np.mean(Ms, axis=0), "M_std": np.std(Ms, axis=0),
            "intervals": np.array(intervals), "first_times": np.array(first_times)}
