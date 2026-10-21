import argparse, random, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gridworld import four_by_four

GAMMA = 1.0


def true_values(env):
    n = env.n_states
    A, b = np.eye(n), np.zeros(n)
    for s in range(n):
        if s in env.terminals:
            continue
        for a in range(env.n_actions):
            s2, r, done = env.P[s][a]
            p = 1.0 / env.n_actions
            b[s] += p * r
            if not done:
                A[s, s2] -= GAMMA * p
    return np.linalg.solve(A, b)


def run(method, env, v_true, seed, episodes, alpha):
    random.seed(seed); np.random.seed(seed)
    n, nA = env.n_states, env.n_actions
    V, N = np.zeros(n), np.zeros(n)
    mask = np.array([s not in env.terminals for s in range(n)])
    errs = np.empty(episodes)
    for ep in range(episodes):
        s = env.reset()
        if method == "td0":
            done = False
            while not done:
                s2, r, done, _ = env.step(random.randrange(nA))
                target = r + (0.0 if done else GAMMA * V[s2])
                V[s] += alpha * (target - V[s])
                s = s2
        else:
            states, rewards, done = [], [], False
            while not done:
                s2, r, done, _ = env.step(random.randrange(nA))
                states.append(s); rewards.append(r)
                s = s2
            first = method.startswith("mc_first")
            firsts = {}
            if first:
                for t, st in enumerate(states):
                    firsts.setdefault(st, t)
            G = 0.0
            for t in range(len(states) - 1, -1, -1):
                G = rewards[t] + GAMMA * G
                st = states[t]
                if first and firsts[st] != t:
                    continue
                if method.endswith("inc"):
                    N[st] += 1
                    V[st] += (G - V[st]) / N[st]
                else:
                    V[st] += alpha * (G - V[st])
        errs[ep] = np.sqrt(np.mean((V[mask] - v_true[mask]) ** 2))
    return errs, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()

    env = four_by_four()
    v_true = true_values(env)
    print("True V_pi:\n", np.round(v_true.reshape(4, 4), 2))

    methods = {
        "MC first-visit (inc. mean)": "mc_first_inc",
        f"MC first-visit (alpha={args.alpha})": "mc_first_alpha",
        "MC every-visit (inc. mean)": "mc_every_inc",
        f"MC every-visit (alpha={args.alpha})": "mc_every_alpha",
        f"TD(0) (alpha={args.alpha})": "td0",
    }
    curves = {}
    for label, m in methods.items():
        t0 = time.time()
        errs = [run(m, env, v_true, r, args.episodes, args.alpha)[0] for r in range(args.runs)]
        curves[label] = np.mean(errs, axis=0)
        print(f"{label:36s} final RMS={curves[label][-1]:.4f} ({time.time()-t0:.0f}s)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, c in curves.items():
        ax[0].plot(c, label=label, lw=1.2)
        ax[1].plot(c[:500], label=label, lw=1.2)
    ax[0].set_yscale("log")
    ax[0].set_title("RMS error, all episodes (log y)")
    ax[1].set_title("RMS error, first 500 episodes")
    for a in ax:
        a.set_xlabel("episode"); a.set_ylabel(f"RMS error (mean of {args.runs} runs)"); a.grid(alpha=.3)
    ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig("prediction_rms.png", dpi=150)


if __name__ == "__main__":
    main()