import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMA = 0.9
REWARDS = [0.0, 0.0, 0.0, 0.0, 10.0]   # fixed 5-state chain, same every episode
N = len(REWARDS)
PHI = np.array([[1.0, s / (N - 1)] for s in range(N)])  # 2 shared weights, 5 states


def true_returns():
    G, out = 0.0, np.zeros(N)
    for t in range(N - 1, -1, -1):
        G = REWARDS[t] + GAMMA * G
        out[t] = G
    return out


def mc_episode(w, alpha):
    G, returns = 0.0, np.zeros(N)
    for t in range(N - 1, -1, -1):
        G = REWARDS[t] + GAMMA * G
        returns[t] = G
    for t in range(N):
        w += alpha * (returns[t] - PHI[t] @ w) * PHI[t]
    return w


def td_episode(w, alpha):
    for t in range(N):
        v_next = PHI[t + 1] @ w if t + 1 < N else 0.0
        w += alpha * (REWARDS[t] + GAMMA * v_next - PHI[t] @ w) * PHI[t]
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()

    V_star = true_returns()
    w = {"MC": np.zeros(2), "TD(0)": np.zeros(2)}   # same initial weights
    step = {"MC": mc_episode, "TD(0)": td_episode}
    mae = {k: [] for k in w}
    snap = {k: {} for k in w}
    checkpoints = [1, 10, 25, 50, 100]
    for ep in range(1, args.episodes + 1):
        for k in w:                                  # weights never reset
            w[k] = step[k](w[k], args.alpha)
            v = PHI @ w[k]
            mae[k].append(np.mean(np.abs(v - V_star)))
            if ep in checkpoints:
                snap[k][ep] = (w[k].copy(), v.copy(), mae[k][-1])

    print("True V* =", np.round(V_star, 3).tolist())
    print("| Ep | Algo | w0 | w1 | v(s0) | v(s1) | v(s2) | v(s3) | v(s4) | MAE |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for ep in checkpoints:
        if ep > args.episodes: continue
        for k in w:
            ww, v, m = snap[k][ep]
            print(f"| {ep} | {k} | {ww[0]:.3f} | {ww[1]:.3f} | " +
                  " | ".join(f"{x:.3f}" for x in v) + f" | {m:.3f} |")

    plt.figure(figsize=(7, 4.5))
    for k in mae:
        plt.plot(range(1, args.episodes + 1), mae[k], label=k, lw=1.8)
    plt.xlabel("episode"); plt.ylabel("MAE vs true values")
    plt.title(f"Linear VFA on fixed trajectory (alpha={args.alpha})")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig("vfa_mae.png", dpi=150)


if __name__ == "__main__":
    main()