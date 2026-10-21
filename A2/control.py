import argparse, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gridworld import cliff_world, ARROWS

GAMMA, EPS, MAX_STEPS = 1.0, 0.1, 5000


def eps_greedy(Q, s, nA, rng):
    if rng.random() < EPS:
        return rng.randrange(nA)
    best = Q[s].max()
    return rng.choice([a for a in range(nA) if Q[s][a] == best])


def run(algo, env, seed, episodes, alpha):
    rng = random.Random(seed); np.random.seed(seed)
    nA = env.n_actions
    Q = np.zeros((env.n_states, nA))
    steps_log, return_log = np.empty(episodes), np.empty(episodes)
    for ep in range(episodes):
        s = env.reset()
        a = eps_greedy(Q, s, nA, rng)
        steps, G, done = 0, 0.0, False
        while not done and steps < MAX_STEPS:
            s2, r, done, _ = env.step(a)
            a2 = eps_greedy(Q, s2, nA, rng)
            if algo == "sarsa":
                target = r + (0.0 if done else GAMMA * Q[s2, a2])
            else:
                target = r + (0.0 if done else GAMMA * Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
            steps += 1; G += r
        steps_log[ep], return_log[ep] = steps, G
    return steps_log, return_log, Q


def policy_grid(env, Q):
    lines = []
    for r in range(env.rows):
        line = ""
        for c in range(env.cols):
            s = r * env.cols + c
            if s in env.terminals: line += "G "
            elif s in env.cliff: line += "C "
            else: line += ARROWS[int(np.argmax(Q[s]))] + " "
        lines.append(line.rstrip())
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    env = cliff_world()

    res = {}
    for algo, name in [("sarsa", "Sarsa"), ("qlearning", "Q-learning")]:
        S, R, Qs = [], [], []
        for run_id in range(args.runs):
            s, r, Q = run(algo, env, run_id, args.episodes, args.alpha)
            S.append(s); R.append(r); Qs.append(Q)
        res[name] = (np.mean(S, 0), np.mean(R, 0))
        print(f"{name} greedy policy (G=goal, C=cliff):\n{policy_grid(env, np.mean(Qs, 0))}\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (st, rt) in res.items():
        ax[0].plot(st, label=name); ax[1].plot(rt, label=name)
    ax[0].set_ylabel("steps per episode"); ax[0].set_ylim(0, 150)
    ax[0].set_title(f"Steps per episode (mean of {args.runs} runs)")
    ax[1].set_ylabel("sum of rewards per episode"); ax[1].set_ylim(-150, 0)
    ax[1].set_title("Online return per episode")
    for a in ax: a.set_xlabel("episode"); a.grid(alpha=.3); a.legend()
    plt.tight_layout(); plt.savefig("control_curves.png", dpi=150)


if __name__ == "__main__":
    main()