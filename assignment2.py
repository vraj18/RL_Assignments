import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ASSIGNMENT 2: Monte Carlo, TD(0), SARSA, Q-learning,
#              and Value Function Approximation
# Dependencies: NumPy, Matplotlib only
# ============================================================

SEED_RUNS = 50
PREDICTION_EPISODES = 10000
CONTROL_EPISODES = 500
APPROX_EPISODES = 100

ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.1
MAX_STEPS = 200            # cap for control episodes
PRED_MAX_STEPS = 5000      # cap for prediction episodes (avoids biased truncated MC returns)

ROWS, COLS = 5, 5
START = (0, 0)
GOAL = (4, 4)
WALLS = {(1, 1), (1, 3), (3, 1)}

STEP_REWARD = -1.0
GOAL_REWARD = 10.0

# Up, down, left, right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_SYMBOLS = ["↑", "↓", "←", "→"]

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


# ============================================================
# 1. GRIDWORLD ENVIRONMENT
# ============================================================

class GridWorld:
    def __init__(self):
        self.states = [
            (r, c)
            for r in range(ROWS)
            for c in range(COLS)
            if (r, c) not in WALLS
        ]
        self.state_to_id = {state: i for i, state in enumerate(self.states)}
        self.id_to_state = {i: state for state, i in self.state_to_id.items()}
        self.n_states = len(self.states)
        self.n_actions = len(ACTIONS)
        self.reset()

    def reset(self):
        self.state = START
        return self.state_to_id[self.state]

    def step(self, action):
        r, c = self.state
        dr, dc = ACTIONS[action]
        candidate = (r + dr, c + dc)

        if (
            candidate[0] < 0 or candidate[0] >= ROWS
            or candidate[1] < 0 or candidate[1] >= COLS
            or candidate in WALLS
        ):
            candidate = self.state

        self.state = candidate
        done = candidate == GOAL
        reward = GOAL_REWARD if done else STEP_REWARD
        return self.state_to_id[candidate], reward, done


def epsilon_greedy(q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    best = np.flatnonzero(q[state] == np.max(q[state]))
    return int(rng.choice(best))


def generate_episode(env, rng, policy=None, max_steps=MAX_STEPS):
    state = env.reset()
    episode = []

    for _ in range(max_steps):
        if policy is None:
            action = int(rng.integers(env.n_actions))
        else:
            action = policy(state)

        next_state, reward, done = env.step(action)
        episode.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            break

    return episode


# ============================================================
# 1A. MONTE CARLO PREDICTION
# First-visit and every-visit, constant alpha and incremental mean
# ============================================================

def mc_prediction_update(v, counts, episode, mode, alpha):
    returns = []
    G = 0.0

    for s, a, reward, ns, done in reversed(episode):
        G = reward + GAMMA * G
        returns.append((s, G))

    returns.reverse()
    visited = set()

    for s, G in returns:
        if mode.startswith("first") and s in visited:
            continue
        visited.add(s)

        if mode.endswith("mean"):
            counts[s] += 1
            step_size = 1.0 / counts[s]
        else:
            step_size = alpha

        v[s] += step_size * (G - v[s])


# ============================================================
# 1B. TD(0) PREDICTION
# ============================================================

def td_prediction_update(v, episode, alpha):
    for s, a, reward, ns, done in episode:
        target = reward
        if not done:
            target += GAMMA * v[ns]
        v[s] += alpha * (target - v[s])


def policy_values(env):
    """Exact V_pi of the equiprobable random policy (solves the Bellman equations)."""
    n = env.n_states
    A, b = np.eye(n), np.zeros(n)

    for s, state in env.id_to_state.items():
        if state == GOAL:
            continue

        for a, (dr, dc) in enumerate(ACTIONS):
            nxt = (state[0] + dr, state[1] + dc)

            if (
                nxt[0] < 0 or nxt[0] >= ROWS
                or nxt[1] < 0 or nxt[1] >= COLS
                or nxt in WALLS
            ):
                nxt = state

            done = nxt == GOAL
            b[s] += 0.25 * (GOAL_REWARD if done else STEP_REWARD)

            if not done:
                A[s, env.state_to_id[nxt]] -= 0.25 * GAMMA

    return np.linalg.solve(A, b)


def run_prediction():
    env = GridWorld()
    reference = policy_values(env)
    mask = np.array([env.id_to_state[i] != GOAL for i in range(env.n_states)])

    names = [
        "First visit - constant alpha",
        "Every visit - constant alpha",
        "First visit - incremental mean",
        "Every visit - incremental mean",
        "TD(0)",
    ]

    errors = {name: np.zeros(PREDICTION_EPISODES) for name in names}

    for run in range(SEED_RUNS):
        rng = np.random.default_rng(run)
        values = {name: np.zeros(env.n_states) for name in names}
        counts = {name: np.zeros(env.n_states, dtype=int) for name in names}

        for ep in range(PREDICTION_EPISODES):
            episode = generate_episode(env, rng, max_steps=PRED_MAX_STEPS)

            mc_prediction_update(values[names[0]], counts[names[0]], episode, "first_constant", ALPHA)
            mc_prediction_update(values[names[1]], counts[names[1]], episode, "every_constant", ALPHA)
            mc_prediction_update(values[names[2]], counts[names[2]], episode, "first_mean", ALPHA)
            mc_prediction_update(values[names[3]], counts[names[3]], episode, "every_mean", ALPHA)
            td_prediction_update(values[names[4]], episode, ALPHA)

            for name in names:
                rms = np.sqrt(np.mean((values[name][mask] - reference[mask]) ** 2))
                errors[name][ep] += rms / SEED_RUNS

    plt.figure(figsize=(10, 6))
    for name in names:
        plt.plot(errors[name], label=name, linewidth=1)
    plt.yscale("log")
    plt.xlabel("Episode")
    plt.ylabel("RMS error vs true V_pi (random policy)")
    plt.title(f"MC Prediction vs TD(0) Prediction (mean of {SEED_RUNS} runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/prediction_rms.png", dpi=160)
    plt.close()

    with open(f"{OUT}/prediction_rms.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode"] + names)
        for ep in range(PREDICTION_EPISODES):
            writer.writerow([ep + 1] + [errors[name][ep] for name in names])

    print("Part 1 complete: prediction_rms.png")
    for name in names:
        print(f"  {name:32s} RMS@100={errors[name][min(99, PREDICTION_EPISODES-1)]:.3f}  "
              f"final={errors[name][-1]:.3f}")


# ============================================================
# 2A. SARSA
# 2B. Q-LEARNING
# ============================================================

def train_control(env, algorithm, seed):
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions))
    steps = np.zeros(CONTROL_EPISODES)

    for ep in range(CONTROL_EPISODES):
        state = env.reset()
        action = epsilon_greedy(q, state, EPSILON, rng)

        for t in range(1, MAX_STEPS + 1):
            ns, reward, done = env.step(action)

            if algorithm == "SARSA":
                if done:
                    target = reward
                    next_action = 0
                else:
                    next_action = epsilon_greedy(q, ns, EPSILON, rng)
                    target = reward + GAMMA * q[ns, next_action]
            else:  # Q-learning
                target = reward
                if not done:
                    target += GAMMA * np.max(q[ns])
                next_action = epsilon_greedy(q, ns, EPSILON, rng) if not done else 0

            q[state, action] += ALPHA * (target - q[state, action])

            state = ns
            steps[ep] = t

            if done:
                break

            action = next_action

    return q, steps


def save_policy(env, q, title, filename):
    policy = np.argmax(q, axis=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which="minor", linewidth=1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    for r in range(ROWS):
        for c in range(COLS):
            state = (r, c)
            if state in WALLS:
                symbol = "■"
            elif state == START:
                symbol = "S"
            elif state == GOAL:
                symbol = "G"
            else:
                symbol = ACTION_SYMBOLS[policy[env.state_to_id[state]]]
            ax.text(c, r, symbol, ha="center", va="center", fontsize=18)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def run_control():
    env = GridWorld()
    curves = {"SARSA": np.zeros(CONTROL_EPISODES), "Q-learning": np.zeros(CONTROL_EPISODES)}
    q_tables = {"SARSA": [], "Q-learning": []}

    for run in range(SEED_RUNS):
        for algorithm in ("SARSA", "Q-learning"):
            q, steps = train_control(env, algorithm, run)
            curves[algorithm] += steps / SEED_RUNS
            q_tables[algorithm].append(q)

    plt.figure(figsize=(9, 5))
    for algorithm, curve in curves.items():
        smoothed = np.convolve(curve, np.ones(10) / 10, mode="valid")
        plt.plot(np.arange(10, CONTROL_EPISODES + 1), smoothed, label=algorithm)
    plt.xlabel("Episode")
    plt.ylabel("Steps per episode (10-episode moving average)")
    plt.title(f"SARSA vs Q-learning: Average of {SEED_RUNS} Runs (eps={EPSILON}, alpha={ALPHA})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/sarsa_q_learning.png", dpi=160)
    plt.close()

    for algorithm in ("SARSA", "Q-learning"):
        mean_q = np.mean(q_tables[algorithm], axis=0)
        filename = "sarsa_policy.png" if algorithm == "SARSA" else "q_learning_policy.png"
        save_policy(env, mean_q, f"Greedy policy learned by {algorithm}", f"{OUT}/{filename}")

    with open(f"{OUT}/control_learning_curves.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "SARSA", "Q-learning"])
        for ep in range(CONTROL_EPISODES):
            writer.writerow([ep + 1, curves["SARSA"][ep], curves["Q-learning"][ep]])

    print("Part 2 complete: learning curve and policies saved.")
    for algorithm in curves:
        print(f"  {algorithm}: mean steps over last 100 episodes = {curves[algorithm][-100:].mean():.2f}")


# ============================================================
# 3. VALUE FUNCTION APPROXIMATION
# Linear: V(s) = w . features(s). 3 weights shared by 8 trajectory states.
# Same fixed trajectory every episode, same initial weights, no weight reset.
# True values = discounted returns of the (deterministic) trajectory.
# ============================================================

def features(state):
    r, c = state
    return np.array([1.0, r / (ROWS - 1), c / (COLS - 1)])


def fixed_trajectory(env):
    """Deterministic route: right along the top row, then down the right column."""
    state = env.reset()
    trajectory = []
    actions = [3] * (COLS - 1) + [1] * (ROWS - 1)

    for action in actions:
        ns, reward, done = env.step(action)
        trajectory.append((state, action, reward, ns, done))
        state = ns

    return trajectory


def run_approximation():
    env = GridWorld()
    trajectory = fixed_trajectory(env)
    traj_states = [t[0] for t in trajectory]

    # True value of each trajectory state = discounted return from that point
    G = 0.0
    true_v = {}
    for s, a, reward, ns, done in reversed(trajectory):
        G = reward + GAMMA * G
        true_v[s] = G
    reference = np.array([true_v[s] for s in traj_states])

    # Identical initial parameters
    w_mc = np.zeros(len(features(START)))
    w_td = w_mc.copy()

    checkpoints = [1, 10, 25, 50, 100]
    rows = []
    mc_mae, td_mae = [], []

    for ep in range(1, APPROX_EPISODES + 1):

        # MC: full-return targets, first visit per state (weights NOT reset between episodes)
        G = 0.0
        returns = []
        for s, a, reward, ns, done in reversed(trajectory):
            G = reward + GAMMA * G
            returns.append((s, G))
        returns.reverse()

        visited = set()
        for s, target in returns:
            if s in visited:
                continue
            visited.add(s)
            x = features(env.id_to_state[s])
            w_mc += ALPHA * (target - w_mc @ x) * x

        # TD(0): semi-gradient update after each transition
        for s, a, reward, ns, done in trajectory:
            x = features(env.id_to_state[s])
            target = reward
            if not done:
                target += GAMMA * (w_td @ features(env.id_to_state[ns]))
            w_td += ALPHA * (target - w_td @ x) * x

        values_mc = np.array([w_mc @ features(env.id_to_state[s]) for s in traj_states])
        values_td = np.array([w_td @ features(env.id_to_state[s]) for s in traj_states])

        mc_mae.append(np.mean(np.abs(values_mc - reference)))
        td_mae.append(np.mean(np.abs(values_td - reference)))

        if ep in checkpoints:
            rows.append([ep, mc_mae[-1], td_mae[-1], values_mc.copy(), values_td.copy()])

    # Table: estimates per trajectory state at checkpoints
    labels = [str(env.id_to_state[s]) for s in traj_states]
    print("\nPart 3: MC vs TD(0) with linear VFA")
    print("True values:", {l: round(float(v), 3) for l, v in zip(labels, reference)})
    header = f"{'Ep':>4} | {'Algo':<5} | " + " | ".join(f"{l:>7}" for l in labels) + " |   MAE"
    print(header)
    print("-" * len(header))
    for ep, m1, m2, v_mc, v_td in rows:
        print(f"{ep:>4} | {'MC':<5} | " + " | ".join(f"{x:7.3f}" for x in v_mc) + f" | {m1:6.3f}")
        print(f"{ep:>4} | {'TD':<5} | " + " | ".join(f"{x:7.3f}" for x in v_td) + f" | {m2:6.3f}")

    with open(f"{OUT}/mc_td_approximation_checkpoints.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Algo"] + labels + ["MAE"])
        writer.writerow(["-", "True"] + np.round(reference, 4).tolist() + [0])
        for ep, m1, m2, v_mc, v_td in rows:
            writer.writerow([ep, "MC"] + np.round(v_mc, 4).tolist() + [round(m1, 4)])
            writer.writerow([ep, "TD(0)"] + np.round(v_td, 4).tolist() + [round(m2, 4)])

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, APPROX_EPISODES + 1), mc_mae, label="Monte Carlo")
    plt.plot(range(1, APPROX_EPISODES + 1), td_mae, label="TD(0)")
    plt.xlabel("Episode")
    plt.ylabel("MAE vs true trajectory values")
    plt.title("MC vs TD(0) with Value Function Approximation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/mc_td_approximation_mae.png", dpi=160)
    plt.close()

    with open(f"{OUT}/mc_td_approximation_mae.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "MC_MAE", "TD_MAE"])
        for ep in range(APPROX_EPISODES):
            writer.writerow([ep + 1, mc_mae[ep], td_mae[ep]])

    print("\nExplanation:")
    print("MC's target is the actual return, which equals the true value here, so it moves the shared")
    print("weights straight toward the right answer and has the lower error in the first ~10 episodes.")
    print("TD(0)'s target is reward + gamma * its own estimate of the next state. Early on that")
    print("estimate is poor (weights start at 0), so the goal reward reaches earlier states only")
    print("gradually and TD starts with a larger error. Because all states share the same weights,")
    print("each update also shifts the other states' values, so TD's path is not monotone per state")
    print("(e.g. it can overshoot the true value slightly before settling). Once its estimates are")
    print("good, bootstrapped targets are consistent and TD catches up or edges ahead.")
    print("Part 3 complete: MAE plot and checkpoint table saved.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Starting Assignment 2 experiments...")
    print("Environment: 5x5 GridWorld with fixed walls.")
    print(f"Runs: {SEED_RUNS} | Prediction episodes: {PREDICTION_EPISODES}")
    print(f"Control episodes: {CONTROL_EPISODES} | Approximation episodes: {APPROX_EPISODES}")

    run_prediction()
    run_control()
    run_approximation()

    print(f"\nAll outputs are saved in the '{OUT}' directory.")