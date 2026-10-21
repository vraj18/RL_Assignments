import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# REINFORCEMENT LEARNING ASSIGNMENT
# CSL 412 - Markov Decision Processes
# 4x4 Frozen Lake
#
# Environment:
# States: 0 ... 15
# Actions: Left, Down, Right, Up
# Holes: 5, 7, 12
# Goal: 15
# Intended transition probability: 0.25
# Stay probability: 0.75
# Living reward: -0.01
# Goal reward: +1
# Hole reward: -1
# Gamma: 0.9
# ============================================================


# ============================================================
# PART 1: ENVIRONMENT SETUP
# ============================================================

GRID_SIZE = 4
NUM_STATES = 16

# Action numbering
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

actions = [LEFT, DOWN, RIGHT, UP]

action_names = {
    LEFT: "Left",
    DOWN: "Down",
    RIGHT: "Right",
    UP: "Up"
}

# Movement corresponding to each action
action_delta = {
    LEFT:  (0, -1),
    DOWN:  (1, 0),
    RIGHT: (0, 1),
    UP:    (-1, 0)
}

# Terminal states
HOLES = {5, 7, 12}
GOAL = 15
TERMINAL_STATES = HOLES | {GOAL}

# Rewards
LIVING_REWARD = -0.01
HOLE_REWARD = -1.0
GOAL_REWARD = 1.0

# Discount factor
GAMMA = 0.9

# Probability of intended movement
P_INTENDED = 0.25

# Probability of staying in the same cell
P_STAY = 0.75

# Convergence tolerance
TOLERANCE = 1e-10


# ============================================================
# DISPLAY GRID
# ============================================================

def print_grid(policy=None):
    """
    Print the Frozen Lake grid.

    If a policy is supplied, print the corresponding action
    arrows instead of F for normal states.
    """

    arrows = {
        LEFT: "←",
        DOWN: "↓",
        RIGHT: "→",
        UP: "↑"
    }

    for s in range(NUM_STATES):

        if s in HOLES:
            symbol = "H"

        elif s == GOAL:
            symbol = "G"

        elif s == 0:
            symbol = "S"

        elif policy is not None:
            symbol = arrows[policy[s]]

        else:
            symbol = "F"

        print(f"{symbol:^5}", end="")

        if (s + 1) % GRID_SIZE == 0:
            print()

    print()


print("INITIAL GRID")
print_grid()


# ============================================================
# PART 1: TRANSITION FUNCTION
# ============================================================

def transitions(s, action):
    """
    Return possible transitions from state s
    when action is taken.

    Format:
        (probability, next_state, reward)

    According to the PPT:

        P(intended) = 0.25
        P(stay)     = 0.75
    """

    # Terminal states are absorbing.
    if s in TERMINAL_STATES:
        return [(1.0, s, 0.0)]

    row = s // GRID_SIZE
    col = s % GRID_SIZE

    dr, dc = action_delta[action]

    new_row = row + dr
    new_col = col + dc

    # Boundary collision:
    # blocked movement remains in the same state.
    if (
        new_row < 0
        or new_row >= GRID_SIZE
        or new_col < 0
        or new_col >= GRID_SIZE
    ):
        intended_state = s
    else:
        intended_state = new_row * GRID_SIZE + new_col

    # Reward associated with intended transition
    if intended_state == GOAL:
        intended_reward = GOAL_REWARD

    elif intended_state in HOLES:
        intended_reward = HOLE_REWARD

    else:
        intended_reward = LIVING_REWARD

    # 75% probability of staying.
    # Staying in a non-terminal state receives living reward.
    stay_reward = LIVING_REWARD

    return [
        (P_INTENDED, intended_state, intended_reward),
        (P_STAY, s, stay_reward)
    ]


# ============================================================
# SHOW SOME EXAMPLE TRANSITIONS
# ============================================================

print("Example transitions from state 6:")

for action in actions:

    print(f"\nAction: {action_names[action]}")

    for probability, next_state, reward in transitions(6, action):

        print(
            f"Probability = {probability:.2f}, "
            f"Next State = {next_state}, "
            f"Reward = {reward:.2f}"
        )


# ============================================================
# PART 2: UNIFORM RANDOM POLICY
# ============================================================

# π(a|s) = 0.25 for all actions
policy_random = np.ones(
    (NUM_STATES, 4)
) / 4.0


# ============================================================
# PART 2: POLICY EVALUATION
# ============================================================

def policy_evaluation(
    policy,
    gamma=GAMMA,
    tolerance=TOLERANCE
):
    """
    Iterative policy evaluation.

    V_0(s) = 0

    Repeatedly apply the Bellman expectation equation:

    V(s) = sum_a π(a|s)
           sum_s' P(s'|s,a)
           [R + gamma V(s')]
    """

    # Initial value function
    V = np.zeros(NUM_STATES)

    # Store convergence errors
    errors = []

    for iteration in range(1, 10000):

        # Synchronized update:
        # V_new is calculated entirely from old V.
        V_new = np.zeros(NUM_STATES)

        for s in range(NUM_STATES):

            # Terminal states have value 0
            if s in TERMINAL_STATES:
                V_new[s] = 0.0
                continue

            # Expected value over all actions
            for action in actions:

                action_probability = policy[s, action]

                # Expected value over transitions
                for probability, next_state, reward in transitions(
                    s, action
                ):

                    V_new[s] += (
                        action_probability
                        * probability
                        * (
                            reward
                            + gamma * V[next_state]
                        )
                    )

        # Infinity norm:
        # maximum absolute change over all states
        error = np.max(
            np.abs(V_new - V)
        )

        errors.append(error)

        # Update
        V = V_new

        # Stop when convergence criterion is satisfied
        if error < tolerance:
            break

    return V, errors, iteration


# Run policy evaluation

V_random, evaluation_errors, evaluation_iterations = (
    policy_evaluation(policy_random)
)


# ============================================================
# PRINT POLICY EVALUATION RESULT
# ============================================================

print("\n" + "=" * 60)
print("PART 2: POLICY EVALUATION")
print("=" * 60)

print(
    "Number of iterations:",
    evaluation_iterations
)

print("\nV^π:")

print(
    np.array2string(
        V_random.reshape(4, 4),
        formatter={
            "float_kind":
            lambda x: f"{x: .4f}"
        }
    )
)


# ============================================================
# PLOT CONVERGENCE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(evaluation_errors) + 1),
    evaluation_errors
)

plt.yscale("log")

plt.xlabel("Iteration k")
plt.ylabel(r"$||V_{k+1} - V_k||_\infty$")

plt.title(
    "Policy Evaluation Convergence"
)

plt.grid(True)

plt.show()


# ============================================================
# PART 2: MANUAL CALCULATION CHECK
# STATE 6
# ============================================================

print("\n" + "=" * 60)
print("PART 2: HAND CALCULATION CHECK FOR STATE 6")
print("=" * 60)


def one_policy_evaluation_iteration(
    V,
    policy,
    gamma=GAMMA
):

    V_new = np.zeros(NUM_STATES)

    for s in range(NUM_STATES):

        if s in TERMINAL_STATES:
            continue

        for action in actions:

            for probability, next_state, reward in transitions(
                s, action
            ):

                V_new[s] += (
                    policy[s, action]
                    * probability
                    * (
                        reward
                        + gamma * V[next_state]
                    )
                )

    return V_new


# V0
V0 = np.zeros(NUM_STATES)

# V1
V1 = one_policy_evaluation_iteration(
    V0,
    policy_random
)

# V2
V2 = one_policy_evaluation_iteration(
    V1,
    policy_random
)

print(
    f"Python V1(6) = {V1[6]:.8f}"
)

print(
    f"Python V2(6) = {V2[6]:.8f}"
)

print(
    f"Expected V1(6) = -0.13375000"
)

print(
    f"Expected V2(6) = -0.22515625"
)

print(
    f"Delta V1 = {abs(V1[6] - (-0.13375)):.8f}"
)

print(
    f"Delta V2 = {abs(V2[6] - (-0.22515625)):.8f}"
)


# ============================================================
# PART 2: Q-VALUES
# ============================================================

def q_values(V, gamma=GAMMA):
    """
    Calculate Q(s,a) for every state-action pair.

    Q(s,a) =
        sum_s' P(s'|s,a)
        [R(s,a,s') + gamma V(s')]
    """

    Q = np.zeros(
        (NUM_STATES, 4)
    )

    for s in range(NUM_STATES):

        # Terminal states have no actions
        if s in TERMINAL_STATES:
            continue

        for action in actions:

            for probability, next_state, reward in transitions(
                s, action
            ):

                Q[s, action] += (
                    probability
                    * (
                        reward
                        + gamma * V[next_state]
                    )
                )

    return Q


Q_random = q_values(V_random)


print("\n" + "=" * 60)
print("PART 2: Q-VALUES")
print("=" * 60)

print(
    f"{'State':<8}"
    f"{'Left':>12}"
    f"{'Down':>12}"
    f"{'Right':>12}"
    f"{'Up':>12}"
    f"{'Greedy':>15}"
)

for s in range(NUM_STATES):

    if s in TERMINAL_STATES:

        print(
            f"{s:<8}"
            f"{'Terminal':>12}"
        )

        continue

    best_action = np.argmax(
        Q_random[s]
    )

    print(
        f"{s:<8}"
        f"{Q_random[s, LEFT]:>12.3f}"
        f"{Q_random[s, DOWN]:>12.3f}"
        f"{Q_random[s, RIGHT]:>12.3f}"
        f"{Q_random[s, UP]:>12.3f}"
        f"{action_names[best_action]:>15}"
    )


# ============================================================
# PART 3: POLICY ITERATION
# ============================================================

def improve_policy(V):
    """
    Greedy policy improvement.

    π_new(s) = argmax_a Q(s,a)
    """

    Q = q_values(V)

    new_policy = np.zeros(
        NUM_STATES,
        dtype=int
    )

    for s in range(NUM_STATES):

        if s in TERMINAL_STATES:
            new_policy[s] = -1

        else:
            new_policy[s] = np.argmax(
                Q[s]
            )

    return new_policy, Q


# Start with uniform random policy
current_policy = policy_random.copy()

policy_history = []

print("\n" + "=" * 60)
print("PART 3: POLICY ITERATION")
print("=" * 60)


for policy_iteration in range(100):

    # Evaluate current policy
    V_current, errors, eval_iters = (
        policy_evaluation(
            current_policy
        )
    )

    # Improve policy
    improved_policy, Q_current = (
        improve_policy(
            V_current
        )
    )

    # Save policy
    policy_history.append(
        improved_policy.copy()
    )

    print(
        f"\nPolicy improvement {policy_iteration + 1}"
    )

    # Check whether policy changed
    if policy_iteration == 0:

        old_deterministic_policy = np.full(
            NUM_STATES,
            -1
        )

    else:

        old_deterministic_policy = (
            policy_history[-2]
        )

    changed = False

    for s in range(NUM_STATES):

        if s in TERMINAL_STATES:
            continue

        if (
            improved_policy[s]
            != old_deterministic_policy[s]
        ):
            changed = True

    print(
        "Policy changed:",
        changed
    )

    # New policy becomes deterministic
    new_policy_matrix = np.zeros(
        (NUM_STATES, 4)
    )

    for s in range(NUM_STATES):

        if s in TERMINAL_STATES:
            continue

        new_policy_matrix[
            s,
            improved_policy[s]
        ] = 1.0

    # If policy did not change, stable
    if not changed:

        optimal_policy = improved_policy
        break

    current_policy = new_policy_matrix


# ============================================================
# DISPLAY POLICY ITERATION RESULT
# ============================================================

print("\nFinal Optimal Policy from Policy Iteration:")

print_grid(optimal_policy)

print(
    "Policy improvement steps:",
    policy_iteration + 1
)


# ============================================================
# PART 3: VALUE ITERATION
# ============================================================

def value_iteration(
    gamma=GAMMA,
    tolerance=TOLERANCE
):
    """
    Value iteration:

    V_{k+1}(s) =
        max_a sum_s'
        P(s'|s,a)
        [R + gamma V_k(s')]
    """

    V = np.zeros(NUM_STATES)

    errors = []

    for iteration in range(1, 10000):

        V_new = np.zeros(NUM_STATES)

        for s in range(NUM_STATES):

            if s in TERMINAL_STATES:
                continue

            action_values = []

            for action in actions:

                action_value = 0.0

                for probability, next_state, reward in transitions(
                    s, action
                ):

                    action_value += (
                        probability
                        * (
                            reward
                            + gamma * V[next_state]
                        )
                    )

                action_values.append(
                    action_value
                )

            V_new[s] = max(
                action_values
            )

        error = np.max(
            np.abs(V_new - V)
        )

        errors.append(error)

        V = V_new

        if error < tolerance:
            break

    # Recover greedy policy
    Q = q_values(V, gamma)

    optimal_policy = np.full(
        NUM_STATES,
        -1,
        dtype=int
    )

    for s in range(NUM_STATES):

        if s not in TERMINAL_STATES:

            optimal_policy[s] = np.argmax(
                Q[s]
            )

    return V, optimal_policy, errors, iteration


V_star, vi_policy, vi_errors, vi_iterations = (
    value_iteration()
)


# ============================================================
# DISPLAY VALUE ITERATION RESULT
# ============================================================

print("\n" + "=" * 60)
print("PART 3: VALUE ITERATION")
print("=" * 60)

print(
    "Value iteration iterations:",
    vi_iterations
)

print("\nOptimal Value Function:")

print(
    np.array2string(
        V_star.reshape(4, 4),
        formatter={
            "float_kind":
            lambda x: f"{x: .4f}"
        }
    )
)

print("\nOptimal Policy from Value Iteration:")

print_grid(vi_policy)


# ============================================================
# COMPARE POLICY ITERATION AND VALUE ITERATION
# ============================================================

same_policy = True

for s in range(NUM_STATES):

    if s in TERMINAL_STATES:
        continue

    if optimal_policy[s] != vi_policy[s]:
        same_policy = False


print(
    "Policy Iteration == Value Iteration:",
    same_policy
)


# ============================================================
# VALUE ITERATION CONVERGENCE PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(vi_errors) + 1),
    vi_errors
)

plt.yscale("log")

plt.xlabel("Iteration k")

plt.ylabel(
    r"$||V_{k+1} - V_k||_\infty$"
)

plt.title(
    "Value Iteration Convergence"
)

plt.grid(True)

plt.show()


# ============================================================
# PART 4: EXACT LINEAR ALGEBRA
# ============================================================

def build_policy_matrices(
    policy
):
    """
    Construct P_pi and R_pi.

    P_pi[s,s'] =
        sum_a pi(a|s) P(s'|s,a)

    R_pi[s] =
        expected immediate reward under policy
    """

    P_pi = np.zeros(
        (NUM_STATES, NUM_STATES)
    )

    R_pi = np.zeros(
        NUM_STATES
    )

    for s in range(NUM_STATES):

        if s in TERMINAL_STATES:

            P_pi[s, s] = 1.0
            R_pi[s] = 0.0

            continue

        for action in actions:

            action_probability = (
                policy[s, action]
            )

            for probability, next_state, reward in transitions(
                s, action
            ):

                P_pi[
                    s,
                    next_state
                ] += (
                    action_probability
                    * probability
                )

                R_pi[s] += (
                    action_probability
                    * probability
                    * reward
                )

    return P_pi, R_pi


# Use the original uniform policy for Part 4
P_pi, R_pi = build_policy_matrices(
    policy_random
)


# Solve:
#
# (I - gamma P_pi)V = R_pi
#
# using np.linalg.solve()

A = (
    np.eye(NUM_STATES)
    - GAMMA * P_pi
)

V_exact = np.linalg.solve(
    A,
    R_pi
)


print("\n" + "=" * 60)
print("PART 4: EXACT LINEAR ALGEBRA")
print("=" * 60)

print("\nExact V^π:")

print(
    np.array2string(
        V_exact.reshape(4, 4),
        formatter={
            "float_kind":
            lambda x: f"{x: .4f}"
        }
    )
)


# Compare exact and iterative values

max_difference = np.max(
    np.abs(
        V_exact - V_random
    )
)

print(
    "\nmax |V_exact - V_iterative| =",
    max_difference
)


# ============================================================
# PART 5.1: GAMMA SENSITIVITY
# ============================================================

def run_gamma_experiment(gamma):

    V, policy, errors, iterations = (
        value_iteration(
            gamma=gamma,
            tolerance=TOLERANCE
        )
    )

    return V, policy, iterations


gammas = [
    0.0,
    0.5,
    0.9,
    0.999
]


print("\n" + "=" * 60)
print("PART 5.1: DISCOUNT FACTOR EXPERIMENT")
print("=" * 60)

for gamma in gammas:

    V_gamma, policy_gamma, iterations_gamma = (
        run_gamma_experiment(gamma)
    )

    print(
        f"gamma = {gamma:.3f}"
    )

    print(
        f"V*(0) = {V_gamma[0]:.6f}"
    )

    print(
        f"Iterations = {iterations_gamma}"
    )

    print()


# ============================================================
# PART 5.2: LIVING REWARD EXPERIMENT
# ============================================================

def transitions_with_living_reward(
    s,
    action,
    living_reward
):

    if s in TERMINAL_STATES:

        return [
            (1.0, s, 0.0)
        ]

    row = s // GRID_SIZE
    col = s % GRID_SIZE

    dr, dc = action_delta[action]

    new_row = row + dr
    new_col = col + dc

    # Boundary collision
    if (
        new_row < 0
        or new_row >= GRID_SIZE
        or new_col < 0
        or new_col >= GRID_SIZE
    ):

        intended_state = s

    else:

        intended_state = (
            new_row * GRID_SIZE
            + new_col
        )

    # Reward for intended movement
    if intended_state == GOAL:

        intended_reward = GOAL_REWARD

    elif intended_state in HOLES:

        intended_reward = HOLE_REWARD

    else:

        intended_reward = living_reward

    return [
        (
            P_INTENDED,
            intended_state,
            intended_reward
        ),
        (
            P_STAY,
            s,
            living_reward
        )
    ]


def value_iteration_with_living_reward(
    living_reward,
    gamma=GAMMA,
    tolerance=TOLERANCE
):

    V = np.zeros(NUM_STATES)

    for iteration in range(1, 10000):

        V_new = np.zeros(NUM_STATES)

        for s in range(NUM_STATES):

            if s in TERMINAL_STATES:
                continue

            action_values = []

            for action in actions:

                value = 0.0

                for probability, next_state, reward in (
                    transitions_with_living_reward(
                        s,
                        action,
                        living_reward
                    )
                ):

                    value += (
                        probability
                        * (
                            reward
                            + gamma
                            * V[next_state]
                        )
                    )

                action_values.append(value)

            V_new[s] = max(
                action_values
            )

        error = np.max(
            np.abs(V_new - V)
        )

        V = V_new

        if error < tolerance:
            break

    # Greedy policy
    policy = np.full(
        NUM_STATES,
        -1,
        dtype=int
    )

    for s in range(NUM_STATES):

        if s in TERMINAL_STATES:
            continue

        action_values = []

        for action in actions:

            value = 0.0

            for probability, next_state, reward in (
                transitions_with_living_reward(
                    s,
                    action,
                    living_reward
                )
            ):

                value += (
                    probability
                    * (
                        reward
                        + gamma
                        * V[next_state]
                    )
                )

            action_values.append(value)

        policy[s] = np.argmax(
            action_values
        )

    return V, policy


print("\n" + "=" * 60)
print("PART 5.2: LIVING REWARD EXPERIMENT")
print("=" * 60)


V_negative, policy_negative = (
    value_iteration_with_living_reward(
        -0.01
    )
)

V_zero, policy_zero = (
    value_iteration_with_living_reward(
        0.0
    )
)


print(
    "Living reward = -0.01"
)

print(
    "V*(0) =",
    V_negative[0]
)

print()

print(
    "Living reward = 0.00"
)

print(
    "V*(0) =",
    V_zero[0]
)


# Check whether policies are identical

policies_same = True

for s in range(NUM_STATES):

    if s in TERMINAL_STATES:
        continue

    if policy_negative[s] != policy_zero[s]:

        policies_same = False


print()

print(
    "Optimal policies identical:",
    policies_same
)


# ============================================================
# PART 5.3: MODEL-FREE Q-LEARNING DEMONSTRATION
# ============================================================

def q_learning_demo(
    episodes=5000,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1
):

    """
    Demonstrates how the code changes when P and R
    are unknown.

    Instead of using transitions(),
    the agent samples one transition from
    the environment and updates Q directly.

    This is a simple Q-learning demonstration.
    """

    Q = np.zeros(
        (NUM_STATES, 4)
    )

    for episode in range(episodes):

        # Start at state 0
        state = 0

        for step in range(100):

            if state in TERMINAL_STATES:
                break

            # ε-greedy action selection

            if np.random.random() < epsilon:

                action = np.random.choice(
                    actions
                )

            else:

                action = np.argmax(
                    Q[state]
                )

            # Sample environment transition
            possible = transitions(
                state,
                action
            )

            probabilities = [
                x[0] for x in possible
            ]

            choice = np.random.choice(
                len(possible),
                p=probabilities
            )

            probability, next_state, reward = (
                possible[choice]
            )

            # Q-learning update

            if next_state in TERMINAL_STATES:

                target = reward

            else:

                target = (
                    reward
                    + gamma
                    * np.max(
                        Q[next_state]
                    )
                )

            Q[state, action] += (
                alpha
                * (
                    target
                    - Q[state, action]
                )
            )

            state = next_state

    learned_policy = np.full(
        NUM_STATES,
        -1,
        dtype=int
    )

    for s in range(NUM_STATES):

        if s not in TERMINAL_STATES:

            learned_policy[s] = np.argmax(
                Q[s]
            )

    return Q, learned_policy


Q_learned, learned_policy = (
    q_learning_demo()
)


print("\n" + "=" * 60)
print("PART 5.3: MODEL-FREE Q-LEARNING")
print("=" * 60)

print(
    "Learned Q-table shape:",
    Q_learned.shape
)

print("\nLearned policy:")

print_grid(
    learned_policy
)