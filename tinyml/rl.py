"""Reinforcement learning on a small grid world (Chapter 14)."""
import numpy as np

DEFAULT_LAYOUT = ("S...#...",
                  ".#..#.#.",
                  ".#....P.",
                  "...#...G")


class GridWorld:
    """A grid with walls (#), a start (S), a goal (G), and pits (P).

    Moving into a wall or off the grid leaves the agent where it is. With
    probability `slip`, the agent moves at right angles to the chosen
    direction instead (each side half the time). Reaching G or P ends the
    episode. States are numbered row * n_cols + col.
    """
    MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))   # up, right, down, left
    ARROWS = "^>v<"

    def __init__(self, layout=DEFAULT_LAYOUT, slip=0.0, step_reward=-0.04,
                 goal_reward=1.0, pit_reward=-1.0, seed=0):
        self.grid = [list(row) for row in layout]
        self.n_rows, self.n_cols = len(layout), len(layout[0])
        self.n_states, self.n_actions = self.n_rows * self.n_cols, 4
        self.slip, self.step_reward = slip, step_reward
        self.rewards = {"G": goal_reward, "P": pit_reward}
        self.start = next(r * self.n_cols + c for r, row in enumerate(layout)
                          for c, ch in enumerate(row) if ch == "S")
        self.rng = np.random.default_rng(seed)
        self.state = self.start

    def cell(self, s):
        return self.grid[s // self.n_cols][s % self.n_cols]

    def is_terminal(self, s):
        return self.cell(s) in "GP"

    def _move(self, s, a):
        r, c = divmod(s, self.n_cols)
        dr, dc = self.MOVES[a]
        r2, c2 = r + dr, c + dc
        if 0 <= r2 < self.n_rows and 0 <= c2 < self.n_cols \
                and self.grid[r2][c2] != "#":
            return r2 * self.n_cols + c2
        return s

    def transitions(self, s, a):
        """The model: a list of (probability, next_state, reward, done)."""
        if self.is_terminal(s) or self.cell(s) == "#":
            return [(1.0, s, 0.0, True)]
        outcomes = [(1.0 - self.slip, a)]
        if self.slip > 0:
            outcomes += [(self.slip / 2, (a + 1) % 4), (self.slip / 2, (a + 3) % 4)]
        result = []
        for p, actual in outcomes:
            s2 = self._move(s, actual)
            reward = self.rewards.get(self.cell(s2), self.step_reward)
            result.append((p, s2, reward, self.is_terminal(s2)))
        return result

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, a):
        """Samples one transition. Returns (next_state, reward, done)."""
        options = self.transitions(self.state, a)
        i = self.rng.choice(len(options), p=[o[0] for o in options])
        _, self.state, reward, done = options[i]
        return self.state, reward, done


def render(env, values=None, policy=None):
    """Draws the grid as text, with a value or an arrow in each open cell."""
    lines = []
    for r in range(env.n_rows):
        cells = []
        for c in range(env.n_cols):
            s = r * env.n_cols + c
            ch = env.cell(s)
            if ch in "#GP":
                cells.append(f"{ch:>6}")
            elif values is not None:
                cells.append(f"{values[s]:6.2f}")
            else:
                cells.append(f"{env.ARROWS[policy[s]]:>6}")
        lines.append("".join(cells))
    return "\n".join(lines)


def q_from_v(env, V, gamma):
    """One-step lookahead: Q[s, a] = sum over outcomes of p * (r + gamma V[s'])."""
    Q = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        for a in range(env.n_actions):
            for p, s2, reward, done in env.transitions(s, a):
                Q[s, a] += p * (reward + (0.0 if done else gamma * V[s2]))
    return Q


def value_iteration(env, gamma=0.95, theta=1e-8, max_sweeps=10_000):
    """Applies the Bellman optimality backup until values stop changing.

    Returns (V, policy, sweeps), where policy[s] is the greedy action.
    """
    V = np.zeros(env.n_states)
    for sweep in range(1, max_sweeps + 1):
        Q = q_from_v(env, V, gamma)
        V_new = Q.max(axis=1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            break
    return V, q_from_v(env, V, gamma).argmax(axis=1), sweep


def policy_evaluation(env, policy, gamma=0.95):
    """Solves the linear Bellman equations V = r_pi + gamma P_pi V exactly."""
    n = env.n_states
    P, r = np.zeros((n, n)), np.zeros(n)
    for s in range(n):
        for p, s2, reward, done in env.transitions(s, policy[s]):
            r[s] += p * reward
            if not done:
                P[s, s2] += p
    return np.linalg.solve(np.eye(n) - gamma * P, r)


def policy_iteration(env, gamma=0.95, max_iter=100):
    """Alternates exact evaluation and greedy improvement until stable.

    Returns (V, policy, iterations).
    """
    policy = np.zeros(env.n_states, dtype=int)
    for it in range(1, max_iter + 1):
        V = policy_evaluation(env, policy, gamma)
        Q = q_from_v(env, V, gamma)
        # Keep the current action unless another is strictly better, so
        # ties between equally good actions cannot make the loop cycle.
        better = Q.max(axis=1) > Q[np.arange(env.n_states), policy] + 1e-12
        if not better.any():
            break
        policy = np.where(better, Q.argmax(axis=1), policy)
    return V, policy, it


class QLearningAgent:
    """Tabular Q-learning, or SARSA with method="sarsa", with an
    epsilon-greedy behavior policy.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=0.1, method="q-learning", seed=0):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.method = method
        self.Q = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng(seed)

    def act(self, s):
        """Explores with probability epsilon; otherwise acts greedily,
        breaking ties at random."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.Q.shape[1]))
        best = np.flatnonzero(self.Q[s] == self.Q[s].max())
        return int(self.rng.choice(best))

    def update(self, s, a, reward, s2, done, a2=None):
        """One temporal-difference update of Q[s, a]."""
        if done:
            target = reward
        elif self.method == "sarsa":
            target = reward + self.gamma * self.Q[s2, a2]    # on-policy
        else:
            target = reward + self.gamma * self.Q[s2].max()  # off-policy
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def greedy_policy(self):
        return self.Q.argmax(axis=1)


def train_td(env, agent, n_episodes=500, max_steps=200):
    """Runs episodes, updating the agent after every step. Returns the
    undiscounted return of each episode."""
    returns = []
    for _ in range(n_episodes):
        s, total = env.reset(), 0.0
        a = agent.act(s)
        for _ in range(max_steps):
            s2, reward, done = env.step(a)
            a2 = None if done else agent.act(s2)
            agent.update(s, a, reward, s2, done, a2)
            total += reward
            if done:
                break
            s, a = s2, a2
        returns.append(total)
    return np.array(returns)


def softmax_policy(theta, s):
    """Action probabilities pi(. | s) for a tabular softmax policy."""
    z = theta[s] - theta[s].max()
    return np.exp(z) / np.exp(z).sum()


def reinforce(env, n_episodes=2000, lr=0.1, lr_baseline=0.1, gamma=0.95,
              baseline=True, max_steps=100, seed=0):
    """REINFORCE (Williams, 1992) with a tabular softmax policy.

    theta[s, a] are action preferences. With baseline=True, a learned
    state value b[s] is subtracted from each return to reduce variance.
    Returns (theta, undiscounted return of each episode).
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n_states, env.n_actions))
    b = np.zeros(env.n_states)
    returns = []
    for _ in range(n_episodes):
        # 1. Generate an episode by following the current policy.
        s, done, episode = env.reset(), False, []
        for _ in range(max_steps):
            a = rng.choice(env.n_actions, p=softmax_policy(theta, s))
            s2, reward, done = env.step(a)
            episode.append((s, a, reward))
            s = s2
            if done:
                break
        returns.append(sum(r for _, _, r in episode))

        # 2. Walk backward, accumulating the return G_t from each step.
        G = 0.0
        for t in reversed(range(len(episode))):
            s, a, reward = episode[t]
            G = reward + gamma * G
            advantage = G - b[s] if baseline else G
            if baseline:
                b[s] += lr_baseline * (G - b[s])
            grad_log_pi = -softmax_policy(theta, s)
            grad_log_pi[a] += 1.0              # d/dtheta[s] of log pi(a|s)
            theta[s] += lr * (gamma ** t) * advantage * grad_log_pi
    return theta, np.array(returns)
