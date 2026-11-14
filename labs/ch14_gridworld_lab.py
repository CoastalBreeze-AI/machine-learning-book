"""Chapter 14 lab: Listings 14.7, 14.8, 14.9, 14.10.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch14_gridworld_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 14.7 ----
from tinyml.rl import GridWorld, value_iteration, policy_iteration, render

for slip in (0.0, 0.2):
    env = GridWorld(slip=slip)
    V, policy, sweeps = value_iteration(env, gamma=0.95)
    print(f"slip={slip}: value iteration converged in {sweeps} sweeps")
    print(render(env, values=V))
    print(render(env, policy=policy))
    V_pi, policy_pi, iterations = policy_iteration(env, gamma=0.95)
    same = (policy_pi == policy).mean()
    print(f"policy iteration: {iterations} iterations, "
          f"max |V_vi - V_pi| = {abs(V - V_pi).max():.1e}, "
          f"same action in {same:.0%} of cells\n")


# ---- Listing 14.8 ----
import numpy as np
from tinyml.rl import (GridWorld, QLearningAgent, train_td, value_iteration,
                       policy_evaluation, render)

env = GridWorld(slip=0.2, seed=0)
V_star, optimal, _ = value_iteration(env, gamma=0.95)
agent = QLearningAgent(env.n_states, env.n_actions, alpha=0.1, gamma=0.95,
                       epsilon=0.1, seed=0)
returns = train_td(env, agent, n_episodes=2000)
for start in range(0, 2000, 400):
    print(f"episodes {start:4d}-{start + 399:4d}: "
          f"mean return {returns[start:start + 400].mean():6.3f}")

print("\nGreedy policy learned by Q-learning:")
print(render(env, policy=agent.greedy_policy()))
print("max_a Q(s, a):")
print(render(env, values=agent.Q.max(axis=1)))

open_cells = [s for s in range(env.n_states) if env.cell(s) in "S."]
agree = np.mean([agent.greedy_policy()[s] == optimal[s] for s in open_cells])
V_learned = policy_evaluation(env, agent.greedy_policy(), gamma=0.95)
print(f"\nAgrees with the optimal action in {agree:.0%} of open cells")
print(f"Value of start state: optimal {V_star[env.start]:.4f}, "
      f"learned greedy policy {V_learned[env.start]:.4f}")


# ---- Listing 14.9 ----
import numpy as np
from tinyml.rl import GridWorld, QLearningAgent, train_td, render

CLIFF = ("............",
         "............",
         "............",
         "SPPPPPPPPPPG")


def greedy_return(agent):
    """Return of one episode that follows the greedy policy, no exploration."""
    env = GridWorld(CLIFF, step_reward=-1, goal_reward=-1, pit_reward=-100)
    s, total = env.reset(), 0
    for _ in range(100):
        s, reward, done = env.step(agent.greedy_policy()[s])
        total += reward
        if done:
            break
    return total


for method in ("q-learning", "sarsa"):
    online, greedy = [], []
    for seed in range(10):
        env = GridWorld(CLIFF, step_reward=-1, goal_reward=-1,
                        pit_reward=-100, seed=seed)
        agent = QLearningAgent(env.n_states, env.n_actions, alpha=0.1,
                               gamma=1.0, epsilon=0.1, method=method, seed=seed)
        returns = train_td(env, agent, n_episodes=1000)
        online.append(returns[-200:].mean())
        greedy.append(greedy_return(agent))
        if seed == 0:
            print(f"{method}, seed 0, greedy policy:")
            print(render(env, policy=agent.greedy_policy()))
    print(f"{method}: mean return while exploring (last 200 episodes) "
          f"{np.mean(online):.1f}; greedy returns {greedy}\n")


# ---- Listing 14.10 ----
import numpy as np
from tinyml.rl import GridWorld, reinforce, policy_evaluation, render

for baseline in (False, True):
    curves, start_values = [], []
    for seed in range(5):
        env = GridWorld(slip=0.2, seed=seed)
        theta, returns = reinforce(env, n_episodes=2000, lr=0.1,
                                   baseline=baseline, seed=seed)
        curves.append(returns.reshape(4, 500).mean(axis=1))
        greedy = theta.argmax(axis=1)
        start_values.append(policy_evaluation(env, greedy)[env.start])
    blocks = "  ".join(f"{x:6.3f}" for x in np.mean(curves, axis=0))
    print(f"baseline={baseline!s:5}  mean return per 500 episodes: {blocks}")
    print(f"   spread across seeds, episodes 0-499: "
          f"std {np.std([c[0] for c in curves]):.3f}; "
          f"greedy start value {np.mean(start_values):.4f}")
print("\nGreedy policy from the last run (baseline, seed 4):")
print(render(env, policy=theta.argmax(axis=1)))
