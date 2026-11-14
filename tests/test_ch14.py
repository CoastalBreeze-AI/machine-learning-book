"""Tests for Chapter 14: tinyml/rl.py."""
import unittest

import numpy as np

from tinyml.rl import (GridWorld, QLearningAgent, policy_evaluation,
                       policy_iteration, q_from_v, reinforce, render,
                       softmax_policy, train_td, value_iteration)

CORRIDOR = ("S..G",)


class TestGridWorld(unittest.TestCase):
    def test_transition_probabilities_sum_to_one(self):
        env = GridWorld(slip=0.2)
        for s in range(env.n_states):
            for a in range(env.n_actions):
                total = sum(p for p, *_ in env.transitions(s, a))
                self.assertAlmostEqual(total, 1.0)

    def test_walls_and_edges_block_movement(self):
        env = GridWorld()
        s = env.reset()
        self.assertEqual(s, 0)
        self.assertEqual(env.step(0)[0], 0)          # up, off the grid
        self.assertEqual(env.step(3)[0], 0)          # left, off the grid
        env.reset()
        env.step(2)                                  # down to (1, 0)
        self.assertEqual(env.step(1)[0], env.n_cols)  # right is a wall

    def test_goal_ends_episode_with_reward(self):
        env = GridWorld(CORRIDOR, step_reward=-0.1)
        env.reset()
        self.assertEqual(env.step(1), (1, -0.1, False))
        env.step(1)
        self.assertEqual(env.step(1), (3, 1.0, True))

    def test_slip_frequency(self):
        env = GridWorld(("...", ".S.", "..."), slip=0.4, seed=1)
        moved_up = 0
        for _ in range(4000):
            env.reset()
            moved_up += env.step(1)[0] == 1          # chose right, slipped up
        self.assertAlmostEqual(moved_up / 4000, 0.2, delta=0.02)


class TestDynamicProgramming(unittest.TestCase):
    def test_corridor_known_values(self):
        env = GridWorld(CORRIDOR, step_reward=0.0)
        V, policy, _ = value_iteration(env, gamma=0.9)
        np.testing.assert_allclose(V[:3], [0.81, 0.9, 1.0], atol=1e-6)
        self.assertTrue(np.all(policy[:3] == 1))

    def test_value_function_satisfies_bellman_optimality(self):
        env = GridWorld(slip=0.2)
        V, _, _ = value_iteration(env, gamma=0.95, theta=1e-12)
        np.testing.assert_allclose(q_from_v(env, V, 0.95).max(axis=1), V,
                                   atol=1e-9)

    def test_policy_iteration_matches_value_iteration(self):
        env = GridWorld(slip=0.2)
        V1, _, _ = value_iteration(env, gamma=0.95, theta=1e-12)
        V2, policy, _ = policy_iteration(env, gamma=0.95)
        np.testing.assert_allclose(V1, V2, atol=1e-8)
        np.testing.assert_allclose(policy_evaluation(env, policy, 0.95), V2)

    def test_render_shape(self):
        env = GridWorld()
        V, policy, _ = value_iteration(env)
        lines = render(env, values=V).splitlines()
        self.assertEqual(len(lines), env.n_rows)
        self.assertIn("#", lines[0])
        self.assertIn("G", render(env, policy=policy))


class TestLearning(unittest.TestCase):
    def test_q_learning_update_rule(self):
        agent = QLearningAgent(3, 2, alpha=0.5, gamma=0.9)
        agent.Q[1] = [2.0, 4.0]
        agent.update(0, 0, 1.0, 1, False)
        self.assertAlmostEqual(agent.Q[0, 0], 0.5 * (1.0 + 0.9 * 4.0))

    def test_sarsa_uses_next_action(self):
        agent = QLearningAgent(3, 2, alpha=0.5, gamma=0.9, method="sarsa")
        agent.Q[1] = [2.0, 4.0]
        agent.update(0, 0, 1.0, 1, False, a2=0)
        self.assertAlmostEqual(agent.Q[0, 0], 0.5 * (1.0 + 0.9 * 2.0))

    def test_q_learning_solves_deterministic_grid(self):
        env = GridWorld(seed=0)
        _, optimal, _ = value_iteration(env)
        agent = QLearningAgent(env.n_states, 4, alpha=0.5, epsilon=0.2, seed=0)
        train_td(env, agent, n_episodes=400)
        V_star = policy_evaluation(env, optimal)
        V_learned = policy_evaluation(env, agent.greedy_policy())
        self.assertAlmostEqual(V_learned[env.start], V_star[env.start], places=6)

    def test_softmax_policy_is_distribution(self):
        theta = np.array([[1000.0, 0.0, -5.0, 2.0]])
        p = softmax_policy(theta, 0)
        self.assertAlmostEqual(p.sum(), 1.0)
        self.assertTrue(np.all(p >= 0))

    def test_reinforce_learns_corridor(self):
        env = GridWorld(CORRIDOR, seed=0)
        theta, returns = reinforce(env, n_episodes=300, lr=0.2, seed=0)
        self.assertTrue(np.all(theta[:3].argmax(axis=1) == 1))
        self.assertGreater(returns[-50:].mean(), returns[:50].mean())


if __name__ == "__main__":
    unittest.main()
