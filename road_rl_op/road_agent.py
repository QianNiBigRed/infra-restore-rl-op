"""
Module for road agent by epsilon-greedy Q learning
"""

from collections import defaultdict
import numpy as np
import gymnasium as gym


class RoadAgentQLearning:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.99,
    ):
        """
        Tabular Q-Learning agent for RoadEnv using ε-greedy exploration.

        Args:
            env: Instance of RoadEnv
            learning_rate: α in Q-update rule
            initial_epsilon: starting exploration rate ε
            epsilon_decay: amount to multiply ε per episode
            final_epsilon: lower bound on ε
            discount_factor: γ in Q-update rule
        """
        self.env = env
        # Q-table: state_key -> {action_tuple: q_value}
        self.q_values = defaultdict(lambda: defaultdict(float))

        self.lr = learning_rate
        self.gamma = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def get_action(self, state: tuple) -> tuple:
        """
        Both exploration and exploitation draw from env.feasible_actions() instead of env.action_space.sample() or over all actions.
        Return an epsilon-greedy action for the given state.

        With probability ε, samples a random action from env.action_space;
        otherwise, picks argmax_a Q(state,a).
        """
        feas = self.env.feasible_actions(np.array(state))

        if np.random.rand() < self.epsilon:
            # explore uniformly among feasible
            return tuple(random.choice(feas).tolist())
        else:
            # exploit: pick feasible action with max Q
            # if all zeros, then return the first action in feas
            best = max(feas, key=lambda a: self.q_values[state][tuple(a.tolist())])
            return tuple(best.tolist())

    def update(
            self,
            state: tuple,
            action: tuple,
            reward: float,
            next_state: tuple,
            terminated: bool,
    ):
        """
        Perform Q-learning update for a single transition, but only for feasible actions.
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
        """

        # standard Q-learning update
        q_sa = self.q_values[state][action]

        if terminated:
            target = reward
        else:
            next_max = (
                max(self.q_values[next_state].values())
                if self.q_values[next_state]
                else 0.0)  # when self.q_values[next_state] haven't had any feasible action yet, set to default value 0.0
            target = reward + self.gamma * next_max

        # TD update
        self.q_values[state][action] = q_sa + self.lr * (target - q_sa)

    def decay_epsilon(self):
        """Decay exploration rate ε, but not below final_epsilon."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
