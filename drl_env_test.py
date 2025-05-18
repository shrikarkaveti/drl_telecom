import sys
import random
import numpy as np
from os import path

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import drl_agent_test as dat

register(
    id='drl-agent-test',
    entry_point='drl_env_test:DRLAgentEnv',
)

# ------------------------- One-Based Action Wrapper -------------------------
class OneBasedActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Discrete), \
            "This wrapper only supports Discrete action spaces."

        self.action_space = spaces.Discrete(env.action_space.n)
        self.one_based_action_space = spaces.Discrete(env.action_space.n + 1)

    def action(self, action):
        if action < 1 or action > self.env.action_space.n:
            raise ValueError(f"Action must be between 1 and {self.env.action_space.n}")
        return action

    # def reverse_action(self, action):
    #     return action + 1

# ------------------------- DRLAgentEnv -------------------------
class DRLAgentEnv(gym.Env):
    def __init__(self, n_users=4, n_rbs=5, n_subslots=7, render_mode=None):
        self.n_users = n_users
        self.n_rbs = n_rbs
        self.n_subslots = n_subslots
        self.render_mode = render_mode

        self.seed_value = None
        self.drl_agent = dat.DrlAgent(n_RBs = n_rbs, n_users = n_users, n_subslots = n_subslots)

        self.action_space = spaces.Discrete(self.n_subslots)
        self.observation_space = spaces.Box(
            low=0,
            high=self.n_subslots,
            shape=(self.n_users * 2, self.n_rbs),
            dtype=np.int64
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.seed_value = seed if seed is not None else random.randint(0, 1e6)

        self.np_random, _ = gym.utils.seeding.np_random(self.seed_value)
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)

        self.drl_agent.reset()

        obs = np.concatenate([
            self.drl_agent.user_rb_matrix,
            self.drl_agent.target_allocation_matrix
        ])
        return obs, {}

    def step(self, action):
        target_reached = self.drl_agent.perform_action(action)
        self.render()

        reward = 10 if target_reached else 0
        terminated = target_reached

        obs = np.concatenate([
            self.drl_agent.user_rb_matrix,
            self.drl_agent.target_allocation_matrix
        ])
        return obs, reward, terminated, False, {}

    def render(self):
        self.drl_agent.render()

# ------------------------- Registration & Test -------------------------
if __name__ == "__main__":
    
    env1 = OneBasedActionWrapper(gym.make('drl-agent-test'))
    env2 = OneBasedActionWrapper(gym.make('drl-agent-test'))

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    print(obs1)
    print()
    print(obs2)

    obs1, _, _, _, _ = env1.step(3)
    obs2, _, _, _, _ = env2.step(3)
    
    print(obs1)
    print()
    print(obs2)


    # print("Initial observations equal:", np.array_equal(obs1, obs2))

    # # obs1, _, _, _, _ = env1.step(3)
    # # obs2, _, _, _, _ = env2.step(3)

    # print("Observations after one step equal:", np.array_equal(obs1, obs2))

    # print("Check environment begin")
    # check_env(env1.unwrapped)
    # print("Check environment end")
