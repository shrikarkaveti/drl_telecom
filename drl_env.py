import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import drl_agent as da
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='drl-agent-v0',                                # call it whatever you want
    entry_point='v0_drl_agent_env:DRLAgentEnv', # module_name:class_name
)

# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/

class DRLAgentEnv(gym.Env):
    def __init__(self, n_users = 4, n_rbs = 5, render_mode = None):
        self.n_users = n_users
        self.n_rbs = n_rbs
        self.render_mode = render_mode

        # Initialize the DRL Agent problem
        self.drl_agent = drl_agent

    # Gym required function (and parameters) to reset the environment
    def reset(self):

    # Gym required function (and parameters) to perform an action
    def step(self, action):

    # Gym required function to render environment
    def render(self):
        self.warehouse_robot.render()

# For unit testing
if __name__=="__main__":
    env = gym.make('drl-agent-v0', render_mode='human')

    # Use this to check our custom environment
    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    while(True):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)

        if(terminated):
            obs = env.reset()[0]
