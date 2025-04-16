import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import drl_agent as da
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='drl-agent',                               # call it whatever you want
    entry_point='drl_env:DRLAgentEnv', # module_name:class_name
)

# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/

class DRLAgentEnv(gym.Env):
    # metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(self, n_users = 4, n_rbs = 5, n_subslots = 7,  render_mode = None):
        self.n_users = n_users
        self.n_rbs = n_rbs
        self.n_subslots = n_subslots
        self.render_mode = render_mode

        # Initialize the DRL Agent problem
        self.drl_agent = da.DrlAgent(n_RBs = n_rbs, n_users = n_users, n_subslots = n_subslots)

        # Gym requires defining the action space. The action space is agent's set of possible actions.
        # Training code can call action_space.sample() to randomly select an action.
        self.action_space = spaces.Discrete(self.n_subslots)

        # Gym requires defining the observation space. The observation space consists of the agent's and target's set of possible positions.
        # The observation space is used to validate the observation returned by reset() and step().
        # Use a 1D vector: [robot_row_pos, robot_col_pos, target_row_pos, target_col_pos]
        self.observation_space = spaces.Box(
            low = 0,
            high = self.n_subslots - 1,
            shape = (self.n_users * 2, self.n_rbs),
            dtype = np.int64
        )

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed = None, options = None):
        super().reset(seed = seed) # gym requires this call to control randomness and reproduce scenarios.

        # Reset the WarehouseRobot. Optionally, pass in seed control randomness and reproduce scenarios.
        self.drl_agent.reset(seed = seed)

        # Construct the obervation state:
        # [[user_rb_allocation_matirx], [target_user_rb_matrix]]
        obs = np.concatenate([self.drl_agent.user_rb_matrix, self.drl_agent.target_allocation_matrix])

        # Additional info to return. For debugging or whatever.
        info = {}

        # Return observation and info
        return obs, info

    # Gym required function (and parameters) to perform an action
    def step(self, action):
        # Perform action
        target_reached = self.drl_agent.perform_action(action)

        # Determine reward and termination
        reward = 0
        terminated = False

        if target_reached:
            reward = 1
            terminated = True

        # Construct the obseravation state:
        # [[user_rb_allocation_matrix], [target_allocation_matrix]]
        obs = np.concatenate([self.drl_agent.user_rb_matrix, self.drl_agent.target_allocation_matrix])

        # Additonal info to return. for debugging or whatever.
        info = {}

        # Return observation, reward, terminated, truncated (not used), info
        return obs, reward, terminated, False, info

    # Gym required function to render environment
    def render(self):
        self.drl_agent.render()

# For unit testing
if __name__=="__main__":
    env = gym.make('drl-agent')

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
