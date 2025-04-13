import sys
import random
import numpy as np
from os import path

import gymnasium
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete

class DrlAgent:

    # Initialize the grid size
    def __init__(self, n_RBs = 5, n_users = 3, n_subslots = 7):

        '''
        n_RBs: Number of Resource Blocks Available
        n_users: Number of Users requesting for Access
        n_subslots: Number of Subslots in the RBs

        Example:
            When n_users = 3 and n_RBs = 5

            [[1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1]]

            rows -> each Users Status
            columns -> each RBs Status
        '''
        self.n_RBs = n_RBs
        self.n_users = n_users
        self.n_subslots = n_subslots
        self.reset()

        self.last_action = (0, 0, None)

    def allocate_round_robin_matrix(n_users, n_rbs):
        """
        Allocates resource blocks to users using a round-robin approach and returns
        the allocation matrix.

        Args:
            n_users (int): The number of users (rows in the matrix).
            n_rbs (int): The number of resource blocks (columns in the matrix).

        Returns:
            numpy.ndarray: An allocation matrix of shape (n_users, n_rbs) where
                        matrix[i, j] = 1 if user i is allocated resource block j,
                        and 0 otherwise.
        """

        allocation_matrix = np.zeros((n_users, n_rbs), dtype=int)
        rb_index = 0
        while rb_index < n_rbs:
            user_index = rb_index % n_users
            allocation_matrix[user_index, rb_index] = 1
            rb_index += 1
        return allocation_matrix

    def reset(self, seed = None):
        # Initilize the Grid - Setting all the n_RBs x n_users Matrix to Zeros
        # self.user_rb_matrix = np.zeros((self.n_users, self.n_RBs))
        self.user_rb_matrix = DrlAgent.allocate_round_robin_matrix(self.n_users, self.n_RBs)

        # Random Target Position
        random.seed(seed)

        # Target Matrix
        # self.target_allocation_matrix = DrlAgent.allocate_round_robin_matrix(self.n_users, self.n_RBs)
        self.target_allocation_matrix = self.user_rb_matrix.copy()

        # Iterate over each position in the mask in target matrix
        for i in range(self.target_allocation_matrix.shape[0]):  # Rows
            for j in range(self.target_allocation_matrix.shape[1]):  # Columns
                if self.target_allocation_matrix[i, j] == 1:
                    # If mask allows action, sample a value in [0, M-1]
                    self.target_allocation_matrix[i, j] = np.random.randint(0, self.n_subslots)
                else:
                    # If mask disallows action, set it to 0
                    self.target_allocation_matrix[i, j] = 0
        return self.target_allocation_matrix


    def perform_action(self, agent_action) -> bool:
        # Iterate over each position in the rb_user_martrix to allocate the resources
        for i in range(self.last_action[0], self.user_rb_matrix.shape[0]):  # Rows
            for j in range(self.last_action[1], self.user_rb_matrix.shape[1]):  # Columns
                if self.user_rb_matrix[i, j] != 0:
                    # If mask allows action, sample a value in [0, M-1]
                    self.user_rb_matrix[i, j] = agent_action
                    self.last_action = (i, j + 1, agent_action)
                    return
                else:
                    continue

    def render(self):
        # Print current state on console
        for r in range(self.n_users):
            for c in range(self.n_RBs):
                print(self.user_rb_matrix[r][c], end = ' ')
            print() # new line
        print() # new line

# For unit testing
if __name__=="__main__":
    drlagent = DrlAgent()
    drlagent.render()
    count = 10

    while(count > 0):
        rand_action = random.randint(0, 6)
        print(rand_action)

        drlagent.perform_action(rand_action)
        drlagent.render()

        count = count - 1