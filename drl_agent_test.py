import sys
import random
import numpy as np
from os import path

import gymnasium as gym
from gymnasium import spaces

class DrlAgent:
    def __init__(self, n_RBs=4, n_users=5, n_subslots=6, np_random=None):
        self.n_RBs = n_RBs
        self.n_users = n_users
        self.n_subslots = n_subslots
        self.last_action = (0, 0, None)
        self.current_pos = (0, 0)
        self.np_random = np_random

    def allocate_matrix(self, n_users, n_rbs):
        allocation_matrix = np.zeros((n_users, n_rbs), dtype=int)

        if n_users == n_rbs:
            allocated_rbs_index = np.random.permutation(n_rbs)
            for user_index in range(n_users):
                allocation_matrix[user_index][allocated_rbs_index[user_index]] = 1

        elif n_users < n_rbs:
            allocated_rbs_index = np.random.choice(n_rbs, n_users, replace=False)
            for user_index in range(n_users):
                allocation_matrix[user_index][allocated_rbs_index[user_index]] = 1

            remaining_allocation_rbs = [rb for rb in range(n_rbs) if rb not in allocated_rbs_index]
            for rb_index in remaining_allocation_rbs:
                user_index = np.random.randint(0, n_users)
                allocation_matrix[user_index][rb_index] = 1

        else:
            allocated_users_index = np.random.choice(n_users, n_rbs, replace=False)
            for rb_index in range(n_rbs):
                allocation_matrix[allocated_users_index[rb_index]][rb_index] = 1

        return allocation_matrix

    def reset(self):
        self.user_rb_matrix = self.allocate_matrix(self.n_users, self.n_RBs)

        self.target_allocation_matrix = self.user_rb_matrix.copy()
        for i in range(self.target_allocation_matrix.shape[0]):
            for j in range(self.target_allocation_matrix.shape[1]):
                if self.target_allocation_matrix[i, j] == 1:
                    self.target_allocation_matrix[i, j] = np.random.randint(1, self.n_subslots)
                else:
                    self.target_allocation_matrix[i, j] = 0

        self.current_pos = (0, 0)
        return self.target_allocation_matrix

    def perform_action(self, agent_action) -> bool:
        user_index, rb_index = self.current_pos
        found = False

        for _ in range(self.n_users * self.n_RBs):
            if self.user_rb_matrix[user_index][rb_index] != 0:
                self.user_rb_matrix[user_index][rb_index] = agent_action
                self.last_action = (user_index, rb_index, agent_action)
                found = True

            rb_index += 1
            if rb_index >= self.n_RBs:
                rb_index = 0
                user_index += 1
                if user_index >= self.n_users:
                    user_index = 0

            if found:
                break

        self.current_pos = (user_index, rb_index)
        return np.array_equal(self.user_rb_matrix, self.target_allocation_matrix)

    def render(self):
        for r in range(self.n_users):
            print(" ".join(str(x) for x in self.user_rb_matrix[r]))
        print()