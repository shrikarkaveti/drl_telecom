import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import drl_env_test
from drl_env_test import OneBasedActionWrapper

# ---------------------- Actor-Critic Network ----------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.Linear(64, action_dim)
        self.fc_v = nn.Linear(64, 1)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # Apply logarithmic softmax for numerical stability
        pi_logits = self.fc_pi(x)
        # Clip logits to prevent extreme values
        pi_logits = torch.clamp(pi_logits, -20, 20)
        pi = F.softmax(pi_logits, dim=-1)
        v = self.fc_v(x)
        return pi, v

# ---------------------- Training Setup ----------------------
env = OneBasedActionWrapper(gym.make('drl-agent-test'))
obs_shape = env.observation_space.shape
state_dim = obs_shape[0] * obs_shape[1]  # Flattened input
action_dim = env.action_space.n

# Initialize the agent and optimizer with gradient clipping
agent = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(agent.parameters(), lr=0.0005)  # Lower learning rate for stability

num_episodes = 1000
discount_factor = 0.99

# Add exploration parameter
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995
epsilon = epsilon_start

# ---------------------- Training Loop ----------------------
for episode in range(num_episodes):
    with open('test_result.txt', 'a') as file:
        state, _ = env.reset()
        
        # Handle potential NaN in observations
        if np.isnan(state).any():
            print("Warning: NaN in initial state. Resetting to zeros.")
            state = np.zeros_like(state)
            
        state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        max_steps = 1000
        
        while not (done or truncated) and steps < max_steps:
            try:
                probs, value = agent(state)
                probs = probs.squeeze(0)
                
                # Check for NaN in probabilities and handle it
                if torch.isnan(probs).any() or torch.sum(probs) == 0:
                    print("Warning: NaN in probabilities or zero sum. Using uniform distribution.")
                    probs = torch.ones(action_dim) / action_dim
                    
                value = value.squeeze(0)
                
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(0, action_dim)
                else:
                    # Convert to numpy safely
                    prob_np = probs.detach().numpy()
                    
                    # Ensure the probabilities are valid
                    if np.isnan(prob_np).any() or not np.isclose(np.sum(prob_np), 1.0):
                        prob_np = np.ones(action_dim) / action_dim
                        
                    action = np.random.choice(action_dim, p=prob_np)
                
                # Convert to 1-based for wrapper
                next_state, reward, done, truncated, info = env.step(action + 1)
                
                # Handle potential NaN in next_state
                if np.isnan(next_state).any():
                    print("Warning: NaN in next_state. Using previous state.")
                    next_state = state.squeeze(0).numpy().reshape(obs_shape)
                
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).flatten().unsqueeze(0)
                
                with torch.no_grad():
                    next_probs, next_value = agent(next_state_tensor)
                    next_value = next_value.squeeze(0)
                    
                    # Handle NaN in next_value
                    if torch.isnan(next_value):
                        next_value = torch.tensor(0.0)
                    
                terminal = done or truncated
                
                # Calculate TD target and error
                td_target = reward + discount_factor * next_value * (1 - int(terminal))
                td_error = td_target - value
                
                # Check for NaN before calculating loss
                if not torch.isnan(probs[action]) and torch.isfinite(probs[action]) and probs[action] > 0:
                    actor_loss = -torch.log(probs[action]) * td_error.detach()
                else:
                    actor_loss = torch.tensor(0.0, requires_grad=True)
                
                critic_loss = td_error.pow(2)
                
                # Total loss with gradient clipping
                loss = actor_loss + critic_loss
                
                # Update network with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)  # Clip gradients
                optimizer.step()
                
                total_reward += reward
                state = next_state_tensor
                steps += 1
                
            except Exception as e:
                print(f"Error during training: {e}")
                # Reset to a safe state and continue
                action = np.random.randint(0, action_dim)
                next_state, reward, done, truncated, info = env.step(action + 1)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).flatten().unsqueeze(0)
                state = next_state_tensor
                steps += 1
        
        # Decay epsilon
        epsilon = max(epsilon_final, epsilon * epsilon_decay)
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {steps}, Epsilon = {epsilon:.4f}")
        file.write(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {steps}, Epsilon = {epsilon:.4f}\n")