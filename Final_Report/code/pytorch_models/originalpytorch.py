import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import random
import math
import os


# Assuming highway_env is already installed and available
import highway_env  
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Setup matplotlib for plotting
plt.ion()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 1024
GAMMA = 0.8 #0.9 for highway, 0.8 for merge
EPS_START = 0.999
EPS_END = 0.05
EPS_DECAY = 500
LEARNING_RATE = 0.00002679
NUM_EPISODES = 200
MAX_STEPS_PER_EPISODE = 100
MEMORY_REPLAY = 20000

# Initialize environment
env = gym.make("merge-v0",render_mode="human")
n_actions = env.action_space.n
n_observations = np.prod(env.observation_space.shape)

# Define DQN Architecture
class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, outputs)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Select Action
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

# Optimize Model
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Networks and Optimizer
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_REPLAY)

# Visualization Function
episode_rewards = []

# Visualization Function
def plot_rewards():
    plt.figure(2)
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')

    # Plot the rewards
    plt.plot(episode_rewards, label='Rewards')

    # Calculate and plot the line of best fit if we have enough data points
    if len(episode_rewards) > 1:
        episodes = np.arange(len(episode_rewards))
        slope, intercept = np.polyfit(episodes, episode_rewards, 1)
        plt.plot(episodes, slope * episodes + intercept, 'r-', label=f'Best Fit: y={slope:.2f}x+{intercept:.2f}')

    plt.legend()
    plt.pause(0.001)  # pause a bit so that plots are updated

def soft_update(target_net, policy_net, tau=0.005):
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)

# Training Loop
for i_episode in range(NUM_EPISODES):
    env.reset()
    current_state_tuple = env.reset()  # This now correctly captures the entire tuple
    current_state = current_state_tuple[0]  # Assuming the first element is the state array

    # Now, flatten the NumPy array part of current_state
    state_tensor = torch.tensor([current_state.flatten()], device=device, dtype=torch.float32)

    total_reward = 0

    for t in range(MAX_STEPS_PER_EPISODE):
        action = select_action(state_tensor)
        output = env.step(action.item())
        #env.render()
        #print(output)
        
        # Properly handle the environment's step output, assuming similar structure to current_state
        next_state, reward, done, _, info = output

        # Convert next_state to a PyTorch tensor
        next_state_tensor = torch.tensor([next_state.flatten()], device=device, dtype=torch.float32)

        # Store the transition in memory
        memory.push(state_tensor, action, next_state_tensor if not done else None, torch.tensor([reward], device=device, dtype=torch.float32))

        # Update state_tensor for the next iteration
        state_tensor = next_state_tensor if not done else None
        total_reward += reward

        if done:
            print(f"Episode {i_episode+1} ended after {t+1} steps with done={done}. Total reward: {total_reward}")
            break

    episode_rewards.append(total_reward)  # Record the total reward for this episode
    plot_rewards()
    # Periodically update the target network
    soft_update(target_net, policy_net)

    # Log episode information periodically
    if i_episode % 10 == 0:
        print(f"Episode {i_episode+1}: total reward = {total_reward}, steps = {t+1}")



print('Training Complete')
env.close()
plt.ioff()
plt.show()
