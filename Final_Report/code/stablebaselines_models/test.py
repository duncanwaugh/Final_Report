import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
import os
import matplotlib.pyplot as plt
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Environment setup and optimization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import highway_env  # noqa: F401

# Configuration
TRAIN = True
ENV_NAME = "highway-fast-v0"
MODEL_SAVE_PATH = "highway_dqn/model"
VIDEO_FOLDER = "highway_dqn/videos"
TOTAL_TIMESTEPS = int(1e4)
SIMULATION_FREQUENCY = 15
RENDER_MODE = "rgb_array"  # Change to "rgb_array" for headless environments
NUM_TEST_EPISODES = 20
from stable_baselines3.common.utils import get_device
device = get_device("auto")  # "auto" will automatically use CUDA if available
print(f"Using device: {device}")

# Create and configure the environment
def create_env(env_name, render_mode="rgb_array"):
    env = gym.make(env_name, render_mode=render_mode)
    env.configure({"simulation_frequency": SIMULATION_FREQUENCY})
    return env

# Training function
def train_model(env, save_path):
    model = DQN(
        "MlpPolicy",
        env,
        device=device,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log=save_path,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(save_path)

# Updated run_simulation for testing and collecting metrics
def test_model(env, model_path, num_episodes=10):
    model = DQN.load(model_path, env=env)
    total_rewards = []
    steps_per_episode = []

    for episode in range(num_episodes):
        done = truncated = False
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)
        steps_per_episode.append(steps)

    env.close()
    return total_rewards, steps_per_episode

# Visualizing metrics with matplotlib
def plot_metrics(total_rewards, steps_per_episode):
    episodes = range(1, len(total_rewards) + 1)
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(episodes, total_rewards, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Steps', color=color)  # we already handled the x-label with ax1
    ax2.plot(episodes, steps_per_episode, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

if __name__ == "__main__":
    env = create_env(ENV_NAME, RENDER_MODE)
    if TRAIN:
        train_model(env, MODEL_SAVE_PATH)
        env.close()  # Close the env to release resources

    # Prepare the environment for testing
    env = create_env(ENV_NAME, RENDER_MODE)
    total_rewards, steps_per_episode = test_model(env, MODEL_SAVE_PATH, num_episodes=NUM_TEST_EPISODES)
    plot_metrics(total_rewards, steps_per_episode)
