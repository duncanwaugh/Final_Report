import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import os
import torch
import matplotlib.pyplot as plt
import highway_env  # This line is necessary for registering the environments

# Automatically use CUDA if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Setting an environment variable to address a known issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ENV_NAME = "highway-fast-v0"
MODEL_SAVE_DIR = "./highway_dqn_model"
TOTAL_TIMESTEPS = 3000
SIMULATION_FREQUENCY = 15
RENDER_MODE = "human"  # Change to "human" if you want to see the simulation
NUM_TEST_EPISODES = 20
TRAIN = False

# Ensure the MODEL_SAVE_DIR exists
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "model")

def create_env(env_name, render_mode="human"):
    env = gym.make(env_name, render_mode=render_mode)
    env.configure({"simulation_frequency": SIMULATION_FREQUENCY})
    env = Monitor(env)  # Wrap the environment with Monitor here
    return env

def train_model(env, save_path, learning_rate=0.0022612406819308988, net_arch=[128, 128], batch_size=64, gamma=0.9755349680416233):
    model = DQN(
        "MlpPolicy",
        env,
        device=device,
        policy_kwargs=dict(net_arch=net_arch),
        learning_rate=learning_rate,
        buffer_size=15000,
        learning_starts=200,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log=os.path.dirname(save_path),
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(save_path)
    return model

def test_model(env, model_path, num_episodes=10):
    model = DQN.load(model_path, env=env)
    total_rewards = []
    steps_per_episode = []

    for episode in range(num_episodes):
        obs = env.reset()  # Reset the environment at the start of each episode
        if isinstance(obs, tuple):  # Check if the observation is wrapped in a tuple
            obs = obs[0]  # Assume the first element of the tuple is the observation

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            results = env.step(action)
            if isinstance(results, tuple):  # Check if the step results are in a tuple
                obs, reward, done, *info = results
            else:
                obs, reward, done, info = results  # Assuming a standard step output

            total_reward += reward
            steps += 1

            if done and episode < num_episodes - 1:  # Reset only if there are more episodes to process
                obs = env.reset()
                if isinstance(obs, tuple):  # Check again after resetting
                    obs = obs[0]

        total_rewards.append(total_reward)
        steps_per_episode.append(steps)

    return total_rewards, steps_per_episode











if __name__ == "__main__":
    if TRAIN:
        env = create_env(ENV_NAME, RENDER_MODE)
        model = train_model(env, MODEL_SAVE_PATH)
        env.close()
    else:
        env = create_env(ENV_NAME, RENDER_MODE)
        total_rewards, steps_per_episode = test_model(env, MODEL_SAVE_PATH, num_episodes=NUM_TEST_EPISODES)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, NUM_TEST_EPISODES + 1), total_rewards, label='Total Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.title('Rewards per Episode')
        plt.legend()
        plt.show()
