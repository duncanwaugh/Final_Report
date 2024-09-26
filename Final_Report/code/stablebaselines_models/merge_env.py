import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
import os
import torch
import matplotlib.pyplot as plt
import highway_env  # This line is necessary for registering the environments

# Automatically use CUDA if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Setting an environment variable to address a known issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ENV_NAME = "merge-v0"
MODEL_SAVE_DIR = "./highwaymerge_dqn_model"
TOTAL_TIMESTEPS = 3000
SIMULATION_FREQUENCY = 15
RENDER_MODE = "rgb_array"  # Change to "human" if you want to see the simulation
NUM_TEST_EPISODES = 20
TRAIN = True

# Ensure the MODEL_SAVE_DIR exists
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "model")

def create_env(env_name, render_mode="human"):
    env = gym.make(env_name, render_mode=render_mode)
    env.configure({"simulation_frequency": SIMULATION_FREQUENCY})
    env = Monitor(env)  # Wrap the environment with Monitor here
    return env

def train_model(env, save_path, learning_rate, net_arch, batch_size, gamma):
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
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        total_rewards.append(total_reward)
        steps_per_episode.append(steps)

    return total_rewards, steps_per_episode

def optimize_dqn(trial):
    """Optimize DQN model hyperparameters."""
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    net_arch = trial.suggest_categorical('net_arch', [[64], [128, 128], [256, 256]])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    gamma = trial.suggest_uniform('gamma', 0.8, 0.9999)
    
    env = create_env(ENV_NAME, RENDER_MODE)
    model = train_model(env, MODEL_SAVE_PATH, learning_rate, net_arch, batch_size, gamma)
    
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    
    env.close()
    return mean_reward

if __name__ == "__main__":
    if TRAIN:
        study = optuna.create_study(direction='maximize')
        study.optimize(optimize_dqn, n_trials=10)
        
        print("Number of finished trials:", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        
        print(" Value:", trial.value)
        print(" Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
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
