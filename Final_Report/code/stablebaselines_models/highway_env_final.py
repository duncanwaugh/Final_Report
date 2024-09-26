import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import optuna
import os
import matplotlib.pyplot as plt

import highway_env  # This line is necessary for registering the environments
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def optimize_dqn(trial):
    """Hyperparameter optimization function."""
    log_dir = "./logs_highway"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Suggested values for the hyperparameters using the updated method
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1, log=True)
    net_arch = trial.suggest_categorical('net_arch', [(64,), (128, 128), (256, 256)])
    gamma = trial.suggest_float('gamma', 0.8, 0.9999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    
    env = gym.make("highway-fast-v0")
    env = Monitor(env)  # Wrap the environment with Monitor for accurate tracking
    model = DQN("MlpPolicy", env, policy_kwargs=dict(net_arch=net_arch),
                learning_rate=learning_rate, buffer_size=15000,
                learning_starts=200, batch_size=batch_size, gamma=gamma,
                train_freq=1, gradient_steps=1, target_update_interval=50, verbose=0, tensorboard_log=log_dir)
    
    eval_env = gym.make("highway-fast-v0")
    eval_env = Monitor(eval_env)  # Ensure the evaluation environment is also wrapped with Monitor
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=500,
                                 deterministic=True, render=False)
    
    
    model.learn(total_timesteps=5000, callback=eval_callback)
    eval_env.close()
    env.close()
    
    best_model_path = os.path.join(log_dir, "best_model.zip")
    best_model = DQN.load(best_model_path, env=gym.make("highway-fast-v0", render_mode="human"))
    
    eval_env = gym.make("highway-fast-v0", render_mode="human")
    eval_env = Monitor(eval_env)  # Wrap again for evaluation
    mean_reward, _ = evaluate_policy(best_model, eval_env, n_eval_episodes=10)
    eval_env.close()
    
    return mean_reward

if __name__ == "__main__":
    TRAIN = False
    
    if TRAIN:
        study = optuna.create_study(direction='maximize')
        study.optimize(optimize_dqn, n_trials=20)
        
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        # Evaluate the trained model
        env = gym.make("highway-fast-v0", render_mode="human")
        env = Monitor(env)  # Ensure evaluation env is wrapped with Monitor
        model_path = r"C:\4412_labs\progress_report\logs_highway\best_model.zip"  # Update this path
        model = DQN.load(model_path, env=env)
        all_rewards = []

        for video in range(20):
            done = truncated = False
            obs, info = env.reset()
            total_reward = 0
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
            all_rewards.append(total_reward)

        env.close()

        # Plot the rewards
        plt.figure(figsize=(10, 5))
        plt.plot(all_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.show()
