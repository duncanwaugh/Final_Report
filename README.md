# Intelligent Transportation Systems - Deep Q-Network for Autonomous Driving

This repository contains the code and implementation of a Deep Q-Network (DQN) for autonomous driving tasks within the "highway-env" simulation, developed as part of the Intelligent Transportation Systems course project.

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Scope](#scope)
- [Objectives](#objectives)
- [Code Architecture](#code-architecture)

## Project Overview

This project explores the application of deep reinforcement learning for autonomous driving, specifically focusing on the development of a DQN model to handle complex driving tasks, such as lane-keeping and overtaking. The DQN was initially implemented using the Stable Baselines framework, and later optimized with a custom-built PyTorch implementation for improved flexibility and performance tuning.

## Motivation

As the complexity of real-world driving scenarios increases, traditional algorithmic approaches to autonomous driving face challenges. Reinforcement learning (RL) offers a dynamic alternative, allowing systems to learn from direct interaction with the environment. This project aims to leverage RL, specifically DQNs, to improve the performance of autonomous driving systems in dynamic conditions.

## Scope

The project applies DQN models to the "highway-env" simulation, testing the models in everyday driving scenarios. We implemented and compared two frameworks—Stable Baselines and PyTorch—assessing their efficiency and effectiveness in optimizing driving strategies.

## Objectives

- Develop a DQN model using the Stable Baselines framework and use it as a performance benchmark.
- Transition to a custom PyTorch implementation to improve adaptability and performance.
- Optimize hyperparameters using Optuna to fine-tune the model.
- Evaluate model performance in different driving environments, including highway, merge, and roundabout scenarios.

## Code Architecture

The project code is structured as follows:

- **Stable Baselines Framework**: Used to establish a baseline for model performance.
- **PyTorch Implementation**: Includes custom DQN architecture, memory replay, epsilon-greedy action selection, and training optimization.
- **Optuna**: Hyperparameter optimization framework used for tuning model parameters like learning rate, batch size, and discount factor.

### System Architecture

1. **Environment Configuration**: Simulation setup using "highway-env".
2. **Hyperparameter Optimization**: Optimization using Optuna for key variables such as learning rate, gamma, and network architecture.
3. **Training and Evaluation**: Models are trained and evaluated in different environments, with performance tracked using metrics like reward and loss.


