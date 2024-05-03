import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO, DQN, TD3, DDPG, SAC
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv
import os
import time
from mobileRobot_env import CustomEnv
import pygame

import matplotlib.pyplot as plt

model_dir = f"models/pp02"
logdir = f"logs"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)


env = CustomEnv(2)  # Replace 'screen' with your screen object if needed

env = Monitor(env, logdir)  # Wrap with Monitor for logging
# env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)


# TIMESTEPS = 10000
# iters = 0
# for i in range(1):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="a2c")
#     model.save(f"{model_dir}/{TIMESTEPS*i}")


TIMESTEPS = 10000
for i in range(0, 3):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="pp02")
    model.save(f"{model_dir}/{TIMESTEPS*i}")

    # obs = env.render() 

# Close the environment
# env.close()