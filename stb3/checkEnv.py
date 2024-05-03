from mobileRobot_env import CustomEnv
from stable_baselines3.common.env_checker import check_env
import pygame

if __name__ == "__main__":
    pygame.init()
    env = CustomEnv(2)
    check_env(env)
