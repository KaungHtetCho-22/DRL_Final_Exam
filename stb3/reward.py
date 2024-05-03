import pygame
import time  # Import time module to measure execution time
from stable_baselines3 import PPO
from mobileRobot_env import CustomEnv

# Initialize pygame
pygame.init()

# Set up the display using pygame
screen = pygame.display.set_mode((800, 600))

# Assuming CustomEnv is already defined elsewhere and properly imported
env = CustomEnv(2)

# Load the trained model
model_path = "/home/koala/AIT second semester/DRL/Final_exam/models/ppo/20000.zip"
model = PPO.load(model_path, env=env)

# Lists to store rewards, episode lengths, and times
episode_rewards = []
episode_lengths = []
episode_times = []  # List to store the time taken for each episode

# Inference loop
episodes = 1  # Set the number of episodes to 2
for i in range(episodes):
    start_time = time.time()  # Start timing

    obs, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info, _ = env.step(action)
        episode_reward += reward
        episode_length += 1
        env.render(mode='human')  # Render the environment in human mode during inference
        print(f"Reward: {reward}")
    
    end_time = time.time()  # End timing
    episode_duration = end_time - start_time  # Calculate the duration of the episode
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    episode_times.append(episode_duration)  # Store the time taken for the episode

    print(f"Episode {i + 1}: Total Reward = {episode_reward}, Total Length = {episode_length}, Time Taken = {episode_duration:.2f} seconds")

pygame.quit()  # Make sure to quit pygame when done
