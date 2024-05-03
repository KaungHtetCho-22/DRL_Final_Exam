from mobileRobot_env import CustomEnv
import pygame

screen = pygame.display.set_mode((800, 600))

env = CustomEnv(2)
episodes = 10


for episode in range(episodes):
	done = False
	obs, info = env.reset()
	while True:#not done:
		random_action = env.action_space.sample()
		print("action",random_action)
		obs, reward, done, info, _ = env.step(random_action)
		print('reward',reward)
