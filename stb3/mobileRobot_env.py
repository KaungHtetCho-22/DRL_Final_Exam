import numpy as np
import cv2 as cv
import gymnasium as gym
from gymnasium import spaces
import math
import random

import numpy as np
import cv2 as cv
import gymnasium as gym
from gymnasium import spaces
import math
import random
import time  # Import time to use for the timer


class CustomEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, n_robots=2):
        super(CustomEnv, self).__init__()
        self.n_robots = n_robots
        self.mobile_robot = MobileRobot(n_robots)
        self.observation_space = spaces.Box(low=0, high=255, shape=(600, 1000, 3), dtype=np.uint8)
        self.action_space = spaces.MultiDiscrete([8] * n_robots)
        self.state = None
        self.complete = [False] * n_robots
        self.time_limit = 180  # 5 minutes limit in seconds
        self.start_time = time.time()  # Start the timer

    def step(self, actions):
        # current_time = time.time()
        # elapsed_time = current_time - self.start_time

        # if elapsed_time > self.time_limit:
        #     observation, info = self.reset(timeout=True)
        #     total_reward = -5
        #     terminated = True
        #     truncated = False
        #     return observation, total_reward, terminated, truncated, info

        self.state, rewards, terminals, infos = self.mobile_robot.step_pathPlanning1(actions)

        for i in range(self.n_robots):
            if terminals[i]:
                self.complete[i] = True
        
       

        observation = self.render()
        aggregated_info = {
            'individual_rewards': rewards,
            'terminal_states': terminals,
            'additional_info': infos
        }

        if all(self.complete):
            observation, _ = self.reset()

        total_reward = sum(rewards)
        terminated = all(self.complete)
        truncated = False
        return observation, total_reward, terminated, truncated, aggregated_info

    def reset(self, seed=None, options=None, timeout=False):
        super().reset(seed=seed)
        self.mobile_robot.reset_pathPlanning1()

        self.state = self.mobile_robot.get_stackSet()
        self.complete = [False] * self.n_robots
        self.mobile_robot.goal_reached = [False] * self.n_robots
        self.start_time = time.time()  # Reset the timer
        observation = self.render()
        if timeout:
            return observation, {'timeout': True, 'message': 'Reset due to timeout'}
        return observation, {}


    def render(self, mode='human'):
        return self.mobile_robot.render_pathPlanning1()

    def close(self):
        cv.destroyAllWindows()


class MobileRobot(object):
    def __init__(self, n_robots):
        # Parameters for path planning
        self.mapW = 40
        self.mapH = 24
        self.span_x = 37
        self.span_y = 21
        self.gridW = self.mapW + self.span_x*2
        self.gridH = self.mapH + self.span_y*2
        self.robot_rad = 1
        self.group_rad = self.robot_rad*3
        self.group_gap = self.group_rad - self.robot_rad
        self.n_robots = n_robots
        self.n_stacks = 4
        self.n_actions_pathPlanning = 8
        self.n_actions_motionControl = 2
        self.n_obstacles = 3
        self.obs_size_min = 2
        self.obs_size_max = 4
        self.img_ratio = 50
        self.map_ratio = 4
        self.input_dims_pathPlanning = (self.n_stacks, (self.span_y*2 + 1)*self.map_ratio, (self.span_x*2 + 1)*self.map_ratio)
        self.input_dims_motionControl = 2
        self.robot_width = 120
        self.axis_length = self.robot_width
        self.max_actions_motionControl = [100, 1.0]

        self.goal_reached = [False] * n_robots

        self.robot_positions = [[] for _ in range(n_robots)]
    
        # Colors
        self.floor_color = [255, 255, 255]
        self.obstacle_color = [100, 100, 100]
        self.dropOff_color = [0, 255, 0]
        self.loading_color = [0, 255, 255]
        self.path_center_color = [0, 255, 255]
        self.robots_color = [[255, 128, 0], [128, 0, 255], [0, 128, 255], [0, 0, 255], [0, 64, 128]]
        self.paths_color = [[255, 182, 108], [190, 125, 255], [90, 170, 255], [104, 104, 255], [64, 128, 128]]
        self.err_color = [[246, 130, 50], [80, 134, 240], [34, 125, 55]]
        self.wheel_color = [0, 0, 0]
        self.font_color = [0, 0, 0]


    def reset_pathPlanning1(self):
        self.create_obstacles()
        self.get_gridGlobal2()
        self.create_loading_point()
        self.create_dropOff_point()
        self.get_gridGlobal1()
        self.create_goals()
        self.create_starts()
        self.get_gridSet()
        self.robots_pos = self.starts.copy()
        self.goal_reached = [False] * self.n_robots  # Ensure goal statuses are reset
        self.get_paths()
        self.get_stateSet()
        stack_set = self.get_stackSet()
        return stack_set


    def create_obstacles(self):
        while True:
            self.obstacles = []
            for _ in range(100):
                w = random.randrange(self.obs_size_min, self.obs_size_max + 1)
                h = random.randrange(self.obs_size_min, self.obs_size_max + 1)
                x = random.randrange(self.span_x, self.gridW - self.span_x - w + 1)
                y = random.randrange(self.span_y, self.gridH - self.span_y - h + 1)
                if not self.overlap_obstacle((x, y, w, h)):
                    self.obstacles.append([x, y, w, h])
                if len(self.obstacles) == 3:
                    break
            if len(self.obstacles) == 3:
                break

    def overlap_obstacle(self, obs):
        x, y, w, h = obs
        for x_, y_, w_, h_  in self.obstacles:
            if x in range(x_ - w - self.group_rad*2, x_ + w_ + self.group_rad*2 + 1) and \
                y in range(y_ - h - self.group_rad*2, y_ + h_  + self.group_rad*2 + 1):
                return True
        return False

    def get_gridGlobal2(self):
        self.grid_global2 = np.zeros((self.gridH, self.gridW))
        for i in range(self.gridW):
            for j in range(self.gridH):
                if i not in range(self.span_x + self.group_rad, self.gridW - self.span_x - self.group_rad) or \
                    j not in range(self.span_y + self.group_rad, self.gridH - self.span_y - self.group_rad):
                    self.grid_global2[j, i] = -1
                else:  
                    for x, y, w, h in self.obstacles:
                        if i in range(x - self.group_rad, x + w + self.group_rad) and \
                            j in range(y - self.group_rad, y + h + self.group_rad):
                            self.grid_global2[j, i] = -1

    def create_loading_point(self):
        while True: 
            x = random.randrange(self.span_x + self.group_rad, self.gridW - self.span_x - self.group_rad)
            y = random.randrange(self.span_y + self.group_rad, self.gridH - self.span_y - self.group_rad)
            if self.grid_global2[y,x] == 0:
                self.grid_global2[y,x] = 2
                self.loading_point = [x,y]
                break

    def create_dropOff_point(self):
        while True:
            x = random.randrange(self.span_x + self.group_rad, self.gridW - self.span_x - self.group_rad)
            y = random.randrange(self.span_y + self.group_rad, self.gridH - self.span_y - self.group_rad)
            if self.grid_global2[y,x] == 0 and self.cal_distance([x,y], self.loading_point) > (self.mapH + self.mapW)/4:
                self.grid_global2[y,x] = 1
                self.dropOff_point = [x,y]
                break

    def get_gridGlobal1(self):
        self.grid_global1 = np.zeros((self.gridH, self.gridW))
        for i in range(self.gridW):
            for j in range(self.gridH):
                if i not in range(self.span_x + self.robot_rad, self.gridW - self.span_x - self.robot_rad) or \
                    j not in range(self.span_y + self.robot_rad, self.gridH - self.span_y - self.robot_rad):
                    self.grid_global1[j, i] = -1
                else:             
                    for x, y, w, h in self.obstacles:
                        if i in range(x - self.robot_rad, x + w + self.robot_rad) and \
                            j in range(y - self.robot_rad, y + h + self.robot_rad):
                            self.grid_global1[j, i] = -1
        
    def create_goals(self):
        # set goal values
        self.goals = []
        self.img_load_pts = []
        self.img_drop_pts = []
        theta = 2*math.pi/self.n_robots
        for i in range(self.n_robots):
            img_x = int((self.loading_point[0] - self.span_x + self.group_gap*math.cos(theta*i))*self.img_ratio + self.img_ratio//2)
            img_y = int((self.loading_point[1] - self.span_y + self.group_gap*math.sin(theta*i))*self.img_ratio + self.img_ratio//2)
            img_x_ = int((self.dropOff_point[0] - self.span_x + self.group_gap*math.cos(theta*i))*self.img_ratio + self.img_ratio//2)
            img_y_ = int((self.dropOff_point[1] - self.span_y + self.group_gap*math.sin(theta*i))*self.img_ratio + self.img_ratio//2)
            self.img_load_pts.append([img_x, img_y])
            self.img_drop_pts.append([img_x_, img_y_])
            x = int(img_x//self.img_ratio + self.span_x)
            y = int(img_y//self.img_ratio + self.span_y)
            x_ = int(img_x_//self.img_ratio + self.span_x)
            y_ = int(img_y_//self.img_ratio + self.span_y)
            self.goals.append([x,y])
            self.grid_global1[y,x] = i+1 
            self.grid_global1[y_,x_] = i+21 

    def create_starts(self):
        self.starts = []
        # find start position for robot1
        self.robot_positions = []  # Clear previous positions
        for k in range(self.n_robots):
            x = np.random.randint(low=0, high=self.gridW)  # Example position
            y = np.random.randint(low=0, high=self.gridH)
            self.robot_positions.append([x, y])
        while True:
            x = random.randrange(self.span_x + self.robot_rad + 25, self.gridW - self.span_x - self.robot_rad)
            y = random.randrange(self.span_y + self.robot_rad, self.gridH - self.span_y - self.robot_rad)
            if self.grid_global1[y, x] == 0 and self.available_position([x, y]):
                self.starts.append([x, y])
                self.grid_global1[y,x] = 11
                break
        # find start position for robot2
        while True:
            x = random.randrange(self.span_x + self.robot_rad, self.span_x + self.robot_rad + 25)
            y = random.randrange(self.span_y + self.robot_rad + 11, self.gridH - self.span_y - self.robot_rad)
            if self.grid_global1[y, x] == 0 and self.available_position([x, y]):
                self.starts.append([x, y])
                self.grid_global1[y,x] = 12
                break
        # find start position for robot3
        while True:
            x = random.randrange(self.span_x + self.robot_rad, self.span_x + self.robot_rad + 25)
            y = random.randrange(self.span_y + self.robot_rad, self.span_y + self.robot_rad + 11)
            if self.grid_global1[y, x] == 0 and self.available_position([x, y]):
                self.starts.append([x, y])
                self.grid_global1[y,x] = 13
                break
    
    def available_position(self, point):
        for start_pos in self.starts:
            if self.cal_distance(point, start_pos) < self.robot_rad*5:
                return False
        return True

    def get_gridSet(self):
        self.grid_set = []
        for k in range(self.n_robots):
            grid = np.zeros((self.gridH, self.gridW))
            for i in range(self.gridW):
                for j in range(self.gridH):
                    if self.grid_global1[j,i] == -1:
                        grid[j,i] = -1
                    elif self.grid_global1[j,i] == k+1:
                        grid[j,i] = 1
                    elif self.grid_global1[j,i] == k+11:
                        grid[j,i] = 2
            self.grid_set.append(grid)

    def get_paths(self):
        self.paths = []
        for k in range(self.n_robots):
            path = [self.robots_pos[k]]
            self.paths.append(path)

    def get_stateSet(self):
        self.state_set = []
        for k in range(self.n_robots):
            state = self.grid_set[k][self.robots_pos[k][1] - self.span_y: self.robots_pos[k][1] + self.span_y + 1,
                                    self.robots_pos[k][0] - self.span_x: self.robots_pos[k][0] + self.span_x + 1]
            state_resized = cv.resize(state, ((self.span_x*2 + 1)*self.map_ratio, (self.span_y*2 + 1)*self.map_ratio), 
                                    interpolation=cv.INTER_AREA)
            self.state_set.append(state_resized)

    def get_stackSet(self):
        self.stack_set = []
        stack_set = []
        for k in range(self.n_robots):
            stack = [self.state_set[k].copy()]*self.n_stacks
            self.stack_set.append(stack)
            stack_set.append(np.array(stack))
        return stack_set

    def step_pathPlanning1(self, actions):
        new_positions = []
        rewards = []
        terminals = []
        infos = []
        
        for k in range(self.n_robots):
            if self.goal_reached[k]:
                rewards.append(0)
                terminals.append(True)
                infos.append({'message': f'Robot {k} has already reached its goal.'})
                new_positions.append(self.robots_pos[k])  # Add current position to maintain list size
                continue

            new_position = self.robots_pos[k].copy()
            reward = 0
            terminal = False
            info = {}

            # Execute the action
            if actions[k] == 0: # move up-left
                new_position[0] -= 1
                new_position[1] -= 1
            elif actions[k] == 1: # move up
                new_position[1] -= 1
            elif actions[k] == 2: # move up-right
                new_position[0] += 1
                new_position[1] -= 1
            elif actions[k] == 3: # move left
                new_position[0] -= 1
            elif actions[k] == 4: # move right
                new_position[0] += 1
            elif actions[k] == 5: # move down-left
                new_position[0] -= 1
                new_position[1] += 1
            elif actions[k] == 6: # move down
                new_position[1] += 1
            elif actions[k] == 7: # move down-right
                new_position[0] += 1
                new_position[1] += 1

            if self.reachGoal(new_position, k):
                reward = 10
                terminal = True
                info = {'message': f'Robot {k} reached its goal'}
                self.goal_reached[k] = True
            elif self.HitObstacle(new_position, self.grid_set[k]):
                reward = -3
                new_position = self.robots_pos[k].copy()  # Reset to current if hit obstacle
                info = {'message': 'hit obstacle'}
            else:
                reward = -1  
            new_positions.append(new_position)
            rewards.append(reward)
            terminals.append(terminal)
            infos.append(info)

        self.update_gridSet(new_positions)
        self.add_paths()
        self.get_stateSet()
        stack_set = self.update_stackSet()

        return stack_set, rewards, terminals, infos


    def reachGoal(self, position, idx):
        if position == self.goals[idx]:
            return True
        else:
            return False

    def HitObstacle(self, position, grid):
        if grid[position[1], position[0]] == -1:
            return True
        else:
            return False

    def update_gridSet(self, new_positions):
        for k in range(self.n_robots):
            old_x, old_y = self.robots_pos[k]
            new_x, new_y = new_positions[k]

            # Check grid boundaries and existing grid value before setting new value
            if 0 <= new_y < len(self.grid_global1) and 0 <= new_x < len(self.grid_global1[0]):
                self.grid_global1[old_y, old_x] = 0  # Clear old position
                self.grid_global1[new_y, new_x] = k + 11  # Set new position with robot's unique index

            self.grid_set[k][old_y, old_x] = 0
            if 0 <= new_y < len(self.grid_set[k]) and 0 <= new_x < len(self.grid_set[k][0]):
                self.grid_set[k][new_y, new_x] = 2  # Mark new position on robot-specific grid

            self.robots_pos[k] = [new_x, new_y]

    def add_paths(self):
        for k in range(self.n_robots):
            if self.robots_pos[k] != self.paths[k][-1]:
                self.paths[k].append(self.robots_pos[k])

    def update_stackSet(self):
        stack_set = []
        for k in range(self.n_robots):
            self.stack_set[k].pop(0)
            self.stack_set[k].append(self.state_set[k].copy())
            stack_set.append(np.array(self.stack_set[k]))
        return stack_set


    # def cal_rewards(self, position, idx):
    #     # Calculate distance from the current position to the drop-off point
    #     dist_to_goal = self.cal_distance(position, self.dropOff_point)
    #     # Define radius for reward levels
    #     reward_zone_radii = [5, 10, 15]  # These can be adjusted based on your grid scale

    #     # Assign rewards based on distance
    #     if dist_to_goal <= reward_zone_radii[0]:
    #         return 3  # Nearest zone
    #     elif dist_to_goal <= reward_zone_radii[1]:
    #         return 2  # Middle zone
    #     elif dist_to_goal <= reward_zone_radii[2]:
    #         return 1  # Farthest zone
    #     else:
    #         return 0  # Outside of specified zones

    def cal_distance(self, pos1, pos2):
        '''
        Calculate distance between 2 points.
        '''
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x1-x2)**2+(y1-y2)**2)**0.5

    def render_pathPlanning1(self):
        img = np.zeros((self.gridH, self.gridW, 3), dtype=np.uint8)
        for i in range(self.gridW):
            for j in range(self.gridH):
                # filled color obstacles
                if self.grid_global1[j, i] == -1:
                    img[j, i] = np.array(self.obstacle_color)
                # filled color loading points
                elif self.grid_global1[j, i] in range(1, self.n_robots + 1):
                    img[j, i] = np.array(self.loading_color)
                # filled color dropOff points
                elif self.grid_global1[j, i] in range(21, self.n_robots + 21):
                    img[j, i] = np.array(self.dropOff_color)
                # filled color floor
                else:
                    img[j, i] = np.array(self.floor_color)
                # filled color robot's positions
                for k in range(self.n_robots):
                    if self.grid_global1[j, i] == k + 11:
                        img[j, i] = np.array(self.robots_color[k])

        img_resized = cv.resize(img, (self.gridW * self.img_ratio, self.gridH * self.img_ratio),
                                interpolation=cv.INTER_AREA)

        # draw path
        for k in range(self.n_robots):
            for i, [x, y] in enumerate(self.paths[k]):
                x_point = x * self.img_ratio + self.img_ratio // 2
                y_point = y * self.img_ratio + self.img_ratio // 2
                cv.circle(img_resized, (x_point, y_point), 4, self.paths_color[k], thickness=-1)
                if i > 0:
                    cv.line(img_resized, (x_prev, y_prev), (x_point, y_point), self.paths_color[k], thickness=2)
                x_prev = x_point
                y_prev = y_point

        img_croped = img_resized[self.span_y * self.img_ratio: (self.gridH - self.span_y) * self.img_ratio,
                    self.span_x * self.img_ratio: (self.gridW - self.span_x) * self.img_ratio]

        img_resized = cv.resize(img_croped, (img_croped.shape[1] // 2, img_croped.shape[0] // 2))
        cv.imshow('Mobile Robot Path Planning I', img_resized)
        cv.waitKey(1)

        return img_resized


# if __name__ == "__main__":
#     env = CustomEnv(2)
#     obs = env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()  # Randomly sample an action
#         obs, reward, done, truncated, info = env.step(action)
#         env.render()
#     env.close()