import numpy as np
import math
import cv2 as cv
import random
import time

class MobileRobot(object):
    '''
    This is an environment to evaluate the performance of multiple robots for 
    object transportation task in simulation. it is a combination of environments 
    of path planning, path following, and point following for multiple robots.
    However, we have no object in simulation. Thus, we consider a group of robot 
    moving with a specific formation instead.

    Environemnt details:
    - Path planning
        grid_map size: H24 x W40 pixels 
        Image render size: H600 x W1200 pixels (image ratio = 25 from grid_map)
    - Path following
        map size: H1200 x W2000 pixels 
        Image render size: H600 x W1200 pixels (image ratio = 0.5 from map)
    - Formation control
        map size: H1200 x W2000 pixels 
        Image render size: H600 x W1200 pixels (image ratio = 0.5 from map)
    '''
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

        # Parameters for PID control 
        self.u_KP = 1.0
        self.u_KI = 0.02
        self.u_KD = 0.1
        self.w_KP = 1.0
        self.w_KI = 0.02
        self.w_KD = 0.1

        self.Fu_KP = 5.0
        self.Fu_KI = 0.1
        self.Fu_KD = 0.5
        self.Fw_KP = 1.5
        self.Fw_KI = 0.03
        self.Fw_KD = 0.15

    # ================================================================= #
    #       Step 1: Path planning from start to loading position        #
    # ================================================================= #

    def reset_pathPlanning1(self):
        '''
        Reset map to start new episode.
        - Random obstacles' position.
        - Random loading position.
        - Random dropoff position.
        - Create global grid map from loading point to dropoff point.
        - Create goal positions according to loading point.
        - Random start positions for all robots.
        - Create global grid map from start positions to goal positions.
        - Create grid map for each robot.
        - Reset path for each robot.
        '''
        self.create_obstacles()
        self.get_gridGlobal2()
        self.create_loading_point()
        self.create_dropOff_point()
        self.get_gridGlobal1()
        self.create_goals()
        self.create_starts()
        self.get_gridSet()
        self.robots_pos = self.starts.copy()
        self.get_paths()
        self.get_stateSet()
        stack_set = self.get_stackSet()
        return stack_set

    def create_obstacles(self):
        '''
        - Randomly place 3 static obstacles inside map's area.
        - Obstacle must not overlap to each other.
        - There is free space between obstacles.
        '''
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
        '''
        Check obstacles overlap or not.
        '''
        x, y, w, h = obs
        for x_, y_, w_, h_  in self.obstacles:
            if x in range(x_ - w - self.group_rad*2, x_ + w_ + self.group_rad*2 + 1) and \
                y in range(y_ - h - self.group_rad*2, y_ + h_  + self.group_rad*2 + 1):
                return True
        return False

    def get_gridGlobal2(self):
        '''
        - Create global grid map from loading position to dropoff position.
        - Boundary and obstacles are offset with group's radius.
        - Fill boundary with lable -1.
        - Fill obstacle with label -1.
        '''
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
        '''
        - Random loading position within free space.
        - Fill loading position with label 2.
        '''
        while True: 
            x = random.randrange(self.span_x + self.group_rad, self.gridW - self.span_x - self.group_rad)
            y = random.randrange(self.span_y + self.group_rad, self.gridH - self.span_y - self.group_rad)
            if self.grid_global2[y,x] == 0:
                self.grid_global2[y,x] = 2
                self.loading_point = [x,y]
                break

    def create_dropOff_point(self):
        '''
        - Random dropoff position within free space.
        - Distance between loading and dropoff position more than (mapH + mapW)/4.
        - Fill dropoff position with label 1.
        '''
        while True:
            x = random.randrange(self.span_x + self.group_rad, self.gridW - self.span_x - self.group_rad)
            y = random.randrange(self.span_y + self.group_rad, self.gridH - self.span_y - self.group_rad)
            if self.grid_global2[y,x] == 0 and self.cal_distance([x,y], self.loading_point) > (self.mapH + self.mapW)/4:
                self.grid_global2[y,x] = 1
                self.dropOff_point = [x,y]
                break

    def get_gridGlobal1(self):
        '''
        - Create global grid map for all robots from initial positions to goal positions.
        - Boundary and obstacles are offset with robot's radius.
        - Fill boundary with lable -1.
        - Fill obstacle with label -1.
        '''
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
        '''
        For all robots:
        - Create goal positions according to loading point.
        - Create loading positions in real map according to loading point.
        - Create dropoff positions in real map according to dropoff point.
        '''
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
        '''
        - Random robot's initial position within free space.
        - Robots are created in different zones.
        - Distance between each robot must more than 5 x robot's radius.
        '''
        self.starts = []
        # find start position for robot1
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
        '''
        Check distance between each robot more than 5 x robot's radius or not.
        '''
        for start_pos in self.starts:
            if self.cal_distance(point, start_pos) < self.robot_rad*5:
                return False
        return True

    def get_gridSet(self):
        '''
        Create grid map for each robot refer to global grid map.
        '''
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
        '''
        Create path for each robot.
        '''
        self.paths = []
        for k in range(self.n_robots):
            path = [self.robots_pos[k]]
            self.paths.append(path)

    def get_stateSet(self):
        '''
        Create current state for each robot.
        - size: H172 x W300 
        '''
        self.state_set = []
        for k in range(self.n_robots):
            state = self.grid_set[k][self.robots_pos[k][1] - self.span_y: self.robots_pos[k][1] + self.span_y + 1,
                                    self.robots_pos[k][0] - self.span_x: self.robots_pos[k][0] + self.span_x + 1]
            state_resized = cv.resize(state, ((self.span_x*2 + 1)*self.map_ratio, (self.span_y*2 + 1)*self.map_ratio), 
                                    interpolation=cv.INTER_AREA)
            self.state_set.append(state_resized)

    def get_stackSet(self):
        '''
        Create observation for each robot.
        '''
        self.stack_set = []
        stack_set = []
        for k in range(self.n_robots):
            stack = [self.state_set[k].copy()]*self.n_stacks
            self.stack_set.append(stack)
            stack_set.append(np.array(stack))
        return stack_set

    def step_pathPlanning1(self, actions):
        '''
        Apply actions from RL model for path planning.
        For all robots:
        - Update position.
        - Calculate reward.
        - Update terminal status.
        - Update info.
        - Update grid map.
        - Update path.
        - Update observation.
        '''
        new_positions = []
        rewards = []
        terminals = []
        infos = []
        for k in range(self.n_robots):
            new_position = self.robots_pos[k].copy()
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

            dist = self.cal_distance(self.robots_pos[k], self.goals[k]) - self.cal_distance(new_position, self.goals[k])
            line = self.cal_distance(new_position, self.robots_pos[k])
            reward = dist -line
            terminal = False
            info = ''

            if self.reachGoal(new_position, k):
                reward = 0
                terminal = True
                info = 'reach goal'
            elif self.HitObstacle(new_position, self.grid_set[k]):
                reward = -3
                new_position = self.robots_pos[k].copy()
                info = 'hit obstacle'

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
        '''
        Check robot reaches goal or not.
        '''
        if position == self.goals[idx]:
            return True
        else:
            return False

    def HitObstacle(self, position, grid):
        '''
        Check robot hit obstacle or not.
        '''
        if grid[position[1], position[0]] == -1:
            return True
        else:
            return False

    def cal_distance(self, pos1, pos2):
        '''
        Calculate distance between 2 points.
        '''
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x1-x2)**2+(y1-y2)**2)**0.5
    
    def update_gridSet(self, new_positions):
        '''
        For all robots:
        - Updaate global grid map.
        - Update grid map.
        - Update position.
        '''
        for k in range(self.n_robots):
            # update grid global1
            self.grid_global1[self.robots_pos[k][1], self.robots_pos[k][0]] = 0
            self.grid_global1[new_positions[k][1], new_positions[k][0]] = k+11
            # update grid set
            self.grid_set[k][self.robots_pos[k][1], self.robots_pos[k][0]] = 0
            self.grid_set[k][new_positions[k][1], new_positions[k][0]] = 2
            # update robot's positions
            self.robots_pos[k] = new_positions[k]

    def add_paths(self):
        '''
        Add path for all robots.
        '''
        for k in range(self.n_robots):
            if self.robots_pos[k] != self.paths[k][-1]:
                self.paths[k].append(self.robots_pos[k])

    def update_stackSet(self):
        '''
        Update observation of all robots.
        '''
        stack_set = []
        for k in range(self.n_robots):
            self.stack_set[k].pop(0)
            self.stack_set[k].append(self.state_set[k].copy())
            stack_set.append(np.array(self.stack_set[k]))
        return stack_set

    def render_pathPlanning1(self):
        '''
        Render image displaying robots' curent position and path with ratio 1:25.
        '''
        img = np.zeros((self.gridH, self.gridW, 3), dtype=np.uint8)
        for i in range(self.gridW):
            for j in range(self.gridH):
                # filled color obstacles
                if self.grid_global1[j, i] == -1:
                    img[j, i] = np.array(self.obstacle_color)
                # filled color loading points
                elif self.grid_global1[j, i] in range(1, self.n_robots+1):
                    img[j, i] = np.array(self.loading_color)
                # filled color dropOff points
                elif self.grid_global1[j, i] in range(21, self.n_robots+21):
                    img[j, i] = np.array(self.dropOff_color)
                # filled color floor
                else:
                    img[j, i] = np.array(self.floor_color)
                # filled color robot's positions
                for k in range(self.n_robots):
                    if self.grid_global1[j, i] == k + 11:
                        img[j, i] = np.array(self.robots_color[k])
                
        img_resized = cv.resize(img, (self.gridW*self.img_ratio, self.gridH*self.img_ratio), interpolation=cv.INTER_AREA)

        # draw path
        for k in range(self.n_robots):
            for i, [x,y] in enumerate(self.paths[k]):
                x_point = x*self.img_ratio + self.img_ratio//2
                y_point = y*self.img_ratio + self.img_ratio//2
                cv.circle(img_resized, (x_point, y_point), 4, self.paths_color[k], thickness = -1)
                if i > 0:
                    cv.line(img_resized, (x_prev, y_prev), (x_point, y_point), self.paths_color[k], thickness = 2)
                x_prev = x_point
                y_prev = y_point

        img_croped = img_resized[self.span_y*self.img_ratio: (self.gridH - self.span_y)*self.img_ratio,
                                self.span_x*self.img_ratio: (self.gridW - self.span_x)*self.img_ratio]

        img_resized = cv.resize(img_croped, (img_croped.shape[1]//2, img_croped.shape[0]//2))
        cv.imshow('Moibile Robot Path Planning I', img_resized)
        cv.waitKey(1)

        return img_resized
    
    # ================================================================= #
    #       Step 2: Motion Control from start to loading position       #
    # ================================================================= #

    def reset_motionControl(self):
        '''
        - Transform paths from previous session to real map size.
        - Create initial map for path following task.
        - Set waypoints.
        - Transform robots' initial position to real map size.
        - Set inital heading of robots.
        '''
        # Scale up map and paths
        self.transformed_paths = self.transform_path()
        self.init_Map = self.transform_map1()

        # Reset PID variables 
        self.prev_delta_x = [0]*self.n_robots
        self.prev_delta_theta = [0]*self.n_robots
        self.accu_delta_x = [0]*self.n_robots
        self.accu_delta_theta = [0]*self.n_robots

        # Dynamic variables
        self.waypoints = [1]*self.n_robots
        self.current_pos = self.transform_point(self.starts)
        self.current_theta = self.create_theta()
        self.adjust_heading = [False]*self.n_robots
        self.terminals = [False]*self.n_robots

    def transform_path(self):
        '''
        - Transform path from grid map to real map.
        - Add waypoints within path with ration 1:3.
        '''
        transformed_paths = []
        for k in range(self.n_robots):
            new_path = []
            for i in range(len(self.paths[k])):
                x = (self.paths[k][i][0] - self.span_x)*self.img_ratio + self.img_ratio//2
                y = (self.paths[k][i][1] - self.span_y)*self.img_ratio + self.img_ratio//2
                new_path.append([x,y])
            new_path = self.increase_waypoint(new_path, 3)
            new_path.append(self.img_load_pts[k])
            transformed_paths.append(new_path)
        return transformed_paths
    
    def increase_waypoint(self, path, n):
        '''
        Increasing waypoints within path with ration 1:n.
        '''
        new_path = []
        for i in range(len(path) - 1):
            delta_x = path[i+1][0] - path[i][0]
            delta_y = path[i+1][1] - path[i][1]
            for j in range(n):
                new_path.append([path[i][0] + delta_x*j//n, path[i][1] + delta_y*j//n])
        new_path.append(path[-1])
        return new_path
    
    def transform_map1(self):
        '''
        - Transform grid map to real map for path following task with ration 1:50.
        - Add path into this map.
        '''
        MapW = self.gridW*self.img_ratio
        MapH = self.gridH*self.img_ratio
        grid = np.zeros((self.gridH, self.gridW, 3), dtype=np.uint8)
        for i in range(self.gridW):
            for j in range(self.gridH):
                # filled color goal's positions
                if self.grid_global1[j, i] in range(1, self.n_robots+1):
                    grid[j, i] = np.array(self.loading_color)
                # filled color dropoff's positions
                elif self.grid_global1[j, i] in range(21, self.n_robots+21):
                    grid[j, i] = np.array(self.dropOff_color)
                # filled color floor
                else:
                    grid[j, i] = np.array(self.floor_color)
        # filled color obstacles
        for x, y, w, h in self.obstacles:
            grid[y:y+h, x:x+w] = np.array(self.obstacle_color)

        # scale up grid to map
        init_Map = cv.resize(grid, (MapW, MapH), interpolation=cv.INTER_AREA)

        # crop map to be without span
        init_Map = init_Map[self.span_y*self.img_ratio: (self.gridH - self.span_y)*self.img_ratio,
                            self.span_x*self.img_ratio: (self.gridW - self.span_x)*self.img_ratio]
        
        # draw path
        for k in range(self.n_robots):
            for i, [x,y] in enumerate(self.transformed_paths[k]):
                cv.circle(init_Map, (x, y), 4, self.paths_color[k], thickness = -1)
                if i > 0:
                    cv.line(init_Map, (x_prev, y_prev), (x, y), self.paths_color[k], thickness = 2)
                x_prev = x
                y_prev = y
        
        return init_Map

    def transform_point(self, points):
        '''
        Transform point in grid map to real map.
        '''
        transformed_points = []
        for i in range(len(points)):
            x = (points[i][0] - self.span_x)*self.img_ratio + self.img_ratio//2
            y = (points[i][1] - self.span_y)*self.img_ratio + self.img_ratio//2
            transformed_points.append([x, y])
        return transformed_points
        
    def create_theta(self):
        '''
        Random robots' heading at initial state.
        '''
        robots_theta = []
        for k in range(self.n_robots):
            theta = np.random.random()*math.pi*2
            robots_theta.append(theta)
        return robots_theta
    
    def get_state1(self):
        '''
        For all robots:
        - Create observations.
        - Calculate delta_d_before.
        '''
        states = []
        self.prev_errs = []
        for k in range(self.n_robots):
            delta_x , delta_y = self.transform_local(self.transformed_paths[k][self.waypoints[k]], 
                                                    self.current_pos[k], self.current_theta[k])
            delta_d = (delta_x**2 + delta_y**2)**0.5
            if delta_d == 0:
                delta_theta = 0
            else:
                delta_theta = math.acos(delta_x/delta_d)
                if delta_y < 0:
                    delta_theta *= -1

            self.prev_errs.append(delta_d)
            states.append(np.array([delta_x, delta_theta]))

        return states

    def robot_movePID(self, dt):
        '''
        For all robots:
        - Calculate robot's speed using PID control.
        - Update robot's position and heading.
        - Update waypoints.
        - Update terminal status.
        '''
        for k in range(self.n_robots):
            if self.terminals[k]:
                ul = 0
                ur = 0
            else:
                ul, ur = self.update_speedPID(k, dt)
            u = (ul + ur)/2
            w = (ul - ur)/self.robot_width
            self.current_pos[k][0] += int(round(u*dt*math.cos(self.current_theta[k] + w*dt/2), 0))
            self.current_pos[k][1] += int(round(u*dt*math.sin(self.current_theta[k] + w*dt/2), 0))
            self.current_theta[k] += w*dt
            if self.cal_distance(self.current_pos[k], self.transformed_paths[k][self.waypoints[k]]) < 50 and \
                self.waypoints[k] < len(self.transformed_paths[k]) - 1:
                self.waypoints[k] += 1

            self.terminals[k] = self.isInPosition(self.current_pos[k], self.img_load_pts[k], 3)
    
    def update_speedPID(self, idx, dt):
        '''
        - Caculate robot's linear and angular velocities using PID control.
        - Check adjust_heading.
        '''
        delta_x , delta_y = self.transform_local(self.transformed_paths[idx][self.waypoints[idx]], 
                                                    self.current_pos[idx], self.current_theta[idx])
        delta_d = (delta_x**2 + delta_y**2)**0.5
        if delta_d == 0:
            delta_theta = 0
        else:
            delta_theta = math.acos(delta_x/delta_d)
            if delta_y < 0:
                delta_theta *= -1
        
        self.accu_delta_x[idx] += delta_x*dt
        self.accu_delta_theta[idx] += delta_theta*dt

        u = self.u_KP*delta_x + \
            self.u_KI*self.accu_delta_x[idx] + \
            self.u_KD*(delta_x - self.prev_delta_x[idx])/dt
        w = self.w_KP*delta_theta + \
            self.w_KI*self.accu_delta_theta[idx] + \
            self.w_KD*(delta_theta - self.prev_delta_theta[idx])/dt
            
        self.prev_delta_x[idx] = delta_x
        self.prev_delta_theta[idx] = delta_theta

        if abs(delta_theta) < math.pi/9:
            self.adjust_heading[idx] = True
        if abs(delta_theta) < math.pi/3 and False not in self.adjust_heading:
            pass
        else:
            u = 0
        ul = u + self.robot_width*w/2
        ur = u - self.robot_width*w/2

        return ul, ur

    def transform_local(self, point, local_point, local_theta):
        '''
        Transform point in global coordinate to robot coordinate.
        '''
        point = np.array(point)
        local_point = np.array(local_point)
        rot_mat = np.array([[math.cos(local_theta), math.sin(local_theta)], 
                            [-math.sin(local_theta), math.cos(local_theta)]])
        transformed_point = rot_mat @ (point - local_point)
        return transformed_point

    def step_motionControl(self, actions, dt):
        '''
        Apply actions from agent.
        For all robots:
        - Update robot's position and heading.
        - Calculate delta_d_after.
        - Calculate reward.
        - Update terminal status.
        - Update info.
        - Update waypoint.
        '''
        states = []
        rewards = []
        infos = []
        for k in range(self.n_robots):
            if self.terminals[k]:
                u = 0
                w = 0
            else:
                u, w = actions[k]
            self.current_pos[k][0] += int(round(u*dt*math.cos(self.current_theta[k] + w*dt/2), 0))
            self.current_pos[k][1] += int(round(u*dt*math.sin(self.current_theta[k] + w*dt/2), 0))
            self.current_theta[k] += w*dt

            delta_x , delta_y = self.transform_local(self.transformed_paths[k][self.waypoints[k]], 
                                                    self.current_pos[k], self.current_theta[k])
            delta_d = (delta_x**2 + delta_y**2)**0.5
            if delta_d == 0:
                delta_theta = 0
            else:
                delta_theta = math.acos(delta_x/delta_d)
                if delta_y < 0:
                    delta_theta *= -1

            err = delta_d
            reward = self.prev_errs[k] - err
            terminal = False
            info = ''

            if self.isInPosition(self.current_pos[k], self.img_load_pts[k], 3):
                terminal = True
                info = 'reach goal'
            elif err > self.robot_width:
                terminal = True
                info = 'out of path'

            states.append(np.array([delta_d, delta_theta]))
            rewards.append(reward)
            self.terminals[k] = terminal
            infos.append(info)

            if self.cal_distance(self.current_pos[k], self.transformed_paths[k][self.waypoints[k]]) < 50 and \
                self.waypoints[k] < len(self.transformed_paths[k]) - 1:
                self.waypoints[k] += 1

        return states, rewards, infos

    def isInPosition(self, pos1, pos2, threshold):
        '''
        Check point1 reaches point2 or not.
        '''
        error = self.cal_distance(pos1, pos2)
        # print(error)
        if error < threshold:
            return True
        else:
            return False

    def rotate_rectangle(self, pos, theta):
        '''
        Draw robot's wheels.
        '''
        pt0 = [pos[0]-20, pos[1]-60]
        pt1 = [pos[0]+20, pos[1]-60]
        pt2 = [pos[0]+20, pos[1]+60]
        pt3 = [pos[0]-20, pos[1]+60]

        # Point 0
        rotated_x = math.cos(theta)*(pt0[0] - pos[0]) - math.sin(theta)*(pt0[1] - pos[1]) + pos[0]
        rotated_y = math.sin(theta)*(pt0[0] - pos[0]) + math.cos(theta)*(pt0[1] - pos[1]) + pos[1]
        point_0 = [rotated_x, rotated_y]

        # Point 1
        rotated_x = math.cos(theta)*(pt1[0] - pos[0]) - math.sin(theta)*(pt1[1] - pos[1]) + pos[0]
        rotated_y = math.sin(theta)*(pt1[0] - pos[0]) + math.cos(theta)*(pt1[1] - pos[1]) + pos[1]
        point_1 = [rotated_x, rotated_y]

        # Point 2
        rotated_x = math.cos(theta)*(pt2[0] - pos[0]) - math.sin(theta)*(pt2[1] - pos[1]) + pos[0]
        rotated_y = math.sin(theta)*(pt2[0] - pos[0]) + math.cos(theta)*(pt2[1] - pos[1]) + pos[1]
        point_2 = [rotated_x, rotated_y]

        # Point 3
        rotated_x = math.cos(theta)*(pt3[0] - pos[0]) - math.sin(theta)*(pt3[1] - pos[1]) + pos[0]
        rotated_y = math.sin(theta)*(pt3[0] - pos[0]) + math.cos(theta)*(pt3[1] - pos[1]) + pos[1]
        point_3 = [rotated_x, rotated_y]

        return np.array([point_0, point_1, point_2, point_3], dtype=np.int32)
        
    def render_motionControl(self):
        '''
        For all robots:
        - Render image displaying robot's curent position and heading on real map with ratio 2:1.
        '''
        Map = self.init_Map.copy()

        # draw robots with heading
        for k in range(self.n_robots):
            # draw robot position
            pts = self.rotate_rectangle(self.current_pos[k], self.current_theta[k])
            pts = pts.reshape((-1, 1, 2))
            cv.polylines(Map, [pts], True, self.wheel_color, thickness=20)
            cv.circle(Map, self.current_pos[k], self.robot_width//2, self.robots_color[k], thickness=-1)

            # draw heading
            heading_pt = [int(self.current_pos[k][0] + self.axis_length*math.cos(self.current_theta[k])), 
                            int(self.current_pos[k][1] + self.axis_length*math.sin(self.current_theta[k]))]
            cv.line(Map, self.current_pos[k], heading_pt, self.robots_color[k], thickness=3)

            # draw reference point on path
            cv.circle(Map, self.transformed_paths[k][self.waypoints[k]], 10, self.paths_color[k], thickness=-1)
        
        # draw loading positions
        for x, y in self.img_load_pts:
            cv.circle(Map, (x,y), self.robot_width//2, self.loading_color, thickness=3)
            
        Map_resized = cv.resize(Map, (Map.shape[1]//2, Map.shape[0]//2))
        cv.imshow('Mobile Robot Motion Control', Map_resized)
        cv.waitKey(60)

        return Map_resized

    # ================================================================= #
    #      Step 3: Path planning from loading to dropOff position       #
    # ================================================================= #

    def reset_pathPlanning2(self):
        '''
        Reset map to find path from loading point to dropoff point.
        - Consider virtual robot at center of group of robots.
        - Reset path of virtual robot.
        '''
        self.virtual_pos = self.loading_point.copy()
        self.virtual_path = [self.virtual_pos.copy()]
        state = self.get_state(self.grid_global2, self.virtual_pos)
        self.virtual_stack = [state]*self.n_stacks
        return np.array(self.virtual_stack)

    def get_state(self, grid, pos):
        '''
        Create current state of virtual robot.
        - size: H172 x W300 
        '''
        state = grid[pos[1] - self.span_y: pos[1] + self.span_y + 1,
                    pos[0] - self.span_x: pos[0] + self.span_x + 1]
        state_resized = cv.resize(state, ((self.span_x*2 + 1)*self.map_ratio, (self.span_y*2 + 1)*self.map_ratio), 
                                    interpolation=cv.INTER_AREA)
        return state_resized

    def step_pathPlanning2(self, action):
        '''
        Apply action from RL model for path planning.
        - Update virtual robot's position.
        - Calculate reward.
        - Update terminal status.
        - Update info.
        - Update grid map.
        - Update path
        - Update observation
        '''
        new_position = self.virtual_pos.copy()
        if action == 0: # move up-left
            new_position[0] -= 1
            new_position[1] -= 1
        elif action == 1: # move up
            new_position[1] -= 1
        elif action == 2: # move up-right
            new_position[0] += 1
            new_position[1] -= 1
        elif action == 3: # move left
            new_position[0] -= 1
        elif action == 4: # move right
            new_position[0] += 1
        elif action == 5: # move down-left
            new_position[0] -= 1
            new_position[1] += 1
        elif action == 6: # move down
            new_position[1] += 1
        elif action == 7: # move down-right
            new_position[0] += 1
            new_position[1] += 1

        dist = self.cal_distance(self.virtual_pos, self.dropOff_point) - self.cal_distance(new_position, self.dropOff_point)
        line = self.cal_distance(new_position, self.virtual_pos)
        reward = dist -line
        terminal = False
        info = ''

        if self.reachDropOff(new_position):
            reward = 0
            terminal = True
            info = 'reach dropOff point'
        elif self.HitObstacle(new_position, self.grid_global2):
            reward = -3
            new_position = self.virtual_pos.copy()
            info = 'hit obstacle'
            
        self.grid_global2[self.virtual_pos[1], self.virtual_pos[0]] = 0
        self.grid_global2[new_position[1], new_position[0]] = 2
        self.virtual_pos = new_position
        self.virtual_path.append(new_position)
        state = self.get_state(self.grid_global2, self.virtual_pos)
        self.virtual_stack.pop(0)
        self.virtual_stack.append(state)

        return np.array(self.virtual_stack), reward, terminal, info

    def reachDropOff(self, position):
        '''
        Check virtual robot reaches dropoff point or not.
        '''
        if position == self.dropOff_point:
            return True
        else:
            return False

    def render_pathPlanning2(self):
        '''
        Render image displaying virtual robot's curent position and path with ratio 1:25.
        '''
        img = np.zeros((self.gridH, self.gridW, 3), dtype=np.uint8)
        for i in range(self.gridW):
            for j in range(self.gridH):
                # filled color obstacles
                if self.grid_global2[j, i] == -1:
                    img[j, i] = np.array(self.obstacle_color)
                # filled color dropOff point
                elif self.grid_global2[j, i] == 1:
                    img[j, i] = np.array(self.dropOff_color)
                # filled color virtual position
                elif self.grid_global2[j, i] == 2:
                    img[j, i] = np.array(self.loading_color)
                # filled color floor
                else:
                    img[j, i] = np.array(self.floor_color)
        
        img_resized = cv.resize(img, (self.gridW*self.img_ratio, self.gridH*self.img_ratio), interpolation=cv.INTER_AREA)

        # draw path
        for i, [x,y] in enumerate(self.virtual_path):
            x_point = x*self.img_ratio + self.img_ratio//2
            y_point = y*self.img_ratio + self.img_ratio//2
            cv.circle(img_resized, (x_point, y_point), 4, self.path_center_color, thickness = -1)
            if i > 0:
                cv.line(img_resized, (x_prev, y_prev), (x_point, y_point), self.path_center_color, thickness = 2)
            x_prev = x_point
            y_prev = y_point

        img_croped = img_resized[self.span_y*self.img_ratio: (self.gridH - self.span_y)*self.img_ratio,
                                self.span_x*self.img_ratio: (self.gridW - self.span_x)*self.img_ratio]
    
        img_resized = cv.resize(img_croped, (img_croped.shape[1]//2, img_croped.shape[0]//2))
        cv.imshow('Moibile Robot Path Planning II', img_resized)
        cv.waitKey(1)

        return img_resized
    
    # ================================================================= #
    #     Step 4: Formation Control from loading to dropOff position    #
    # ================================================================= #

    def reset_formationControl(self):
        '''
        - Transform path from previous session to real map size.
        - Create initial map for formaiton control task.
        - Set waypoints of virtual robot.
        - Transform dropoff point from grid map to goal position of virtual robot.
        - Transform loading point from grid map to start position of virtual robot.
        - Set inital heading of virtual robot according to path.
        - Calculate formation error at initial state.
        '''
        # scale up map and paths
        self.transformed_virtual_path = self.transform_virtual_path()
        self.init_Map = self.transform_map2()

        # reset PID variables for virtual
        self.Vprev_delta_x = 0
        self.Vprev_delta_theta = 0
        self.Vaccu_delta_x = 0
        self.Vaccu_delta_theta = 0

        # dynamic variables for virtual
        self.virtual_waypoint = 1
        self.virtual_goal = self.transform_point([self.dropOff_point])[0]
        self.virtual_start = self.transform_point([self.loading_point])[0]
        self.virtual_pos = self.virtual_start.copy()
        self.virtual_theta = self.get_virtual_theta()

        # reset PID variables for followers
        self.prev_delta_x = [0]*self.n_robots
        self.prev_delta_theta = [0]*self.n_robots
        self.accu_delta_x = [0]*self.n_robots
        self.accu_delta_theta = [0]*self.n_robots

        # dynamic variables for followers
        self.adjust_heading = [False]*self.n_robots
        self.terminals = [False]*(self.n_robots + 1)
        self.get_error()

    def transform_virtual_path(self):
        '''
        - Transform path from grid map to real map.
        - Add waypoints within path with ration 1:3.
        '''
        transformed_virtual_path = []
        for i in range(len(self.virtual_path)):
            x = (self.virtual_path[i][0] - self.span_x)*self.img_ratio + self.img_ratio//2
            y = (self.virtual_path[i][1] - self.span_y)*self.img_ratio + self.img_ratio//2
            transformed_virtual_path.append([x,y])
        transformed_virtual_path = self.increase_waypoint(transformed_virtual_path, 3)
        return transformed_virtual_path
    
    def transform_map2(self):
        '''
        - Transform grid map to real map for formation control task with ration 1:50.
        - Add path into this map.
        '''
        MapW = self.gridW*self.img_ratio
        MapH = self.gridH*self.img_ratio
        grid = np.zeros((self.gridH, self.gridW, 3), dtype=np.uint8)
        for i in range(self.gridW):
            for j in range(self.gridH):
                # filled color dropOff point
                if self.grid_global2[j, i] == 1:
                    grid[j, i] = np.array(self.dropOff_color)
                # filled color loading point
                elif self.grid_global2[j, i] == 2:
                    grid[j, i] = np.array(self.dropOff_color)
                # filled color floor
                else:
                    grid[j, i] = np.array(self.floor_color)
        # filled color obstacles
        for x, y, w, h in self.obstacles:
            grid[y:y+h, x:x+w] = np.array(self.obstacle_color)

        # scale up grid to map
        init_Map = cv.resize(grid, (MapW, MapH), interpolation=cv.INTER_AREA)

        # crop map to be without span
        init_Map = init_Map[self.span_y*self.img_ratio: (self.gridH - self.span_y)*self.img_ratio,
                            self.span_x*self.img_ratio: (self.gridW - self.span_x)*self.img_ratio]
        
        # draw path
        for i, [x,y] in enumerate(self.transformed_virtual_path):
            cv.circle(init_Map, (x, y), 4, self.path_center_color, thickness = -1)
            if i > 0:
                cv.line(init_Map, (x_prev, y_prev), (x, y), self.path_center_color, thickness = 2)
            x_prev = x
            y_prev = y
    
        return init_Map

    def get_virtual_theta(self):
        '''
        Create virtual robot's heading at initial state refer to path.
        '''
        delta_x = self.transformed_virtual_path[1][0] - self.transformed_virtual_path[0][0]
        delta_y = self.transformed_virtual_path[1][1] - self.transformed_virtual_path[0][1]
        delta_d = (delta_x**2 + delta_y**2)**0.5
        theta = math.acos(delta_x/delta_d)
        if delta_y < 0:
            theta = 2*math.pi - theta
        return theta
    
    def get_error(self):
        '''
        Calculate formation error at initial state.
        '''
        self.ref_dist = []
        self.errors = []
        ref_points = self.img_load_pts.copy() + [self.img_load_pts[0].copy()]
        current_points = self.current_pos.copy() + [self.current_pos[0].copy()]
        for k in range(self.n_robots):
            ref_dist = self.cal_distance(ref_points[k], ref_points[k+1])
            dist = self.cal_distance(current_points[k], current_points[k+1])
            error = dist - ref_dist
            self.ref_dist.append(ref_dist)
            self.errors.append([error])

    def get_state2(self, dt):
        '''
        Virtual robot moves using PID control.
        For all robots:
        - Create observation for agent using reference point of each robot.
        - Calculate delta_d_before.
        '''
        self.virtual_movePID(dt)
        states = []
        self.prev_errs = []
        for k in range(self.n_robots):
            delta_x, delta_y = self.transform_local(self.ref_points[k], self.current_pos[k], self.current_theta[k])
            delta_d = (delta_x**2 + delta_y**2)**0.5
            if delta_d == 0:
                delta_theta = self.virtual_theta - self.current_theta[k]
                if delta_theta > math.pi:
                    delta_theta -= 2*math.pi
                elif delta_theta < -math.pi:
                    delta_theta += 2*math.pi
            else:
                delta_theta = math.acos(delta_x/delta_d)
                if delta_y < 0:
                    delta_theta *= -1

            self.prev_errs.append(delta_d)
            states.append(np.array([delta_x, delta_theta]))
        return states

    def virtual_movePID(self, dt):
        '''
        Virtual robot moves follow path using PID control.
        - Calculate linear and angular speeds of virtual robot.
        - Update virtual robot's position and heading.
        - Update reference points according to virtual robot's position.
        - Update virtual robot's waypoint.
        - Update virtual robot's terminal status.
        '''
        if self.terminals[-1]:
            ul = 0
            ur = 0
        else:
            ul, ur = self.update_speedVirtual(dt)
        u = (ul + ur)/2
        w = (ul - ur)/self.robot_width
        self.virtual_pos[0] += int(round(u*dt*math.cos(self.virtual_theta + w*dt/2), 0))
        self.virtual_pos[1] += int(round(u*dt*math.sin(self.virtual_theta + w*dt/2), 0))
        self.virtual_theta += w*dt
        self.ref_points = []
        for k in range(self.n_robots):
            delta_x = self.img_load_pts[k][0] - self.virtual_start[0]
            delta_y = self.img_load_pts[k][1] - self.virtual_start[1]
            self.ref_points.append([self.virtual_pos[0] + delta_x, self.virtual_pos[1] + delta_y])
        if self.cal_distance(self.virtual_pos, self.transformed_virtual_path[self.virtual_waypoint]) < 50 and \
            self.virtual_waypoint < len(self.transformed_virtual_path) - 1:
            self.virtual_waypoint += 1
        self.terminals[-1] = self.isInPosition(self.virtual_pos, self.virtual_goal, 3)
    
    def update_speedVirtual(self, dt):
        '''
        Calculate virtual robot's linear and angular speeds using PID control.
        '''
        delta_x, delta_y = self.transform_local(self.transformed_virtual_path[self.virtual_waypoint],
                                                self.virtual_pos, self.virtual_theta)
        delta_d = (delta_x**2 + delta_y**2)**0.5
        if delta_d == 0:
            delta_theta = 0
        else:
            delta_theta = math.acos(delta_x/delta_d)
            if delta_y < 0:
                delta_theta *= -1
        
        self.Vaccu_delta_x += delta_x*dt
        self.Vaccu_delta_theta += delta_theta*dt

        u = self.u_KP*delta_x + \
            self.u_KI*self.Vaccu_delta_x + \
            self.u_KD*(delta_x - self.Vprev_delta_x)/dt
        w = self.w_KP*delta_theta + \
            self.w_KI*self.Vaccu_delta_theta + \
            self.w_KD*(delta_theta - self.Vprev_delta_theta)/dt

        self.Vprev_delta_x = delta_x
        self.Vprev_delta_theta = delta_theta
        
        if abs(delta_theta) < math.pi/3 and False not in self.adjust_heading:
            pass
        else:
            u = 0
        ul = u + self.robot_width*w/2
        ur = u - self.robot_width*w/2

        return ul, ur

    def follower_movePID(self, dt):
        '''
        For all robots:
        - Calculate robot's speed using PID control.
        - Update robot's position and heading.
        - Update robot's terminal status.
        - Update formation error.
        '''
        for k in range(self.n_robots):
            if self.terminals[k]:
                ul = 0
                ur = 0
            else:
                ul, ur = self.update_speedFollower(k, dt)
            u = (ul + ur)/2
            w = (ul - ur)/self.robot_width
            self.current_pos[k][0] += int(round(u*dt*math.cos(self.current_theta[k] + w*dt/2), 0))
            self.current_pos[k][1] += int(round(u*dt*math.sin(self.current_theta[k] + w*dt/2), 0))
            self.current_theta[k] += w*dt
            if self.terminals[-1]:
                self.terminals[k] = self.isInPosition(self.current_pos[k], self.ref_points[k], 3)
        self.update_error()
    
    def update_speedFollower(self, idx, dt):
        '''
        Calculate robot's linear and angular speeds using PID control.
        '''
        delta_x, delta_y = self.transform_local(self.ref_points[idx], self.current_pos[idx], self.current_theta[idx])
        delta_d = (delta_x**2 + delta_y**2)**0.5
        if delta_d < 5:
            delta_theta = self.virtual_theta - self.current_theta[idx]
            if delta_theta > math.pi:
                delta_theta -= 2*math.pi
            elif delta_theta < -math.pi:
                delta_theta += 2*math.pi
        else:
            delta_theta = math.acos(delta_x/delta_d)
            if delta_y < 0:
                delta_theta *= -1
        
        self.accu_delta_x[idx] += delta_x*dt
        self.accu_delta_theta[idx] += delta_theta*dt

        u = self.Fu_KP*delta_x + \
            self.Fu_KI*self.accu_delta_x[idx] + \
            self.Fu_KD*(delta_x - self.prev_delta_x[idx])/dt
        w = self.Fw_KP*delta_theta + \
            self.Fw_KI*self.accu_delta_theta[idx] + \
            self.Fw_KD*(delta_theta - self.prev_delta_theta[idx])/dt

        self.prev_delta_x[idx] = delta_x
        self.prev_delta_theta[idx] = delta_theta

        if abs(delta_theta) < math.pi/9:
            self.adjust_heading[idx] = True
        if abs(delta_theta) < math.pi/3 and False not in self.adjust_heading:
            pass
        else:
            u = 0
        ul = u + self.robot_width*w/2
        ur = u - self.robot_width*w/2

        return ul, ur
    
    def update_error(self):
        '''
        Update formation error in each time step.
        '''
        current_points = self.current_pos.copy() + [self.current_pos[0].copy()]
        for k in range(self.n_robots):
            dist = self.cal_distance(current_points[k], current_points[k+1])
            error = dist - self.ref_dist[k]
            self.errors[k].append(error)
        
    def step_formationControl(self, actions, dt):
        '''
        Apply actions from agent.
        For all robots:
        - Update robot's position and heading.
        - Calculate delta_d_after.
        - Calculate reward.
        - Update terminal status.
        - Update info.
        - Update formation error.
        '''
        states = []
        rewards = []
        infos = []
        for k in range(self.n_robots):
            if self.terminals[k]:
                u = 0
                w = 0
            else:
                u, w = actions[k]
            self.current_pos[k][0] += int(round(u*dt*math.cos(self.current_theta[k] + w*dt/2), 0))
            self.current_pos[k][1] += int(round(u*dt*math.sin(self.current_theta[k] + w*dt/2), 0))
            self.current_theta[k] += w*dt

            delta_x , delta_y = self.transform_local(self.ref_points[k], self.current_pos[k], self.current_theta[k])
            delta_d = (delta_x**2 + delta_y**2)**0.5
            if delta_d == 0:
                delta_theta = self.virtual_theta - self.current_theta[k]
                if delta_theta > math.pi:
                    delta_theta -= 2*math.pi
                elif delta_theta < -math.pi:
                    delta_theta += 2*math.pi
            else:
                delta_theta = math.acos(delta_x/delta_d)
                if delta_y < 0:
                    delta_theta *= -1

            err = delta_d
            reward = self.prev_errs[k] - err - err/10
            terminal = False
            info = ''

            if self.isInPosition(self.current_pos[k], self.ref_points[k], 3):
                if self.terminals[-1]:
                    terminal = True
                    info = 'reach goal'
            elif err > self.robot_width:
                terminal = True
                info = 'fomation collapse'

            states.append(np.array([delta_d, delta_theta]))
            rewards.append(reward)
            self.terminals[k] = terminal
            infos.append(info)
        self.update_error()

        return states, rewards, infos

    def render_formationControl(self):
        '''
        Render image displaying the curent position and heading of virtual robot and robots
        on real map with ratio 2:1.
        '''
        Map = self.init_Map.copy()

        # draw virtual position
        cv.circle(Map, self.virtual_pos, self.robot_width//6, self.loading_color, thickness=-1)

        for k in range(self.n_robots):
            # draw robot position
            pts = self.rotate_rectangle(self.current_pos[k], self.current_theta[k])
            pts = pts.reshape((-1, 1, 2))
            cv.polylines(Map, [pts], True, self.wheel_color, thickness=20)
            cv.circle(Map, self.current_pos[k], self.robot_width//2, self.robots_color[k], thickness=-1)
            
            # draw heading
            heading_pt = [int(self.current_pos[k][0] + self.axis_length*math.cos(self.current_theta[k])), 
                            int(self.current_pos[k][1] + self.axis_length*math.sin(self.current_theta[k]))]
            cv.line(Map, self.current_pos[k], heading_pt, self.robots_color[k], thickness=3)

            # draw reference point and connection line
            delta_x = self.img_load_pts[k][0] - self.virtual_start[0]
            delta_y = self.img_load_pts[k][1] - self.virtual_start[1]
            point = [self.virtual_pos[0] + delta_x, self.virtual_pos[1] + delta_y]
            cv.circle(Map, point, self.robot_width//2, self.loading_color, thickness=3)
            cv.line(Map, point, self.virtual_pos, self.loading_color, thickness=2)
        
        # draw distance between robots
        points = self.current_pos + [self.current_pos[0]]
        for k in range(self.n_robots):
            cv.line(Map, points[k], points[k+1], self.err_color[k], thickness=3)
        
        # draw dropOff positions
        for x, y in self.img_drop_pts:
            cv.circle(Map, (x,y), self.robot_width//2, self.dropOff_color, thickness=3)

        Map_resized = cv.resize(Map, (Map.shape[1]//2, Map.shape[0]//2))
        cv.imshow('Mobile Robot Formation Control', Map_resized)
        cv.waitKey(60)

        return Map_resized