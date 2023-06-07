# import pygame
# import numpy as np

# # Environment constants
# GRID_SIZE = 4
# CELL_SIZE = 100
# WIN_SIZE = GRID_SIZE * CELL_SIZE

# # Colors
# BLACK = (0, 0, 0)
# WHITE = (255, 255, 255)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)
# BLUE = (0, 0, 255)

# # Define the environment grid with obstacles and goal
# def initialize_grid():
#     grid = np.zeros((GRID_SIZE, GRID_SIZE))
#     num_obstacles = np.random.randint(2, 4)  # Random number of obstacles (2 or 3)
#     for _ in range(num_obstacles):
#         while True:
#             i = np.random.randint(0, GRID_SIZE)
#             j = np.random.randint(0, GRID_SIZE)
#             if grid[i, j] != -1:
#                 grid[i, j] = -1  # Obstacle
#                 break
#     while True:
#         i = np.random.randint(0, GRID_SIZE)
#         j = np.random.randint(0, GRID_SIZE)
#         if grid[i, j] == 0:
#             grid[i, j] = 1   # Goal
#             break
#     return grid

# grid = initialize_grid()

# # Initialize the agent's position
# agent_pos = (0, 0)


# def draw_grid():
#     for i in range(GRID_SIZE):
#         for j in range(GRID_SIZE):
#             rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
#             if grid[i, j] == -1:  # Obstacle
#                 pygame.draw.rect(screen, RED, rect)
#             elif grid[i, j] == 1:  # Goal
#                 pygame.draw.rect(screen, GREEN, rect)
#             elif agent_pos == (i, j):  # Agent
#                 pygame.draw.rect(screen, BLUE, rect)
#             else:  # Empty cell
#                 pygame.draw.rect(screen, WHITE, rect, 1)


# def move_agent(action):
#     global agent_pos

#     if action == 0:  # Up
#         new_pos = (agent_pos[0] - 1, agent_pos[1])
#     elif action == 1:  # Down
#         new_pos = (agent_pos[0] + 1, agent_pos[1])
#     elif action == 2:  # Left
#         new_pos = (agent_pos[0], agent_pos[1] - 1)
#     elif action == 3:  # Right
#         new_pos = (agent_pos[0], agent_pos[1] + 1)
#     else:
#         return

#     if (
#         new_pos[0] >= 0
#         and new_pos[0] < GRID_SIZE
#         and new_pos[1] >= 0
#         and new_pos[1] < GRID_SIZE
#         and grid[new_pos] != -1
#     ):
#         agent_pos = new_pos


# # Initialize Pygame
# pygame.init()
# screen = pygame.display.set_mode((WIN_SIZE, WIN_SIZE))
# pygame.display.set_caption("Grid Environment")
# clock = pygame.time.Clock()

# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_UP:
#                 move_agent(0)
#             elif event.key == pygame.K_DOWN:
#                 move_agent(1)
#             elif event.key == pygame.K_LEFT:
#                 move_agent(2)
#             elif event.key == pygame.K_RIGHT:
#                 move_agent(3)

#     screen.fill(BLACK)
#     draw_grid()
#     pygame.display.flip()
#     clock.tick(60)

# pygame.quit()

import numpy as np
import gym
from gym import spaces
import pygame

# Environment constants
GRID_SIZE = 4
CELL_SIZE = 100
WIN_SIZE = GRID_SIZE * CELL_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Load images
agent_img = pygame.image.load('agent.png')
obstacle_img = pygame.image.load('hole.jpg')
goal_img = pygame.image.load('goal.jpg')
empty_img = pygame.image.load('emptycell.jpg')
agent_img = pygame.transform.scale(agent_img, (CELL_SIZE, CELL_SIZE))
obstacle_img = pygame.transform.scale(obstacle_img, (CELL_SIZE, CELL_SIZE))
goal_img = pygame.transform.scale(goal_img, (CELL_SIZE, CELL_SIZE))
empty_img = pygame.transform.scale(empty_img, (CELL_SIZE, CELL_SIZE))

class GridEnvironment(gym.Env):
    def __init__(self):
        super(GridEnvironment, self).__init__()

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Initialize the grid
        self.grid = self.initialize_grid()

        # Initialize the agent's position
        self.agent_pos = (0, 0)

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_SIZE, WIN_SIZE))
        pygame.display.set_caption("Grid Environment")
        self.clock = pygame.time.Clock()

    def initialize_grid(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        num_holes = np.random.randint(2, 4)  # Random number of holes (2 or 3)

        agent_start = (GRID_SIZE // 2, GRID_SIZE // 2)  # Agent's starting position

        hole_positions = [agent_start]

        for _ in range(num_holes):
            hole_pos = agent_start
            while hole_pos == agent_start or hole_pos in hole_positions:
                i = np.random.randint(0, GRID_SIZE)
                j = np.random.randint(0, GRID_SIZE)
                hole_pos = (i, j)
            hole_positions.append(hole_pos)
            grid[hole_pos] = -1  # Hole/Obstacle

        goal_pos = agent_start
        while goal_pos == agent_start or goal_pos in hole_positions:
            i = np.random.randint(0, GRID_SIZE)
            j = np.random.randint(0, GRID_SIZE)
            goal_pos = (i, j)

        grid[goal_pos] = 1  # Goal

        return grid

    def move_agent(self, action):
        if action == 0:  # Up
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == 1:  # Down
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == 2:  # Left
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == 3:  # Right
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        else:
            return

        if (
            new_pos[0] >= 0
            and new_pos[0] < GRID_SIZE
            and new_pos[1] >= 0
            and new_pos[1] < GRID_SIZE
        ):
            if self.grid[new_pos] != -1:  # If the new position is not a hole
                self.agent_pos = new_pos
            else:  # If the new position is a hole, reset the environment
                self.reset()





    def reset(self):
        self.grid = self.initialize_grid()
        self.agent_pos = (0, 0)
        return self.grid.copy()

    def step(self, action):
        self.move_agent(action)
        goal_positions = np.argwhere(self.grid == 1)
        done = len(goal_positions) > 0 and tuple(self.agent_pos) == tuple(goal_positions[0])
        reward = 2.0 if done else -0.1
        return self.grid.copy(), reward, done, {}


    # Modify the clock.tick() value in the render() method
    def render(self, fps=30):
        self.screen.fill(BLACK)
        self.draw_grid()
        pygame.display.flip()
        self.clock.tick(fps)


    def draw_grid(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.grid[i, j] == -1:  # Obstacle
                    self.screen.blit(obstacle_img, rect)
                elif self.grid[i, j] == 1:  # Goal
                    self.screen.blit(goal_img, rect)
                elif self.agent_pos == (i, j):  # Agent
                    self.screen.blit(agent_img, rect)
                else:  # Empty cell
                    self.screen.blit(empty_img, rect)

    def close(self):
        pygame.quit()


# Usage example
env = GridEnvironment()

for _ in range(100):
    done = False
    observation = env.reset()
    while not done:
        env.render(fps=5)
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        print("reward: ", reward)
        print("Observation: ", observation)

env.close()
