import gym
import random
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
import pygame

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import TensorBoard

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import math

# Environment constants
GRID_SIZE = 4
# Define the size of each cell in the grid
CELL_SIZE = 100

class GridEnvironment(gym.Env):
    def __init__(self, max_steps=100):
        super(GridEnvironment, self).__init__()

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Initialize the grid
        self.grid = self.initialize_grid()

        # Steps max and current
        self.max_steps = max_steps
        self.current_step = 0

    def initialize_grid(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        num_holes = np.random.randint(2, 4)  # Random number of holes (2 or 3)

        agent_start = (0, 0)  # Agent's starting position

        hole_positions = [agent_start]

        for _ in range(num_holes):
            hole_pos = agent_start
            while hole_pos == agent_start or hole_pos in hole_positions:
                i = np.random.randint(1, GRID_SIZE)
                j = np.random.randint(1, GRID_SIZE)
                hole_pos = (i, j)
            hole_positions.append(hole_pos)
            grid[hole_pos] = -1  # Hole/Obstacle

        goal_pos = agent_start
        while goal_pos == agent_start or goal_pos in hole_positions:
            i = np.random.randint(1, GRID_SIZE)
            j = np.random.randint(1, GRID_SIZE)
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
            # else:  # If the new position is a hole, reset the environment
            #     self.reset()

    def reset(self):
        self.grid = self.initialize_grid()
        self.agent_pos = (0, 0)
        self.current_step = 0  # Reset current step counter
        return self.grid.copy()

    def step(self, action):
        self.move_agent(action)
        goal_positions = np.argwhere(self.grid == 1)
        done = len(goal_positions) > 0 and tuple(self.agent_pos) == tuple(goal_positions[0])
        reward = 0.0  # Penalize for each step
        self.current_step += 1

        agent_x, agent_y = tuple(self.agent_pos)
        goal_x, goal_y = tuple(goal_positions[0])

        distance = math.sqrt((agent_x - goal_x)**2 + (agent_y - goal_y)**2)

        if done:  # If goal reached
            reward = 1.0
        elif self.current_step >= self.max_steps:
            done = True  # Reset the environment
            reward = -1.0  # Penalize for exceeding maximum steps
        

        else:
            reward = 1/distance

        return self.grid.copy(), reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            if not hasattr(self, 'screen'):
                pygame.init()
                screen_width = GRID_SIZE * CELL_SIZE
                screen_height = GRID_SIZE * CELL_SIZE
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            
            self.screen.fill((255, 255, 255))  # Fill the screen with white color
            
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    cell_left = j * CELL_SIZE
                    cell_top = i * CELL_SIZE
                    
                    if self.grid[i, j] == -1:  # Obstacle
                        obstacle_img = pygame.image.load('hole.jpg')
                        obstacle_img = pygame.transform.scale(obstacle_img, (CELL_SIZE, CELL_SIZE))
                        self.screen.blit(obstacle_img, (cell_left, cell_top))
                    elif self.grid[i, j] == 1:  # Goal
                        goal_img = pygame.image.load('goal.jpg')
                        goal_img = pygame.transform.scale(goal_img, (CELL_SIZE, CELL_SIZE))
                        self.screen.blit(goal_img, (cell_left, cell_top))
                    elif (i, j) == self.agent_pos:  # Agent
                        agent_img = pygame.image.load('agent.png')
                        agent_img = pygame.transform.scale(agent_img, (CELL_SIZE, CELL_SIZE))
                        self.screen.blit(agent_img, (cell_left, cell_top))
                    else:  # Empty cell
                        empty_cell_img = pygame.image.load('emptycell.jpg')
                        empty_cell_img = pygame.transform.scale(empty_cell_img, (CELL_SIZE, CELL_SIZE))
                        self.screen.blit(empty_cell_img, (cell_left, cell_top))
            
            pygame.display.update()
        
        elif mode == 'rgb_array':
            return self.grid.copy()
        
        else:
            super(GridEnvironment, self).render(mode=mode)






# Usage example
env = GridEnvironment(max_steps=10)  # Set maximum steps to 10

states = np.expand_dims(env.observation_space.sample(), axis=0)
actions = env.action_space.n

# episodes = 10
# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action = random.choice([0,1,2,3])
#         _, reward, done, _  = env.step(action)
#         score += reward
#         env.render()

#     print(f"Episode {episode}, Score: {score}")

model = Sequential()
model.add(Flatten(input_shape=(1, GRID_SIZE, GRID_SIZE)))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(actions, activation="linear"))

print("Summary: ", model.summary())

# # # #use the model
model = keras.models.load_model('my_model4.h5')

memory = SequentialMemory(limit=100000, window_length=1)

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=100000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=8,
    target_model_update=0.001
)

agent.compile(Adam(learning_rate=0.001), metrics=["mae"])
agent.fit(env, nb_steps=1000000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=100, visualize=True)  # False training 100 ep to not show render
print(np.mean(results.history["episode_reward"]))

# Save the model
model.save("my_model4.h5")

# Extract the episode rewards
episode_rewards = results.history["episode_reward"]

# Plot the rewards
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards Obtained during Testing')
# Save the plot as a PNG image
plt.savefig('rewards_plot.png')
plt.show()

env.close()
