import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import TensorBoard


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


# env = gym.make('CartPole-v1')
env = gym.make('FrozenLake-v1')

states = env.observation_space.sample()
actions = env.action_space.n

print(states)
print(actions)


model = Sequential()
model.add(Dense(4,input_shape=(1,)))
model.add(Dense(128,activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(actions, activation="linear"))

print("Summary: ", model.summary())

memory = SequentialMemory(limit=100000, window_length=1)

# # # #use the model
# model = keras.models.load_model('my_model2.h5')

agent = DQNAgent(
    model = model,
    memory = SequentialMemory(limit=100000, window_length=1),
    policy= BoltzmannQPolicy(),
    nb_actions= actions,
    nb_steps_warmup = 8,
    target_model_update = 0.001   
)

agent.compile(Adam(learning_rate= 0.001), metrics=["mae"])
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)


results = agent.test(env, nb_episodes=100, visualize=False) #False training 100 ep to not show render
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



# # episodes = 10
# # for episode in range(1, episodes + 1):
# #     state = env.reset()
# #     done = False
# #     score = 0

# #     while not done:
# #         action = random.choice([0,1,2,3])
# #         _, reward, done, _  = env.step(action)
# #         score += reward
# #         env.render()

# #     print(f"Episode {episode}, Score: {score}")

env.close


