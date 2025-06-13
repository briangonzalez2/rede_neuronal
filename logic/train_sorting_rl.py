from stable_baselines3 import DQN
from sorting_env import SortingEnv

env = SortingEnv(list_size=10)
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    buffer_size=10000,
    exploration_fraction=0.3,
    exploration_final_eps=0.02
)
model.learn(total_timesteps=200000)
model.save("models/sorting_dqn")

env = SortingEnv()
obs = env.reset()
done = False
steps = []
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    steps.append({"state": obs.tolist(), "swap": info["swap"], "swapped": info["swapped"]})
print("Sorting steps:", steps)