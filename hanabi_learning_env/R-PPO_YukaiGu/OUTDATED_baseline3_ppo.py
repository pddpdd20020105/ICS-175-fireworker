import gym
import numpy as np
from hanabi_env import HanabiEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class HanabiEnvWrapper(gym.Env):
    def __init__(self):
        super(HanabiEnvWrapper, self).__init__()
        self.env = HanabiEnv()
        self.action_space = self.env.actionSpace
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(658,), dtype=np.float32)
        self._seed = None

    def reset(self):
        observation = self.env.reset()
        return np.array(observation, dtype=np.float32)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return np.array(observation, dtype=np.float32), float(reward), done, info

    def seed(self, seed = None):
        self._seed = seed
        np.random.seed(seed)

    def render(self, mode="human"):
        pass

    def close(self):
        pass


env = make_vec_env(lambda: HanabiEnvWrapper(), n_envs=1)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./runs/Hanabi_PPO/")
model.learn(total_timesteps=10_000_000)
model.save("hanabi_ppo")
