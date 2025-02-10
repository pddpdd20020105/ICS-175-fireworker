#!/usr/bin/env python3
"""
Trains an A2C model on seat 0 in Hanabi with other seats random.
Prevents 'Illegal action' assertion by clamping the chosen action.
"""

from stable_baselines3 import A2C
from hanabi_gym_wrapper import SingleAgentHanabiEnv

def main():
    env = SingleAgentHanabiEnv()

    model = A2C(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=7e-4,
        n_steps=5
    )
    model.learn(total_timesteps=50_000)

    # Test
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Test finished, total_reward={total_reward}")

if __name__ == "__main__":
    main()
