import time

import gymnasium as gym
import stable_baselines3.common.evaluation
from sb3_contrib import RecurrentPPO

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


# Parallel environments
number_of_training_environments = 4
training_environment = make_vec_env(
    "CartPole-v1",
    n_envs=number_of_training_environments,
    env_kwargs={"max_episode_steps": 100_000},
)

model = PPO("MlpPolicy", training_environment, verbose=1, device="cpu")
# model = RecurrentPPO(
#     "MlpLstmPolicy", training_environment, n_steps=2048, verbose=1, device="cuda"
# )


start = time.time()
model.learn(total_timesteps=10 * 100_000)
print("Training Time:", time.time() - start)

test_environment = gym.make("CartPole-v1", max_episode_steps=100_000)
evaluation_rewards = stable_baselines3.common.evaluation.evaluate_policy(
    model=model,
    env=test_environment,
    render=False,
    return_episode_rewards=False,
    n_eval_episodes=10,
)
print(evaluation_rewards)

# model.save("ppo_cartpole")
# del model # remove to demonstrate saving and loading
# model = PPO.load("ppo_cartpole")
