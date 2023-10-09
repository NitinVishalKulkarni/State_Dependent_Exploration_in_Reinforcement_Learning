import time

import gymnasium as gym
import numpy as np
import pandas as pd

from network import MLPPolicy, CNNPolicy, LSTMPolicy, TransformerPolicy
from proximal_policy_optimization import ProximalPolicyOptimization
from tqdm import tqdm

if __name__ == "__main__":
    environment = gym.make(
        "CartPole-v1",
        max_episode_steps=100_000,
        render_mode="rgb_array",
    )

    ppo_configuration = {
        "seed": None,
        "policy_class": CNNPolicy,
        "recurrent_policy": False,
        "environment": environment,
        "stack_observations": True,
        "number_of_observations_to_stack": 4,
        "action_space_type": "discrete",
        "timesteps_per_batch": 2048,  # 2048
        "max_timesteps_per_episode": 100_000,
        "gamma": 0.99,
        "number_of_epochs_per_iteration": 10,
        "actor_learning_rate": 3e-4,  # 3e-4,
        "critic_learning_rate": 3e-4,  # 3e-4,
        "gae_lambda": 0.98,
        "clip": 0.2,  # 0.2
        "save_freq": 1e15,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,  # 0.02
        "entropy_coefficient": 0,
        "minibatch_size": 256,  # 256
        "render_environment_training": False,
        "render_environment_evaluation": False,
        "use_state_dependent_exploration": True,
        "device": "cuda",
        "verbose": False,
        "total_timesteps": 100_000,
        "evaluate_performance_bool": True,
        "number_of_evaluation_episodes": 10,
        "evaluate_performance_every": 5_000,
        "force_image_observation": True,
    }

    total_timesteps = ppo_configuration["total_timesteps"]
    # model = ProximalPolicyOptimization(ppo_configuration=ppo_configuration)
    # start = time.time()
    # model.learn(total_timesteps)
    # print("Training Time:", time.time() - start)

    # State Dependent Exploration Tests:
    individual_training_run_rewards = []
    for i in tqdm(range(1)):
        # print(f"\n\n\nRUN {i + 1}\n\n\n")
        model = ProximalPolicyOptimization(ppo_configuration=ppo_configuration)
        model.learn(total_timesteps)
        individual_training_run_rewards.append(model.evaluation_episode_rewards)

    print("\n\n\nFinal Results:")
    columns = {}
    results = pd.DataFrame()
    results[f"Iteration"] = [i for i in range(1, 1001)]
    for i in range(len(individual_training_run_rewards)):
        results[f"Run {i + 1} Timesteps"] = [np.NAN for _ in range(1000)]
        results[f"Run {i + 1} Min Reward"] = [np.NAN for _ in range(1000)]
        results[f"Run {i + 1} Max Reward"] = [np.NAN for _ in range(1000)]
        results[f"Run {i + 1} Mean Reward"] = [np.NAN for _ in range(1000)]
        results[f"Run {i + 1} Std Reward"] = [np.NAN for _ in range(1000)]
        results[f"Run {i + 1} Median Reward"] = [np.NAN for _ in range(1000)]

    for i in range(len(individual_training_run_rewards)):
        individual_training_run_reward = individual_training_run_rewards[i]
        individual_training_run_timesteps = []
        individual_training_run_mins = []
        individual_training_run_maxs = []
        individual_training_run_means = []
        individual_training_run_stds = []
        individual_training_run_medians = []
        print("Test:", individual_training_run_reward)
        for j in range(len(individual_training_run_reward)):
            individual_training_run_timesteps.append(
                individual_training_run_reward[j][0]
            )
            individual_training_run_mins.append(individual_training_run_reward[j][1])
            individual_training_run_maxs.append(individual_training_run_reward[j][2])
            individual_training_run_means.append(individual_training_run_reward[j][3])
            individual_training_run_stds.append(individual_training_run_reward[j][4])
            individual_training_run_medians.append(individual_training_run_reward[j][5])
        results[f"Run {i + 1} Timesteps"].iloc[
            : len(individual_training_run_timesteps)
        ] = individual_training_run_timesteps
        results[f"Run {i + 1} Min Reward"].iloc[
            : len(individual_training_run_mins)
        ] = individual_training_run_mins
        results[f"Run {i + 1} Max Reward"].iloc[
            : len(individual_training_run_maxs)
        ] = individual_training_run_maxs
        results[f"Run {i + 1} Mean Reward"].iloc[
            : len(individual_training_run_means)
        ] = individual_training_run_means
        results[f"Run {i + 1} Std Reward"].iloc[
            : len(individual_training_run_stds)
        ] = individual_training_run_stds
        results[f"Run {i + 1} Median Reward"].iloc[
            : len(individual_training_run_medians)
        ] = individual_training_run_medians

    results.head(100)
    results.to_csv("./SDE_IMAGE_CARTPOLE.csv", index=False)
