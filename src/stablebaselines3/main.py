import copy

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_policies import CustomActorCriticPolicy
from results_generator import ResultsGenerator
import numpy as np
import json


if __name__ == "__main__":
    random_seed = np.random.randint(2**32 - 1)

    results_generator_configuration = {
        "configuration_name": "SDE",
        # Environment configuration:
        "environment_id": "PongNoFrameskip-v4",
        "environment_keyword_arguments": {},
        "is_atari_environment": True,
        "make_vector_environment_bool": True,
        "vector_environment_class": SubprocVecEnv,
        "number_of_frames_to_stack": 4,
        # Training configration:
        "number_of_training_environments": 10,
        "number_of_training_runs": 2,
        "agent": PPO,
        "policy": CustomActorCriticPolicy,
        "verbose": 1,
        "device": "cuda",
        "total_timesteps": 2_5000,
        "use_state_dependent_exploration": False,
        # Evaluation configuration:
        "evaluate_agent_performance": True,
        "number_of_evaluation_environments": 10,
        "number_of_evaluation_episodes": 10,
        "evaluation_frequency": 2048,
        "deterministic_evaluation_policy": True,
        "render_evaluation": False,
        "store_raw_evaluation_results": True,
        # Algorithm configuration:
        "n_steps": 2048,
        "mini_batch_size": 64,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "ent_coef": 0.00,
        "clip_range": 0.2,
        "decay_learning_rate": False,
        "decay_clip_range": False,
        "seed": random_seed,
    }

    state_dependent_exploration_configuration = {
        "verbose": 0,
        "action_space": None,
        "feature_extractor": "Latent Policy",
        "device": results_generator_configuration["device"],
        "initial_reweighing_strength": 1,
        "reweighing_duration": 1,
        # HDBSCAN configuration:
        "min_cluster_size": 10,
        "min_samples": 10,
        "cluster_selection_epsilon": 0,
        "cluster_selection_method": "leaf",
        "leaf_size": 10,
        "metric": "euclidean",
        "prediction_data": True,
        "cluster_persistence": 0.1,
        # UMAP configuration:
        "n_components": 4,
        "number_of_feature_extractor_train_observations": 10_000,
    }

    with open("results_generator_configuration.json", "w") as fp:
        non_json_serializable_keys = ["vector_environment_class", "agent", "policy"]
        results_generator_configuration_copy = copy.deepcopy(
            results_generator_configuration
        )
        for key in non_json_serializable_keys:
            results_generator_configuration_copy[key] = str(
                results_generator_configuration_copy[key]
            )
        json.dump(
            results_generator_configuration_copy,
            fp,
        )

    with open("state_dependent_exploration_configuration.json", "w") as fp:
        json.dump(state_dependent_exploration_configuration, fp)

    results_generator = ResultsGenerator(configuration=results_generator_configuration)
    results_generator.generate_results()
