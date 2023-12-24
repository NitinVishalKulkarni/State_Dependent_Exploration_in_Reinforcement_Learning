from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from custom_policies import CustomActorCriticPolicy
from results_generator import ResultsGenerator

import json
from src.settings import results_directory
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    results_generator_configuration = {
        "configuration_name": "SDE",
        "environment_id": "BoxingNoFrameskip-v4",
        "is_atari_environment": True,
        "make_vector_environment_bool": True,
        "vector_environment_class": SubprocVecEnv,
        "number_of_frames_to_stack": 4,
        # Training configration:
        "number_of_training_environments": 8,
        "environment_keyword_arguments": {},
        # "environment_keyword_arguments": {"max_episode_steps": 100_000},
        "number_of_training_runs": 1,
        "agent": PPO,
        "policy": CustomActorCriticPolicy,
        # "policy": "CnnPolicy",
        "verbose": 1,
        "device": "cuda",
        "total_timesteps": 5_000_000,
        # Evaluation configuration:
        "number_of_evaluation_environments": 10,
        "number_of_evaluation_episodes": 10,
        "deterministic_evaluation_policy": True,
        "render_evaluation": False,
        "use_state_dependent_exploration": True,
        "evaluate_agent_performance": True,
        "evaluation_frequency": 12500,
        "n_steps": 128,
        "store_raw_evaluation_results": True,
    }

    state_dependent_exploration_configuration = {
        "verbose": 1,
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
        "metric": "euclidean",  # ["euclidean", "cosine"]
        "prediction_data": True,
        "cluster_persistence": 0.1,
        # UMAP configuration:
        "n_components": 16,
        "number_of_feature_extractor_train_observations": 10_000,
    }

    with open("results_generator_configuration.json", "w") as fp:
        json.dump(
            {
                "use_state_dependent_exploration": results_generator_configuration[
                    "use_state_dependent_exploration"
                ]
            },
            fp,
        )

    with open("state_dependent_exploration_configuration.json", "w") as fp:
        json.dump(state_dependent_exploration_configuration, fp)

    results_generator = ResultsGenerator(configuration=results_generator_configuration)
    results_generator.generate_results()
