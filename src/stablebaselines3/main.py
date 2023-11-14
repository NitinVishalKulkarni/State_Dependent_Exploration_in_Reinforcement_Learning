from stable_baselines3 import PPO
from custom_policies import CustomActorCriticPolicy
import json
from results_generator import ResultsGenerator

if __name__ == "__main__":
    results_generator_configuration = {
        "configuration_name": "SDE",
        "environment_id": "PongNoFrameskip-v4",
        "make_vector_environment_bool": True,
        "number_of_training_environments": 10,
        "environment_keyword_arguments": {},
        # "environment_keyword_arguments": {"max_episode_steps": 10_000},
        "number_of_training_runs": 5,
        "agent": PPO,
        "policy": CustomActorCriticPolicy,
        "verbose": 1,
        "device": "cuda",
        "total_timesteps": 1_000_000,
        "number_of_evaluation_environments": 10,
        "number_of_evaluation_episodes": 10,
        "deterministic_evaluation_policy": True,
        "render_evaluation": False,
        "use_state_dependent_exploration": True,
        "evaluate_agent_performance": True,
        "evaluation_frequency": 10_000,
        "store_raw_evaluation_results": True
    }

    state_dependent_exploration_configuration = {
        "verbose": 0,
        "action_space": None,
        "feature_extractor": "UMAP",
        "device": results_generator_configuration["device"],
        "initial_reweighing_strength": 1,
        "reweighing_duration": 0.75,

        # HDBSCAN configuration:
        "min_cluster_size": 10,
        "min_samples": 10,
        "cluster_selection_epsilon": 0,
        "cluster_selection_method": "leaf",
        "metric": "euclidean",
        "prediction_data": True,

        # UMAP configuration:
        "n_components": 16,
        "number_of_umap_observations": 10_000,

    }

    with open('results_generator_configuration.json', 'w') as fp:
        json.dump({"use_state_dependent_exploration": results_generator_configuration[
            "use_state_dependent_exploration"]}, fp)

    with open('state_dependent_exploration_configuration.json', 'w') as fp:
        json.dump(state_dependent_exploration_configuration, fp)

    results_generator = ResultsGenerator(configuration=results_generator_configuration)
    results_generator.generate_results()
