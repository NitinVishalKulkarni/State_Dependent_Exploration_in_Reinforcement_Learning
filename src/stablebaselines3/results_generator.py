import time
from collections import defaultdict

import numpy as np
import tqdm
from stable_baselines3.common.env_util import make_vec_env, make_atari_env

from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CallbackList

import pandas as pd
from src.settings import results_directory
from custom_callbacks import CustomEvalCallback
from typing import Callable

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 1000)


class ResultsGenerator:
    """
    This class trains a StableBaselines3 agent, evaluates the trained agent's performance and stores the results.
    """

    def __init__(self, configuration: dict):
        """
        This method initializes the required parameters.

        :param configuration - The configuration for training, evaluation, and result generation.
        """
        self.configuration = configuration
        self.configuration_name = configuration["configuration_name"]
        self.environment_id = configuration["environment_id"]
        self.is_atari_environment = configuration["is_atari_environment"]
        self.make_vector_environment_bool = configuration[
            "make_vector_environment_bool"
        ]
        self.vector_environment_class = configuration["vector_environment_class"]
        # self.wrapper_class = configuration["wrapper_class"]
        # self.wrapper_kwargs = configuration["wrapper_kwargs"]
        self.number_of_frames_to_stack = configuration["number_of_frames_to_stack"]

        # Train configuration:
        self.number_of_training_environments = configuration[
            "number_of_training_environments"
        ]
        self.environment_kwargs = configuration["environment_keyword_arguments"]
        self.number_of_training_runs = configuration["number_of_training_runs"]
        self.agent = configuration["agent"]
        self.n_steps = configuration["n_steps"]
        self.policy = configuration["policy"]
        self.verbose = configuration["verbose"]
        self.device = configuration["device"]
        self.total_timesteps = configuration["total_timesteps"]
        self.use_state_dependent_exploration = configuration[
            "use_state_dependent_exploration"
        ]

        # Evaluation configuration:
        self.evaluate_agent_performance = configuration["evaluate_agent_performance"]
        self.evaluation_frequency = configuration["evaluation_frequency"]
        self.number_of_evaluation_environments = configuration[
            "number_of_evaluation_environments"
        ]
        self.number_of_evaluation_episodes = configuration[
            "number_of_evaluation_episodes"
        ]
        self.deterministic_evaluation_policy = configuration[
            "deterministic_evaluation_policy"
        ]
        self.render_evaluation = configuration["render_evaluation"]
        self.store_raw_evaluation_results = configuration[
            "store_raw_evaluation_results"
        ]

        # Instantiating the training and evaluation environments:
        if self.is_atari_environment:
            self.training_environment = make_atari_env(
                env_id=self.environment_id,
                n_envs=self.number_of_training_environments,
                env_kwargs=self.environment_kwargs,
                vec_env_cls=self.vector_environment_class,
                # wrapper_kwargs={"screen_size": 128}
            )
            self.evaluation_environment = make_atari_env(
                env_id=self.environment_id,
                n_envs=self.number_of_evaluation_environments,
                env_kwargs=self.environment_kwargs,
                vec_env_cls=self.vector_environment_class,
                # wrapper_kwargs={"screen_size": 128}
            )
        else:
            self.training_environment = make_vec_env(
                env_id=self.environment_id,
                n_envs=self.number_of_training_environments,
                env_kwargs=self.environment_kwargs,
                vec_env_cls=self.vector_environment_class,
            )
            self.evaluation_environment = make_vec_env(
                env_id=self.environment_id,
                n_envs=self.number_of_evaluation_environments,
                env_kwargs=self.environment_kwargs,
                vec_env_cls=self.vector_environment_class,
            )

        if self.number_of_frames_to_stack > 1:
            self.training_environment = VecFrameStack(
                self.training_environment, n_stack=self.number_of_frames_to_stack
            )
            self.evaluation_environment = VecFrameStack(
                self.evaluation_environment, n_stack=self.number_of_frames_to_stack
            )
        print(
            "Training Environment Observation Space:",
            self.training_environment.observation_space,
        )
        print(
            "Evaluation Environment Observation Space:",
            self.evaluation_environment.observation_space,
        )
        print("Training environment action space:", self.training_environment.action_space)

    @staticmethod
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
          current learning rate depending on remaining progress
        """

        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func

    def train_agent(self, model, log_path=None, best_model_path=None):
        """
        This method trains our agent on the environment.
        """

        if self.verbose == 1:
            print("\n\n\033[1mTraining Agent:\033[1m")

        # Determining which callback to use:
        if self.evaluate_agent_performance:
            evaluation_callback = CustomEvalCallback(
                eval_env=self.evaluation_environment,
                store_raw_results=self.store_raw_evaluation_results,
                n_eval_episodes=self.number_of_evaluation_episodes,
                eval_freq=self.evaluation_frequency,
                deterministic=self.deterministic_evaluation_policy,
                verbose=self.verbose,
                best_model_save_path=best_model_path,
                log_path=log_path,
            )

            if self.use_state_dependent_exploration:
                callback = CallbackList(
                    [model.policy.state_dependent_exploration, evaluation_callback]
                )
            else:
                callback = evaluation_callback
        else:
            if self.use_state_dependent_exploration:
                callback = model.policy.state_dependent_exploration
            else:
                callback = None

        training_start_time = time.perf_counter()
        model.learn(
            total_timesteps=self.total_timesteps, callback=callback, progress_bar=True
        )

        if self.verbose >= 1:
            print(
                f"\n\n\033[1mTraining Time: {time.perf_counter() - training_start_time} seconds\033[0m"
            )

    def generate_consolidated_evaluation_statistics(self, log_path):
        """
        This method generates the consolidated evaluation statistics.

        :param log_path: Path to a folder where the evaluation statistics (``evaluation_statistics.npz``) are saved.
        """

        per_training_run_evaluation_statistics = [
            np.load(f"{log_path}/training_run_{i + 1}/evaluation_statistics.npz")
            for i in range(self.number_of_training_runs)
        ]

        for metric, array_name in zip(
            ["Episode Reward", "Episode Length"], ["results", "ep_lengths"]
        ):
            consolidated_evaluation_statistics = pd.DataFrame(
                index=range(len(per_training_run_evaluation_statistics[0]["timesteps"]))
            )

            statistical_measures = {
                "Min": np.nanmin,
                "Max": np.nanmax,
                "Mean": np.nanmean,
                "Std": np.nanstd,
                "Median": np.nanmedian,
            }

            evaluation_metrics = {
                f"Min {metric}": defaultdict(list),
                f"Max {metric}": defaultdict(list),
                f"Mean {metric}": defaultdict(list),
                f"Std {metric}": defaultdict(list),
                f"Median {metric}": defaultdict(list),
                f"Mode {metric}": defaultdict(list),
                f"Mode {metric} Count": defaultdict(list),
                f"One Percentile {metric}": defaultdict(list),
                f"Five Percentile {metric}": defaultdict(list),
                f"Ten Percentile {metric}": defaultdict(list),
                f"Twenty Five Percentile {metric}": defaultdict(list),
                f"Fifty Percentile {metric}": defaultdict(list),
                f"Seventy Five Percentile {metric}": defaultdict(list),
                f"Ninety Percentile {metric}": defaultdict(list),
                f"Ninety Five Percentile {metric}": defaultdict(list),
                f"Ninety Nine Percentile {metric}": defaultdict(list),
            }

            consolidated_evaluation_statistics[
                "Timesteps"
            ] = per_training_run_evaluation_statistics[0]["timesteps"]

            # Populating the evaluation metrics dictionary with the statistics from each training run:
            for i in range(len(per_training_run_evaluation_statistics[0]["timesteps"])):
                for j in range(self.number_of_training_runs):
                    freq = (
                        (i + 1)
                        * self.evaluation_frequency
                        * self.number_of_training_environments
                    )
                    for k, values in enumerate(evaluation_metrics.values()):
                        values[f"{freq}"].append(
                            per_training_run_evaluation_statistics[j][array_name][i][k]
                        )

            # Populating the DataFrame with calculated statistical measures:
            for evaluation_metric, values in zip(
                evaluation_metrics, evaluation_metrics.values()
            ):
                for i, statistical_measure in enumerate(statistical_measures):
                    for j in range(len(values)):
                        freq = (
                            (j + 1)
                            * self.evaluation_frequency
                            * self.number_of_training_environments
                        )
                        consolidated_evaluation_statistics.at[
                            j, f"{statistical_measure} {evaluation_metric}"
                        ] = statistical_measures[statistical_measure](values[f"{freq}"])

            consolidated_evaluation_statistics.to_csv(
                f"{log_path}/consolidated_{metric.lower().replace(' ', '_')}_statistics.csv",
                index=False,
            )
            if self.verbose >= 1:
                print(
                    f"\n\n\033[1mConsolidated Evaluation {metric} Statistics:\033[0m\n",
                    consolidated_evaluation_statistics.head(100),
                )

    def generate_results(self):
        """
        This method generates the results for evaluating the agent's performance.
        """

        current_time = time.localtime()
        current_time = time.strftime("%m-%d-%Y %H-%M-%S", current_time)

        for i in tqdm.tqdm(range(self.number_of_training_runs)):
            model = self.agent(
                self.policy,
                self.training_environment,
                verbose=self.verbose,
                device=self.device,
                n_steps=self.n_steps,
                batch_size=256,
                n_epochs=4,
                learning_rate=self.linear_schedule(2.5e-4),
                ent_coef=0.01,
                clip_range=self.linear_schedule(0.1)
            )

            self.train_agent(
                model=model,
                log_path=f"{results_directory}/{self.environment_id}/{self.configuration_name}"
                f"/{current_time}/training_run_{i + 1}/",
                best_model_path=f"{results_directory}/{self.environment_id}/{self.configuration_name}"
                f"/{current_time}/best_model/",
            )

        if self.evaluate_agent_performance:
            self.generate_consolidated_evaluation_statistics(
                log_path=f"{results_directory}/{self.environment_id}/{self.configuration_name}/{current_time}"
            )
