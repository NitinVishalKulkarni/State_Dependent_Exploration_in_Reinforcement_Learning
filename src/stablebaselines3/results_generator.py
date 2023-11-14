import sys
import time
from collections import defaultdict

import gymnasium as gym
import numpy as np
import tqdm
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

import pandas as pd
from src.settings import results_directory
from custom_callbacks import CustomEvalCallback


class ResultsGenerator:
    """
    This class trains a StableBaselines3 agent, evaluates the trained agent's performance and stores the results.
    """

    def __init__(self, configuration: dict):
        """
        This method initializes the required parameters.

        :param configuration - The configuration for training, evaluation, and result generation."""
        self.configuration = configuration
        self.configuration_name = configuration["configuration_name"]
        self.environment_id = configuration["environment_id"]
        self.make_vector_environment_bool = configuration["make_vector_environment_bool"]

        # Train configuration
        self.number_of_training_environments = configuration["number_of_training_environments"]
        self.environment_kwargs = configuration["environment_keyword_arguments"]
        self.number_of_training_runs = configuration["number_of_training_runs"]
        self.agent = configuration["agent"]
        self.policy = configuration["policy"]
        self.verbose = configuration["verbose"]
        self.device = configuration["device"]
        self.total_timesteps = configuration["total_timesteps"]
        # self.callback = configuration["callback"]
        self.use_state_dependent_exploration = configuration["use_state_dependent_exploration"]

        # Evaluation configuration
        self.evaluate_agent_performance = configuration["evaluate_agent_performance"]
        self.evaluation_frequency = configuration["evaluation_frequency"]
        self.number_of_evaluation_environments = configuration["number_of_evaluation_environments"]
        self.number_of_evaluation_episodes = configuration["number_of_evaluation_episodes"]
        self.deterministic_evaluation_policy = configuration["deterministic_evaluation_policy"]
        self.render_evaluation = configuration["render_evaluation"]
        self.store_raw_evaluation_results = configuration["store_raw_evaluation_results"]

        # Instantiating the training and evaluation environments
        self.training_environment = make_atari_env(
            env_id=self.environment_id,
            n_envs=self.number_of_training_environments,
            env_kwargs=self.environment_kwargs,
        )
        self.training_environment = VecFrameStack(self.training_environment, n_stack=4)
        self.evaluation_environment = make_atari_env(env_id=self.environment_id,
                                                     n_envs=self.number_of_evaluation_environments,
                                                     env_kwargs=self.environment_kwargs)
        self.evaluation_environment = VecFrameStack(self.evaluation_environment, n_stack=4)

        # Train and evaluation statistics
        self.evaluation_episode_rewards = []
        self.evaluation_episode_lengths = []

    def train_agent(self, model, log_path=None, best_model_path=None):
        """
        This method trains our agent on the environment.
        """

        if self.verbose == 1:
            print("Training Agent:")

        # Determining which callback to use.
        if self.evaluate_agent_performance:
            evaluation_callback = CustomEvalCallback(eval_env=self.evaluation_environment,
                                                     store_raw_results=self.store_raw_evaluation_results,
                                                     n_eval_episodes=self.number_of_evaluation_episodes,
                                                     eval_freq=self.evaluation_frequency,
                                                     deterministic=self.deterministic_evaluation_policy, verbose=1,
                                                     best_model_save_path=best_model_path, log_path=log_path)

            if self.use_state_dependent_exploration:
                callback = CallbackList([model.policy.state_dependent_exploration, evaluation_callback])
            else:
                callback = evaluation_callback
        else:
            if self.use_state_dependent_exploration:
                callback = model.policy.state_dependent_exploration
            else:
                callback = None

        training_start_time = time.perf_counter()
        model.learn(total_timesteps=self.total_timesteps, callback=callback,
                    progress_bar=True)

        if self.verbose == 1:
            print(f"Training Time: {time.perf_counter() - training_start_time} seconds")

    def generate_final_evaluation_statistics(self, log_path):
        """

        :return:
        """

        final_evaluation_statistics = pd.DataFrame()
        per_training_run_evaluation_statistics = [np.load(f"{log_path}/training_run_{i + 1}/evaluation_statistics.npz")
                                                  for i in range(self.number_of_training_runs)]

        timesteps = defaultdict(list)
        mins = defaultdict(list)
        maxs = defaultdict(list)
        means = defaultdict(list)
        stds = defaultdict(list)
        medians = defaultdict(list)

        for j in range(len(per_training_run_evaluation_statistics[0]["timesteps"])):
            for i in range(self.number_of_training_runs):
                timesteps[f"{(j + 1) * self.evaluation_frequency}"].append(
                    per_training_run_evaluation_statistics[i][
                        "timesteps"][j])
                mins[f"{(j + 1) * self.evaluation_frequency}"].append(per_training_run_evaluation_statistics[i][
                                                                          "results"][j][0])
                maxs[f"{(j + 1) * self.evaluation_frequency}"].append(per_training_run_evaluation_statistics[i][
                                                                          "results"][j][1])
                means[f"{(j + 1) * self.evaluation_frequency}"].append(per_training_run_evaluation_statistics[i][
                                                                           "results"][j][2])
                stds[f"{(j + 1) * self.evaluation_frequency}"].append(per_training_run_evaluation_statistics[i][
                                                                          "results"][j][3])
                medians[f"{(j + 1) * self.evaluation_frequency}"].append(per_training_run_evaluation_statistics[i][
                                                                             "results"][j][4])

        final_evaluation_statistics["Min Timesteps"] = [np.NAN for _ in range(len(timesteps))]
        final_evaluation_statistics["Max Timesteps"] = [np.NAN for _ in range(len(timesteps))]
        final_evaluation_statistics["Mean Timesteps"] = [np.NAN for _ in range(len(timesteps))]
        final_evaluation_statistics["Std Timesteps"] = [np.NAN for _ in range(len(timesteps))]
        final_evaluation_statistics["Median Timesteps"] = [np.NAN for _ in range(len(timesteps))]

        final_evaluation_statistics["Min Min Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Max Min Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Mean Min Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Std Min Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Median Min Reward"] = [np.NAN for _ in range(len(mins))]

        final_evaluation_statistics["Min Max Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Max Max Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Mean Max Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Std Max Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Median Max Reward"] = [np.NAN for _ in range(len(mins))]

        final_evaluation_statistics["Min Mean Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Max Mean Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Mean Mean Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Std Mean Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Median Mean Reward"] = [np.NAN for _ in range(len(mins))]

        final_evaluation_statistics["Min Std Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Max Std Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Mean Std Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Std Std Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Median Std Reward"] = [np.NAN for _ in range(len(mins))]

        final_evaluation_statistics["Min Median Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Max Median Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Mean Median Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Std Median Reward"] = [np.NAN for _ in range(len(mins))]
        final_evaluation_statistics["Median Median Reward"] = [np.NAN for _ in range(len(mins))]

        for i in range(len(mins)):
            final_evaluation_statistics["Min Timesteps"].iloc[i] = np.nanmin(
                timesteps[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Max Timesteps"].iloc[i] = np.nanmax(
                timesteps[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Mean Timesteps"].iloc[i] = np.nanmean(
                timesteps[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Std Timesteps"].iloc[i] = np.nanstd(
                timesteps[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Median Timesteps"].iloc[i] = np.nanmedian(
                timesteps[f"{(i + 1) * self.evaluation_frequency}"])

            final_evaluation_statistics["Min Min Reward"].iloc[i] = np.nanmin(
                mins[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Max Min Reward"].iloc[i] = np.nanmax(
                mins[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Mean Min Reward"].iloc[i] = np.nanmean(
                mins[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Std Min Reward"].iloc[i] = np.nanstd(
                mins[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Median Min Reward"].iloc[i] = np.nanmedian(
                mins[f"{(i + 1) * self.evaluation_frequency}"])

            final_evaluation_statistics["Min Max Reward"].iloc[i] = np.nanmin(
                maxs[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Max Max Reward"].iloc[i] = np.nanmax(
                maxs[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Mean Max Reward"].iloc[i] = np.nanmean(
                maxs[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Std Max Reward"].iloc[i] = np.nanstd(
                maxs[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Median Max Reward"].iloc[i] = np.nanmedian(
                maxs[f"{(i + 1) * self.evaluation_frequency}"])

            final_evaluation_statistics["Min Mean Reward"].iloc[i] = np.nanmin(
                means[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Max Mean Reward"].iloc[i] = np.nanmax(
                means[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Mean Mean Reward"].iloc[i] = np.nanmean(
                means[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Std Mean Reward"].iloc[i] = np.nanstd(
                means[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Median Mean Reward"].iloc[i] = np.nanmedian(
                means[f"{(i + 1) * self.evaluation_frequency}"])

            final_evaluation_statistics["Min Std Reward"].iloc[i] = np.nanmin(
                stds[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Max Std Reward"].iloc[i] = np.nanmax(
                stds[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Mean Std Reward"].iloc[i] = np.nanmean(
                stds[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Std Std Reward"].iloc[i] = np.nanstd(
                stds[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Median Std Reward"].iloc[i] = np.nanmedian(
                stds[f"{(i + 1) * self.evaluation_frequency}"])

            final_evaluation_statistics["Min Median Reward"].iloc[i] = np.nanmin(
                medians[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Max Median Reward"].iloc[i] = np.nanmax(
                medians[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Mean Median Reward"].iloc[i] = np.nanmean(
                medians[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Std Median Reward"].iloc[i] = np.nanstd(
                medians[f"{(i + 1) * self.evaluation_frequency}"])
            final_evaluation_statistics["Median Median Reward"].iloc[i] = np.nanmedian(
                medians[f"{(i + 1) * self.evaluation_frequency}"])

        final_evaluation_statistics.head(100)
        final_evaluation_statistics.to_csv(f"{log_path}/test_sb3_sde.csv", index=False)

    def generate_results(self):
        """
        This method generates the results.
        """

        current_time = time.localtime()
        current_time = time.strftime("%m-%d-%Y %H-%M-%S", current_time)

        for i in tqdm.tqdm(range(self.number_of_training_runs)):
            model = self.agent(
                self.policy, self.training_environment, verbose=self.verbose, device=self.device
            )

            self.train_agent(model=model,
                             log_path=f"{results_directory}/{self.environment_id}/{self.configuration_name}"
                                      f"/{current_time}/training_run_{i + 1}/")
            # self.train_agent(model=model, log_path=None)

            # evaluation_results = np.load(f"{results_directory}/{self.configuration_name}"
            #                              f"/{current_time}/training_run_{i + 1}/evaluation_statistics.npz")
            # print(f"\n\nTimesteps: {evaluation_results['timesteps']} \nEpisode Rewards:"
            #       f" {evaluation_results['results']} "
            #       f"\nEpisode Lengths: {evaluation_results['ep_lengths']}")

        if self.evaluate_agent_performance:
            self.generate_final_evaluation_statistics(
                log_path=f"{results_directory}/{self.environment_id}/{self.configuration_name}/{current_time}")
