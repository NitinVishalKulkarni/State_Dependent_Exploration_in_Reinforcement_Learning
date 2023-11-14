import sys
import numpy as np
# import hdbscan
from cuml.cluster import hdbscan
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from collections import Counter
from typing import Union
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from scipy import stats
import pandas as pd
import os
import torch
import cupy as cp
from cuml import UMAP
from time import perf_counter


class StateDependentExploration(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    This class implements State Dependent Exploration.
    """

    # def __init__(self, action_space, features_extractor=None, device=None, verbose=0):
    def __init__(self, state_dependent_exploration_configuration: dict):
        """
        This method initializes the required parameters for State Dependent Exploration.
        :param state_dependent_exploration_configuration: The configuration dictionary for SDE. This contains the SDE
                                                          hyperparameters.
        """
        super(StateDependentExploration, self).__init__(state_dependent_exploration_configuration["verbose"])
        self.states_observed = []
        self.actions_taken = []
        self.action_space = state_dependent_exploration_configuration["action_space"]
        self.initial_reweighing_strength = state_dependent_exploration_configuration["initial_reweighing_strength"]
        self.reweighing_duration = state_dependent_exploration_configuration["reweighing_duration"]
        self.feature_extractor = state_dependent_exploration_configuration["feature_extractor"]
        self.device = state_dependent_exploration_configuration["device"]

        self.use_nn_features = True

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=state_dependent_exploration_configuration["min_cluster_size"],
            # max_cluster_size=250,
            min_samples=state_dependent_exploration_configuration["min_samples"],
            cluster_selection_epsilon=state_dependent_exploration_configuration["cluster_selection_epsilon"],
            cluster_selection_method=state_dependent_exploration_configuration["cluster_selection_method"],
            # leaf_size=10,
            metric=state_dependent_exploration_configuration["metric"],
            # core_dist_n_jobs=32,
            prediction_data=state_dependent_exploration_configuration["prediction_data"],
        )

        self.umap = UMAP(n_components=state_dependent_exploration_configuration["n_components"])

        self.cluster_labels_checkpoint = None
        self.previous_number_of_states_clustered = 0
        self.cluster_associated_actions = {}
        self.entire_data_clustered = True

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        print(f"Fitting the UMAP model on random observations:")
        self.training_env.reset()
        number_of_original_clustering_samples = 10_000
        warmup_observations = []
        for i in range(number_of_original_clustering_samples):
            observations, reward, dones, info = self.training_env.step(
                [self.action_space.sample() for _ in range(10)]
            )
            for observation in observations:
                warmup_observations.append(observation.flatten())
            if len(warmup_observations) > number_of_original_clustering_samples:
                break

        dr_start = perf_counter()
        self.umap.fit(np.asarray(warmup_observations))

        del warmup_observations
        # print(f"CPU Dimensionality Reduction Time: {perf_counter() - dr_start}")

        # batch_transform_start = perf_counter()
        # embeddings = self.umap_cpu.transform(np.asarray(warmup_observations))
        # print(f"Batch Transform Time: {perf_counter() - batch_transform_start}")
        #
        # self.training_env.reset()
        # overall_running_data_start = perf_counter()
        # for i in range(100_000):
        #     observations, reward, dones, info = self.training_env.step(
        #         [self.action_space.sample() for _ in range(10)]
        #     )
        #
        #     individual_transform_start = perf_counter()
        #     # for observation in observations:
        #     #     warmup_observations.append(self.umap_cpu.transform(observation.reshape(1, -1)))
        #     embeddings = self.umap_cpu.transform(np.asarray([observation.flatten() for observation in observations]))
        #     print(
        #         f"Individual Transform Time: {perf_counter() - individual_transform_start}"
        #     )
        #
        #     nn_obs_start = perf_counter()
        #     with torch.inference_mode():
        #         state_features = self.features_extractor(torch.tensor(observations, device="cuda"))
        #         state_features = torch.tensor_split(state_features, state_features.size()[0])
        #     print(f"NN OBS Time: {perf_counter() - nn_obs_start}")
        #
        # print(
        #     f"Overall Running Batch Time: {perf_counter() - overall_running_data_start}"
        # )
        # sys.exit()
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """

        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # print(f"TEST STATES OBSERVED:")
        # print(
        #     f"Original Last Observation: {type(self.model._last_obs), self.model._last_obs.shape, self.model._last_obs.flatten().shape, self.model._last_obs}")
        # print(f"List: "
        #       f"{type(list(self.model._last_obs.flatten())), self.model._last_obs[0], self.model._last_obs.flatten(), self.model._last_obs[0][0]}")
        # print(list(self.model._last_obs.flatten()))
        # sys.exit()
        if self.num_timesteps < self.reweighing_duration * self.locals["total_timesteps"]:
            if self.use_nn_features:
                # TESTING CNN FEATURES:
                # # print(self.device)
                # # print(f"Last Observation Shape:", self.model._last_obs.shape)
                # # self.states_observed.append(self.model._last_obs.squeeze())
                # # temp = self.features_extractor(torch.tensor(self.model._last_obs, device="cuda"))
                # # temp = torch.tensor_split(temp, temp.size()[0])
                # # temp = [temp[i].squeeze() for i in range(len(temp))]
                # # print("TEST FS:", type(temp), temp[0].size())
                # with torch.inference_mode():
                #     state_features = self.features_extractor(torch.tensor(self.model._last_obs, device="cuda"))
                #     state_features = torch.tensor_split(state_features, state_features.size()[0])
                #     for state_feature in state_features:
                #         self.states_observed.append(state_feature.squeeze())
                #         print(f"NN State Feature Shape: {state_feature.squeeze().shape}")
                #     # self.states_observed.append(
                #     #     self.features_extractor(torch.tensor(self.model._last_obs, device="cuda")).squeeze())

                # Testing UMAP features:
                state_features = self.umap.transform(
                    np.asarray([observation.flatten() for observation in self.model._last_obs]))
                for state_feature in state_features:
                    self.states_observed.append(state_feature)
                    # print("State Feature:", type(state_feature), type(np.asarray(state_feature)),
                    #       state_feature.flatten().shape, state_feature.ravel().shape, state_feature)
                    #
                # for state_feature in self.model._last_obs:
                #     self.states_observed.append(state_feature.flatten())
            else:
                self.states_observed.append(self.model._last_obs.flatten())
            # print(f"Action: {type(self.locals['actions']), self.locals['actions']}")
            for action in self.locals["actions"]:
                self.actions_taken.append(int(action))
            # self.actions_taken.append(int(self.locals["actions"]))
            # print(len(self.states_observed), self.states_observed[0].size())
            # print(len(self.actions_taken), self.actions_taken)
            # sys.exit()

        # print(f"\nLOCALS:")
        # for key in self.locals:
        #     print(f"{key}: {self.locals[key]}")
        #
        # print(f"\nGLOBALS:")
        # for key in self.locals:
        #     print(f"{key}: {self.locals[key]}")
        # sys.exit()
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """

        pass

    def reweigh_action_probabilities(self, action_probabilities_, share_features_extractor=None,
                                     neural_network=None):
        """
        This method...
        """
        # print(f"Length States Observed: "
        #       f"{len(self.states_observed), self.num_timesteps, self.locals['total_timesteps']}")
        # print(f"States Observed: {self.states_observed}")
        # print(f"Actions Taken: {len(self.actions_taken), self.actions_taken}")
        if (
                len(self.states_observed) < 100
                or self.num_timesteps > self.locals["total_timesteps"] * self.reweighing_duration
        ):
            print("Did this get executed?")
            return action_probabilities_.cpu().detach().numpy()
        action_probabilities_ = torch.tensor_split(action_probabilities_, action_probabilities_.size()[0])
        action_probabilities_ = [action_probabilities.squeeze() for action_probabilities in action_probabilities_]
        # print(f"Action Probabilities:", action_probabilities_[0].size())
        action_probabilities_to_return = []
        for action_probabilities in action_probabilities_:
            # Exponential Decay:
            self.initial_reweighing_strength = 1 * (
                    (0.01 / 1) ** (1 / (self.locals["total_timesteps"] * self.reweighing_duration))
            ) ** len(self.states_observed)

            action_probabilities = action_probabilities.cpu().detach().numpy()
            action_probabilities = {
                i: action_probabilities[i] for i in range(len(action_probabilities))
            }
            # print("Check 1")

            if (
                    len(self.states_observed) < 100
                    or len(self.states_observed)
                    > 1.1 * self.previous_number_of_states_clustered
            ):
                # print("Check 2")
                self.previous_number_of_states_clustered = len(self.states_observed)

                if self.use_nn_features:
                    # TODO: Remove inference mode later. Default is inference mode.
                    # with torch.inference_mode():
                    #     neural_network.sde_critic.load_state_dict(neural_network.policy_net.state_dict())
                    #     # features = feature_extractor(torch.tensor(self.states_observed, device=device))
                    #     # print(f"Features: {type(features), features.size(), features.requires_grad, device}")
                    #     if share_features_extractor:
                    #         # latent_pi, _ = neural_network(features)
                    #         # latent_pi, _ = neural_network(torch.stack(self.states_observed, dim=0))
                    #         latent_vf = neural_network.forward_sde_critic(torch.stack(self.states_observed, dim=0))
                    #         # latent_pi, _ = neural_network(self.states_observed)
                    #         # print(f"Latent Pi: {type(latent_vf), latent_vf.size()}")
                    #     # else:
                    #     #     # pi_features, _ = torch.tensor(features)
                    #     #     pi_features, _ = torch.stack(self.states_observed, dim=0)
                    #     #     latent_pi = neural_network.forward_actor(pi_features)
                    # # print(f"Features: {type(features), features.size()}")
                    # # print(f"Latent Pi: {type(latent_pi), latent_pi.size(), latent_pi.requires_grad}")
                    # # sys.exit()
                    # # latent_pi = latent_pi.cpu().detach()
                    self.clusterer.fit(cp.asarray(self.states_observed))

                else:
                    self.clusterer.fit(self.states_observed)

                self.cluster_labels_checkpoint = self.clusterer.labels_
                self.entire_data_clustered = True
                print(f"\n\nTotal States Observed: {len(self.states_observed)}")
                print(
                    f"Cluster Counts: {sorted(Counter(self.clusterer.labels_.tolist()).values(), reverse=True)}"
                )
                print(
                    f"Ordered Cluster Counts: {sorted(Counter(self.clusterer.labels_.tolist()).items())}"
                )
                print(f"Total Number of Clusters: {len(np.unique(self.clusterer.labels_))}")
            else:
                # print("Check 3")
                if self.use_nn_features:
                    # TODO: Remove inference mode later. Default is inference mode.
                    # with torch.inference_mode():
                    #     # features = feature_extractor(torch.tensor(self.states_observed[-1:], device=device))
                    #     # print(f"Feature: {type(features), features.size(), features.requires_grad}")
                    #     if share_features_extractor:
                    #         # latent_pi, _ = neural_network(features)
                    #         # latent_pi, _ = neural_network(torch.stack(self.states_observed[-1:], dim=0))
                    #         latent_vf = neural_network.forward_sde_critic(
                    #             torch.stack(self.states_observed[-1:], dim=0))
                    #         # latent_pi, _ = neural_network(torch.tensor(self.states_observed[-1:], device=device))
                    #     # else:
                    #     #     # pi_features, _ = torch.tensor(features)
                    #     #     pi_features, _ = torch.stack(self.states_observed[-1:], dim=0)
                    #     #     latent_pi = neural_network.forward_actor(pi_features)
                    # # latent_pi = latent_pi.cpu()
                    new_labels, _ = hdbscan.approximate_predict(
                        self.clusterer, cp.asarray(self.states_observed[-1:])
                    )
                else:
                    new_labels, _ = hdbscan.approximate_predict(
                        self.clusterer, self.states_observed[-1:]
                    )

                # self.clusterer.labels_ = np.concatenate(
                #     (self.clusterer.labels_, new_labels)
                # )
                self.clusterer.labels_ = np.concatenate(
                    (self.clusterer.labels_, cp.asarray(new_labels))
                )

                self.entire_data_clustered = False

            if self.entire_data_clustered:
                # print(f"Action Space: {self.action_space.n}")
                # print("Check 4")
                self.cluster_associated_actions = {
                    int(key): {
                        action: 0 for action in range(self.action_space.n)
                    }
                    for key in np.unique(self.clusterer.labels_)
                }

                for i, action_taken in enumerate(self.actions_taken):
                    self.cluster_associated_actions[int(self.clusterer.labels_[i])][
                        action_taken
                    ] += 1

            else:
                # cluster_associate_actions_update_start_time = time.perf_counter()
                # print("Check 5")
                try:
                    # print("Check 6")
                    self.cluster_associated_actions[int(self.clusterer.labels_[-1])][
                        self.actions_taken[-1]
                    ] += 1
                except KeyError:
                    # print("Check 7")
                    print(f"This error occurs when there's no outlier in the original clustering but the approximate "
                          f"predict results in an outlier.")
                    print(self.actions_taken[-1])
                    print(self.clusterer.labels_)
                    print(self.cluster_associated_actions)
                    self.cluster_associated_actions[-1] = {action: 0 for action in range(self.action_space.n)}
                    self.cluster_associated_actions[self.clusterer.labels_[-1]][
                        self.actions_taken[-1]
                    ] += 1

            # print(f"Cluster AA: {self.cluster_associated_actions}")
            keys = self.cluster_associated_actions[int(self.clusterer.labels_[-1])].keys()
            values = self.cluster_associated_actions[int(self.clusterer.labels_[-1])].values()
            # print(f"Keys: {keys}, Values: {values}")
            total_actions = sum(list(values))

            # TODO: TESTING NOT REWEIGHING THE ACTION PROBABILITIES BASED ON THE TOTAL ACTION COUNT AND INIDIVIDUAL
            #  ACTION COUNT.
            if total_actions > 10000 and min(list(values)) > 1000:
                action_probabilities = list(action_probabilities.values())
                # temp = [val for val in action_probabilities]
                # action_probabilities[0] = 1.0 - sum(action_probabilities[1:])
                action_probabilities_sum = sum(action_probabilities)
                if action_probabilities_sum != 1:
                    for i in range(len(action_probabilities)):
                        action_probabilities[i] = 1.0 - action_probabilities_sum + action_probabilities[i]
                        if 0 <= action_probabilities[i] <= 1:
                            break
                        else:
                            # print("Action Probability less than or greater than 0 after adjusting.",
                            #       i, action_probabilities[i], action_probabilities, action_probabilities_sum,
                            #       sum(action_probabilities))
                            action_probabilities[i] = int(action_probabilities[i])
                            action_probabilities_sum = sum(action_probabilities)
                            # print("TEST:", i + 1, 1.0 - action_probabilities_sum + action_probabilities[i + 1])

                # if any(i < 0 for i in action_probabilities):
                #     print("Negative Action Probabilities:", action_probabilities, self.cluster_associated_actions[
                #         self.clusterer.labels_[-1]], keys, values, total_actions)
                #     print(f"Before Subtraction: {temp, sum(temp)}")
                # sys.exit()
                action_probabilities_to_return.append(action_probabilities)
                continue
                # return action_probabilities

            cluster_associated_actions_ratios = {
                key: value / total_actions for key, value in zip(keys, values)
            }
            sde_action_probabilities = {
                i: action_probabilities[i] for i in range(len(action_probabilities))
            }
            # print(f"CAA Ratios: {cluster_associated_actions_ratios}")
            # print(f"SDE AP: {sde_action_probabilities}")

            sorted_cluster_associated_actions_ratios = sorted(
                cluster_associated_actions_ratios.items(), key=lambda x: x[1], reverse=True
            )
            action_probability_differences = []
            # print(f"Sorted CAA Ratios: {sorted_cluster_associated_actions_ratios}")
            # print("Check 8")
            for i in range(len(sorted_cluster_associated_actions_ratios)):
                # print("Check 9")
                action, caa_ratio = sorted_cluster_associated_actions_ratios[i]
                # print(f"I: {i}, Action: {action}, CAA Ratio: {caa_ratio}")
                action_probability = action_probabilities[action]
                # print(f"AP: {action_probability}")
                action_probability_difference = []
                for j in range(i + 1, len(sorted_cluster_associated_actions_ratios)):
                    next_action, next_caa_ratio = sorted_cluster_associated_actions_ratios[
                        j
                    ]
                    # print(f"J: {j}, Next Action: {next_action}, Next CAA Ratio: {next_caa_ratio}")
                    action_probability_difference.append(
                        self.initial_reweighing_strength
                        * (caa_ratio - next_caa_ratio)
                        * action_probability
                    )

                # print(f"Action Probability Difference: {action_probability_difference}")
                non_zero_values = len(action_probability_difference)
                for value in action_probability_difference:
                    # print(f"Value: {value}")
                    if value == 0:
                        non_zero_values -= 1
                    if value != 0:
                        break
                non_zero_values = non_zero_values if non_zero_values != 0 else 1

                action_probability_difference = list(
                    np.asarray(action_probability_difference) / non_zero_values
                )
                action_probability_differences.append(action_probability_difference)
                sde_action_probabilities[action] -= np.sum(
                    action_probability_differences[-1]
                )
                index = 0
                for j in range(i + 1, len(sorted_cluster_associated_actions_ratios)):
                    next_action, next_caa_ratio = sorted_cluster_associated_actions_ratios[
                        j
                    ]
                    sde_action_probabilities[next_action] += action_probability_difference[
                        index
                    ]
                    index += 1

            # print("Check 10")
            sde_action_probabilities = list(sde_action_probabilities.values())
            temp = [val for val in sde_action_probabilities]
            action_probabilities_sum = sum(sde_action_probabilities)
            if action_probabilities_sum != 1:
                for i in range(len(sde_action_probabilities)):
                    sde_action_probabilities[i] = 1.0 - action_probabilities_sum + sde_action_probabilities[i]
                    if 0 <= sde_action_probabilities[i] <= 1:
                        break
                    else:
                        # print("SDE Action Probability less than or greater than 0 after adjusting.",
                        #       i, sde_action_probabilities[i], sde_action_probabilities, action_probabilities_sum,
                        #       sum(sde_action_probabilities))
                        sde_action_probabilities[i] = int(sde_action_probabilities[i])
                        action_probabilities_sum = sum(sde_action_probabilities)
                        # print("SDE TEST:", i + 1, 1.0 - action_probabilities_sum + sde_action_probabilities[i + 1])

            # if any(i < 0 for i in sde_action_probabilities):
            #     print("Negative SDE Action Probabilities:", sde_action_probabilities, self.cluster_associated_actions[
            #         self.clusterer.labels_[-1]], keys, values, total_actions)
            #     print(f"Before SDE Subtraction: {temp, sum(temp)}")
            # sys.exit()
            action_probabilities_to_return.append(sde_action_probabilities)
            # return sde_action_probabilities
        return action_probabilities_to_return


class CustomEvalCallback(EvalCallback):
    """This class extends the default 'EvalCallback' class to make an evaluation call at the end of training and
    generate the evaluation statistics."""

    def __init__(self, eval_env: Union[gym.Env, VecEnv], store_raw_results, *args, **kwargs):
        super().__init__(eval_env, *args, **kwargs)

        self.store_raw_results = store_raw_results
        if self.log_path:
            # self.log_path = os.path.join(self.log_path[:-11], "raw_evaluation_results") if store_raw_results else (
            #     os.path.join(self.log_path[:-11], "evaluation_statistics"))
            self.log_path = self.log_path.replace("evaluations", "raw_evaluation_results") if store_raw_results \
                else self.log_path.replace("evaluations", "evaluation_statistics")

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)

                # Determine if we want to store the raw rewards/episode lengths or the statistics.
                if self.store_raw_results:
                    self.evaluations_timesteps.append(self.num_timesteps)
                    self.evaluations_results.append(episode_rewards)
                    self.evaluations_length.append(episode_lengths)
                else:
                    episode_rewards, episode_lengths = self.generate_evaluation_statistics(
                        episode_rewards=episode_rewards, episode_lengths=episode_lengths)
                    self.evaluations_timesteps.append(self.num_timesteps)
                    self.evaluations_results.append(episode_rewards)
                    self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """

        self.eval_freq = 1
        self._on_step()

        if self.store_raw_results:
            for i in range(len(self.evaluations_timesteps)):
                self.evaluations_results[i], self.evaluations_length[i] = self.generate_evaluation_statistics(
                    episode_rewards=self.evaluations_results[i], episode_lengths=self.evaluations_length[i])

            kwargs = {}
            # Save success log if present
            if len(self._is_success_buffer) > 0:
                self.evaluations_successes.append(self._is_success_buffer)
                kwargs = dict(successes=self.evaluations_successes)

            self.log_path = self.log_path.replace("raw_evaluation_results", "evaluation_statistics")
            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                **kwargs,
            )

    @staticmethod
    def generate_evaluation_statistics(episode_rewards, episode_lengths):
        """
        :param timestep:
        :param episode_rewards:
        :param episode_lengths:
        :return:
        """

        min_reward = min(episode_rewards)
        max_reward = max(episode_rewards)
        mean_reward = round(np.mean(episode_rewards), 2)
        std_reward = round(np.std(episode_rewards), 2)
        median_reward = round(np.median(episode_rewards), 2)
        mode_reward, mode_reward_count = stats.mode(episode_rewards)
        (
            one_percentile_reward,
            five_percentile_reward,
            ten_percentile_reward,
            twenty_five_percentile_reward,
            fifty_percentile_reward,
            seventy_five_percentile_reward,
            ninty_percentile_reward,
            ninty_five_percentile_reward,
            ninty_nine_percentile_reward,
        ) = np.percentile(a=episode_rewards, q=[1, 5, 10, 25, 50, 75, 90, 95, 99])

        min_episode_lengths = min(episode_lengths)
        max_episode_lengths = max(episode_lengths)
        mean_episode_lengths = np.mean(episode_lengths)
        std_episode_lengths = np.std(episode_lengths)
        median_episode_lengths = np.median(episode_lengths)
        mode_episode_lengths, mode_episode_lengths_count = stats.mode(episode_lengths)
        (
            one_percentile_episode_lengths,
            five_percentile_episode_lengths,
            ten_percentile_episode_lengths,
            twenty_five_percentile_episode_lengths,
            fifty_percentile_episode_lengths,
            seventy_five_percentile_episode_lengths,
            ninty_percentile_episode_lengths,
            ninty_five_percentile_episode_lengths,
            ninty_nine_percentile_episode_lengths,
        ) = np.percentile(a=episode_lengths, q=[1, 5, 10, 25, 50, 75, 90, 95, 99])

        episode_rewards = [
            min_reward,
            max_reward,
            mean_reward,
            std_reward,
            median_reward,
            mode_reward,
            mode_reward_count,
            one_percentile_reward,
            five_percentile_reward,
            ten_percentile_reward,
            twenty_five_percentile_reward,
            fifty_percentile_reward,
            seventy_five_percentile_reward,
            ninty_percentile_reward,
            ninty_five_percentile_reward,
            ninty_nine_percentile_reward,
        ]

        episode_lengths = [
            min_episode_lengths,
            max_episode_lengths,
            mean_episode_lengths,
            std_episode_lengths,
            median_episode_lengths,
            mode_episode_lengths,
            mode_episode_lengths_count,
            one_percentile_episode_lengths,
            five_percentile_episode_lengths,
            ten_percentile_episode_lengths,
            twenty_five_percentile_episode_lengths,
            fifty_percentile_episode_lengths,
            seventy_five_percentile_episode_lengths,
            ninty_percentile_episode_lengths,
            ninty_five_percentile_episode_lengths,
            ninty_nine_percentile_episode_lengths,
        ]

        return episode_rewards, episode_lengths
