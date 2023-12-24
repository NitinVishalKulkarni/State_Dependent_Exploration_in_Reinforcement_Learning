import sys
import time

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
from torchvision.models import AlexNet_Weights, AlexNet
import torch.nn as nn
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import Normalizer
import copy


class CNNFeatureExtractor(AlexNet):
    """
    This class inherits the PyTorch CNN models, and adds in a method to get the features.
    """

    def __init__(self):
        super().__init__()

    def get_features(self, x):
        x = self.features(x)
        # print(f"Output of the Convolutional Layers: {x.size()}")
        x = torch.flatten(x, start_dim=1)
        # print(f"Flattened Output of the Convolutional Layers: {x.size()}")
        return x


class StateDependentExploration(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    This class implements State Dependent Exploration.
    """

    def __init__(self, state_dependent_exploration_configuration: dict):
        """
        This method initializes the required parameters for State Dependent Exploration.
        :param state_dependent_exploration_configuration: The configuration dictionary for SDE. This contains the SDE
                                                          hyperparameters.
        """

        super(StateDependentExploration, self).__init__(
            state_dependent_exploration_configuration["verbose"]
        )

        self.states_observed = []
        self.actions_taken = []
        self.action_space = state_dependent_exploration_configuration["action_space"]
        self.initial_reweighing_strength = state_dependent_exploration_configuration[
            "initial_reweighing_strength"
        ]
        self.reweighing_duration = state_dependent_exploration_configuration[
            "reweighing_duration"
        ]
        self.feature_extractor = state_dependent_exploration_configuration[
            "feature_extractor"
        ]
        self.device = state_dependent_exploration_configuration["device"]

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=state_dependent_exploration_configuration[
                "min_cluster_size"
            ],
            # max_cluster_size=250,
            min_samples=state_dependent_exploration_configuration["min_samples"],
            cluster_selection_epsilon=state_dependent_exploration_configuration[
                "cluster_selection_epsilon"
            ],
            # algorithm="best",
            cluster_selection_method=state_dependent_exploration_configuration[
                "cluster_selection_method"
            ],
            # leaf_size=state_dependent_exploration_configuration["leaf_size"],
            # connectivity="knn",
            metric=state_dependent_exploration_configuration["metric"],
            prediction_data=state_dependent_exploration_configuration[
                "prediction_data"
            ],
        )

        self.cluster_persistence = state_dependent_exploration_configuration[
            "cluster_persistence"
        ]

        # Pre-trained CNN Feature Extractor:
        if self.feature_extractor in ["CNN", "CNN + UMAP"]:
            with torch.inference_mode():
                self.weights = AlexNet_Weights.DEFAULT
                self.cnn_feature_extractor = CNNFeatureExtractor()
                self.cnn_feature_extractor.to(self.device)
                self.cnn_feature_extractor.load_state_dict(
                    self.weights.get_state_dict(progress=True, check_hash=True)
                )
                self.cnn_feature_extractor.eval()

                self.preprocess = self.weights.transforms()

        self.n_components = state_dependent_exploration_configuration["n_components"]
        self.umap = UMAP(
            n_components=state_dependent_exploration_configuration["n_components"]
        )
        self.number_of_feature_extractor_train_observations = (
            state_dependent_exploration_configuration[
                "number_of_feature_extractor_train_observations"
            ]
        )
        self.feature_normalizer = Normalizer(norm="l2")

        self.cluster_labels_checkpoint = None
        self.previous_number_of_states_clustered = 0
        self.cluster_associated_actions = {}
        self.entire_data_clustered = True

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

        feature_extractor_train_start_time = time.perf_counter()
        if self.feature_extractor is None:
            if self.verbose >= 1:
                print(
                    f"\n\n\033[1mUsing Original Observations for Clustering:\033[0m\n\n"
                )
        elif self.feature_extractor == "Pre-trained CNN":
            if self.verbose >= 1:
                print(
                    f"\n\n\033[1mUsing {self.feature_extractor} Feature Extractor for Clustering:\033[0m\n\n"
                )

        elif self.feature_extractor == "Pre-trained CNN + UMAP":
            if self.verbose >= 1:
                print(
                    f"\n\n\033[1mUsing {self.feature_extractor} Feature Extractor for Clustering:\033[0m\n\n"
                )

            self.training_env.reset()
            feature_extractor_train_observations = []

            while (
                len(feature_extractor_train_observations)
                < self.number_of_feature_extractor_train_observations
            ):
                observations, reward, dones, info = self.training_env.step(
                    [
                        self.action_space.sample()
                        for _ in range(self.training_env.num_envs)
                    ]
                )

                observations = [observation[-3:, :, :] for observation in observations]
                with torch.inference_mode():
                    observations = torch.tensor(
                        observations, dtype=torch.float32, device="cuda"
                    )
                    # print("Observation Tensor Size:", observations.size())
                    # self.conv0 = nn.Conv2d(4, 3, 1).to("cuda")
                    # observations = self.conv0(observations)
                    # print(f"Observation Tensor Size after Conv0: {observations.size()}")
                    observations = self.preprocess(observations)
                    # print(f"Observation Tensor Size after Preprocessing: {observations.size()}")
                    observations = self.cnn_feature_extractor.get_features(observations)
                    for observation in observations:
                        feature_extractor_train_observations.append(
                            cp.asarray(observation)
                        )
                    # print(f"Feature Extractor Training Observation: {feature_extractor_train_observations[0].shape}")

            self.umap.fit(cp.asarray(feature_extractor_train_observations))
            del feature_extractor_train_observations

            if self.verbose >= 1:
                print(
                    f"\n\n\033[1mTime to Train Feature Extractor:"
                    f" {time.perf_counter() - feature_extractor_train_start_time} \033[0m\n\n"
                )

        elif self.feature_extractor == "UMAP":
            if self.verbose >= 1:
                print(
                    f"\n\n\033[1mUsing {self.feature_extractor} Feature Extractor for Clustering:\033[0m\n\n"
                )

            self.training_env.reset()
            feature_extractor_train_observations = []
            while (
                len(feature_extractor_train_observations)
                < self.number_of_feature_extractor_train_observations
            ):
                observations, reward, dones, info = self.training_env.step(
                    [
                        self.action_space.sample()
                        for _ in range(self.training_env.num_envs)
                    ]
                )
                for observation in observations:
                    feature_extractor_train_observations.append(observation.flatten())

            self.umap.fit(np.asarray(feature_extractor_train_observations))

            if self.verbose >= 1:
                print(
                    f"\n\n\033[1mTime to Train Feature Extractor:"
                    f" {time.perf_counter() - feature_extractor_train_start_time} \033[0m\n\n"
                )
            del feature_extractor_train_observations

        elif self.feature_extractor == "Latent Policy":
            if self.verbose >= 1:
                print(
                    f"\n\n\033[1mUsing {self.feature_extractor} Feature Extractor for Clustering:\033[0m\n\n"
                )

        else:
            raise ValueError(
                f"Expected Feature Extractor from 1. None 2. Pre-trained CNN 3. UMAP 4. Pre-trained CNN + UMAP, 5. Latent Policy got "
                f"'{self.feature_extractor}'."
            )

        # # Test UMAP Batch Transform Time:
        # batch_transform_start = perf_counter()
        # embeddings = self.umap.transform(
        #     np.asarray(feature_extractor_train_observations)
        # )
        # print(f"Batch Transform Time: {perf_counter() - batch_transform_start}")
        # feature_extractor_train_observations = [embedding for embedding in embeddings]
        #
        # overall_running_data_start = perf_counter()
        # self.training_env.reset()
        # while len(feature_extractor_train_observations) < 100_000:
        #     observations, reward, dones, info = self.training_env.step(
        #         [self.action_space.sample() for _ in range(self.training_env.num_envs)]
        #     )
        #     for observation in observations:
        #         feature_extractor_train_observations.append(observation.flatten())
        #
        #     if len(feature_extractor_train_observations) % 10000 == 0:
        #         print(len(feature_extractor_train_observations))
        #         # print(np.asarray(feature_extractor_train_observations[:]).shape)
        #         embeddings = self.umap.transform(
        #             np.asarray(feature_extractor_train_observations[-10000:])
        #         )
        #         for j, embedding in enumerate(embeddings):
        #             feature_extractor_train_observations[
        #                 len(feature_extractor_train_observations) - 10_000 + j
        #             ] = embedding
        #
        #     # individual_transform_start = perf_counter()
        #     # for observation in observations:
        #     #     feature_extractor_train_observations.append(self.umap_cpu.transform(observation.reshape(1, -1)))
        #     # embeddings = self.umap.transform(
        #     #     np.asarray([observation.flatten() for observation in observations])
        #     # )
        #
        #     # print(
        #     #     f"Individual Transform Time: {perf_counter() - individual_transform_start}"
        #     # )
        #
        #     # nn_obs_start = perf_counter()
        #     # with torch.inference_mode():
        #     #     state_features = self.cnn_feature_extractor.get_features(
        #     #         torch.tensor(observations, device="cuda")
        #     #     )
        #     #     state_features = torch.tensor_split(
        #     #         state_features, state_features.size()[0]
        #     #     )
        #     # print(f"NN OBS Time: {perf_counter() - nn_obs_start}")
        #
        # clustering_start_time = time.perf_counter()
        # self.clusterer.fit(cp.asarray(feature_extractor_train_observations[:100_000]))
        # print(f"Clustering Time: {time.perf_counter() - clustering_start_time}")
        # print(
        #     f"Overall Running Batch Time: {perf_counter() - overall_running_data_start}"
        # )
        #
        # new_labels, _ = hdbscan.approximate_predict(
        #     self.clusterer, cp.asarray(self.states_observed[-1:])
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
        if (
            self.num_timesteps
            < self.reweighing_duration * self.locals["total_timesteps"]
        ):
            if self.feature_extractor is None:
                for observation in self.model._last_obs:
                    self.states_observed.append(observation)

            elif self.feature_extractor == "Latent Policy":

                start_storing_observation = perf_counter()
                observations = torch.tensor(self.model._last_obs, device="cpu")
                for observation in observations:
                    self.states_observed.append(observation / 255)
                if self.verbose >= 1:
                    print("Time to store observations on a step:", {perf_counter() - start_storing_observation})

            elif self.feature_extractor in ["CNN", "CNN + UMAP"]:
                # Testing Pre-trained CNN feature extractor.
                with torch.inference_mode():
                    img = self.model._last_obs
                    img = [img_[-3:, :, :] for img_ in img]
                    # img = torch.tensor(self.model._last_obs, dtype=torch.float32, device="cuda")
                    img = torch.tensor(img, dtype=torch.float32, device="cuda")

                    # img = self.conv0(img)
                    img = self.preprocess(img)
                    img = self.cnn_feature_extractor.get_features(img)
                    # print("Final Img Size", img.size())
                if self.feature_extractor == "CNN":
                    state_features = img
                else:
                    state_features = self.umap.transform(img)

                for state_feature in state_features:
                    self.states_observed.append(state_feature)
                # print(f"SF:", state_features.shape)
                # sys.exit()

            elif self.feature_extractor == "UMAP":
                # Testing UMAP features:
                state_features = self.umap.transform(
                    np.asarray(
                        [observation.flatten() for observation in self.model._last_obs]
                    )
                )
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

    def reweigh_action_probabilities(
        self, action_probabilities_, share_features_extractor=None, neural_network=None
    ):
        """
        This method...
        """

        # print(f"Length States Observed: "
        #       f"{len(self.states_observed), self.num_timesteps, self.locals['total_timesteps']}")
        # print(f"States Observed: {self.states_observed}")
        # print(f"Actions Taken: {len(self.actions_taken), self.actions_taken}")
        if (
            len(self.states_observed) < 100
            or self.num_timesteps
            > self.locals["total_timesteps"] * self.reweighing_duration
        ):
            print("Did this get executed?")
            return action_probabilities_.cpu().detach().numpy()
        action_probabilities_ = torch.tensor_split(
            action_probabilities_, action_probabilities_.size()[0]
        )
        action_probabilities_ = [
            action_probabilities.squeeze()
            for action_probabilities in action_probabilities_
        ]

        # print(f"Action Probabilities:", action_probabilities_)

        action_probabilities_to_return = []
        for state_index, action_probabilities in enumerate(action_probabilities_):
            # Exponential Decay:
            self.initial_reweighing_strength = 1 * (
                (0.01 / 1)
                ** (1 / (self.locals["total_timesteps"] * self.reweighing_duration))
            ) ** len(self.states_observed)

            action_probabilities = action_probabilities.cpu().detach().numpy()
            action_probabilities = {
                i: action_probabilities[i] for i in range(len(action_probabilities))
            }
            # print("Check 1", action_probabilities)

            if (
                len(self.states_observed) < 100
                or len(self.states_observed)
                > (self.cluster_persistence + 1)
                * self.previous_number_of_states_clustered
            ):
                # print("Check 2")
                self.previous_number_of_states_clustered = len(self.states_observed)
                if self.feature_extractor is None:
                    self.clusterer.fit(cp.asarray(self.states_observed))
                elif self.feature_extractor == "Latent Policy":
                    # TODO: Remove inference mode later. Default is inference mode.
                    with torch.inference_mode():
                        load_neural_network_parameters_start = time.perf_counter()
                        neural_network.sde_actor_cnn.load_state_dict(
                            neural_network.cnn.state_dict()
                        )
                        neural_network.sde_actor_linear.load_state_dict(
                            neural_network.linear.state_dict()
                        )

                        neural_network.sde_actor = nn.Sequential(
                            neural_network.sde_actor_cnn,
                            neural_network.sde_actor_linear,
                        )
                        neural_network.sde_actor.to(device="cpu")
                        if self.verbose >= 1:
                            print("Time to load NN parameters:", time.perf_counter() - load_neural_network_parameters_start)
                        # neural_network.sde_actor.load_state_dict(
                        #     neural_network.policy_net.state_dict()
                        # )
                        # features = feature_extractor(torch.tensor(self.states_observed, device=device))
                        # print(f"Features: {type(features), features.size(), features.requires_grad, device}")
                        if share_features_extractor:
                            # latent_pi, _ = neural_network(features)
                            # latent_pi, _ = neural_network(torch.stack(self.states_observed, dim=0))
                            forward_pass_start = time.perf_counter()
                            latent_pi = neural_network.forward_sde_actor(
                                torch.stack(self.states_observed, dim=0)
                            )
                            if self.verbose >= 1:
                                print(f"Time to forward pass all observations: {time.perf_counter() - forward_pass_start}")
                            # latent_pi, _ = neural_network(self.states_observed)
                            # print(f"Latent Pi: {type(latent_vf), latent_vf.size()}")
                        # else:
                        #     # pi_features, _ = torch.tensor(features)
                        #     pi_features, _ = torch.stack(self.states_observed, dim=0)
                        #     latent_pi = neural_network.forward_actor(pi_features)
                    # print(f"Features: {type(features), features.size()}")
                    # print(
                    #     f"Latent Pi: {type(latent_pi), latent_pi.size(), latent_pi.requires_grad}"
                    # )
                    latent_pi = latent_pi.cpu().detach()
                    self.clusterer.fit(cp.asarray(latent_pi))
                    # sys.exit()
                    # latent_pi = latent_pi.cpu().detach()
                elif self.feature_extractor == "UMAP":
                    self.clusterer.fit(cp.asarray(self.states_observed))
                    # norm_data = self.feature_normalizer.fit_transform(
                    #     self.states_observed
                    # )
                    # self.clusterer.fit(cp.asarray(norm_data))
                    # pw_distance = pairwise_distances(
                    #     np.asarray(self.states_observed), metric="cosine"
                    # )
                    # print(pw_distance.shape)
                    # self.clusterer.fit(pw_distance)

                else:
                    cluster_all_observations_start = time.perf_counter()
                    self.clusterer.fit(self.states_observed)
                    if self.verbose >= 1:
                        print(f"Time to cluster all observations: {time.perf_counter() - cluster_all_observations_start}")

                self.cluster_labels_checkpoint = self.clusterer.labels_
                self.entire_data_clustered = True

                if self.verbose >= 2:
                    print(f"\n\nTotal States Observed: {len(self.states_observed)}")
                    print(
                        f"Cluster Counts: {sorted(Counter(self.clusterer.labels_.tolist()).values(), reverse=True)}"
                    )
                    print(
                        f"Ordered Cluster Counts: {sorted(Counter(self.clusterer.labels_.tolist()).items())}"
                    )
                    print(
                        f"Total Number of Clusters: {len(np.unique(self.clusterer.labels_))}"
                    )
            else:
                # print("Check 3")
                if self.feature_extractor is None:
                    new_labels, _ = hdbscan.approximate_predict(
                        self.clusterer,
                        cp.asarray(
                            self.states_observed[
                                -self.training_env.num_envs + state_index
                            ].reshape(1, 4)
                        ),
                    )
                    # print("This is what should be executed.")
                elif self.feature_extractor == "Latent Policy":
                    # TODO: Remove inference mode later. Default is inference mode.
                    with torch.inference_mode():
                        if share_features_extractor:
                            # latent_pi, _ = neural_network(torch.stack(self.states_observed[-1:], dim=0))

                            # latent_pi = neural_network.forward_sde_actor(
                            #     torch.tensor(
                            #         self.states_observed[
                            #             -self.training_env.num_envs + state_index
                            #         ],
                            #         device="cpu",
                            #     ).unsqueeze(0)
                            # )
                            forward_pass_single_observation_start = time.perf_counter()
                            latent_pi = neural_network.forward_sde_actor(
                                self.states_observed[
                                    -self.training_env.num_envs + state_index
                                ].unsqueeze(0)
                            )
                            if self.verbose >= 1:
                                print(f"Time to forward pass a single observation: {time.perf_counter() - forward_pass_single_observation_start}")

                    cluster_single_observation_start = time.perf_counter()
                    new_labels, _ = hdbscan.approximate_predict(
                        self.clusterer,
                        cp.asarray(latent_pi),
                    )
                    if self.verbose >= 1:
                        print(f"Time to cluster a single observation: {time.perf_counter() - cluster_single_observation_start}")
                elif self.feature_extractor == "UMAP":
                    # new_labels, _ = hdbscan.approximate_predict(
                    #     self.clusterer,
                    #     cp.asarray(
                    #         self.feature_normalizer.transform(
                    #             self.states_observed[
                    #                 -self.training_env.num_envs + state_index
                    #             ].reshape(1, self.n_components)
                    #         )
                    #     ),
                    # )

                    # Changing the input ot the clusterer to be a NumPy array as a cp array was causing issues with
                    # illegal memory address access with CUDA.
                    new_labels, _ = hdbscan.approximate_predict(
                        self.clusterer,
                        np.asarray(
                            self.states_observed[
                                -self.training_env.num_envs + state_index
                            ].reshape(1, self.n_components)
                        ),
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
                    int(key): {action: 0 for action in range(self.action_space.n)}
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
                    # self.cluster_associated_actions[int(self.clusterer.labels_[-1])][
                    #     self.actions_taken[-1]
                    # ] += 1
                    self.cluster_associated_actions[int(self.clusterer.labels_[-1])][
                        self.actions_taken[-self.training_env.num_envs + state_index]
                    ] += 1
                except KeyError:
                    # print("Check 7")
                    print(
                        f"This error occurs when there's no outlier in the original clustering but the approximate "
                        f"predict results in an outlier."
                    )
                    print(self.actions_taken[-1])
                    print(self.clusterer.labels_)
                    print(self.cluster_associated_actions)
                    self.cluster_associated_actions[-1] = {
                        action: 0 for action in range(self.action_space.n)
                    }
                    # self.cluster_associated_actions[self.clusterer.labels_[-1]][
                    #     self.actions_taken[-1]
                    # ] += 1
                    self.cluster_associated_actions[int(self.clusterer.labels_[-1])][
                        self.actions_taken[-self.training_env.num_envs + state_index]
                    ] += 1

            # print(f"Cluster AA: {self.cluster_associated_actions}")
            keys = self.cluster_associated_actions[
                int(self.clusterer.labels_[-1])
            ].keys()
            values = self.cluster_associated_actions[
                int(self.clusterer.labels_[-1])
            ].values()
            # keys = self.cluster_associated_actions[
            #     int(self.clusterer.labels_[-self.training_env.num_envs + state_index])
            # ].keys()
            # values = self.cluster_associated_actions[
            #     int(self.clusterer.labels_[-self.training_env.num_envs + state_index])
            # ].values()
            # print(f"Keys: {keys}, Values: {values}")
            total_actions = sum(list(values))

            # TODO: TESTING NOT REWEIGHING THE ACTION PROBABILITIES BASED ON THE TOTAL ACTION COUNT AND INIDIVIDUAL
            #  ACTION COUNT.
            if total_actions > 50000 and min(list(values)) > 10000:
                action_probabilities = list(action_probabilities.values())
                # temp = [val for val in action_probabilities]
                # action_probabilities[0] = 1.0 - sum(action_probabilities[1:])
                action_probabilities_sum = sum(action_probabilities)
                if action_probabilities_sum != 1:
                    for i in range(len(action_probabilities)):
                        action_probabilities[i] = (
                            1.0 - action_probabilities_sum + action_probabilities[i]
                        )
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
                cluster_associated_actions_ratios.items(),
                key=lambda x: x[1],
                reverse=True,
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
                    (
                        next_action,
                        next_caa_ratio,
                    ) = sorted_cluster_associated_actions_ratios[j]
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
                        # print("Value equal to 0.", non_zero_values)
                    if value != 0:
                        # print("Value not equal to 0.", non_zero_values)
                        break
                non_zero_values = non_zero_values if non_zero_values != 0 else 1
                # print("Final Non-zero values:", non_zero_values)

                action_probability_difference = list(
                    np.asarray(action_probability_difference) / non_zero_values
                )
                action_probability_differences.append(action_probability_difference)
                sde_action_probabilities[action] -= np.sum(
                    action_probability_differences[-1]
                )
                index = 0
                for j in range(i + 1, len(sorted_cluster_associated_actions_ratios)):
                    (
                        next_action,
                        next_caa_ratio,
                    ) = sorted_cluster_associated_actions_ratios[j]
                    sde_action_probabilities[
                        next_action
                    ] += action_probability_difference[index]
                    index += 1

            # print("Check 10")
            sde_action_probabilities = list(sde_action_probabilities.values())
            # temp = [val for val in sde_action_probabilities]
            action_probabilities_sum = sum(sde_action_probabilities)
            if action_probabilities_sum != 1:
                for i in range(len(sde_action_probabilities)):
                    sde_action_probabilities[i] = (
                        1.0 - action_probabilities_sum + sde_action_probabilities[i]
                    )
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

    def __init__(
        self, eval_env: Union[gym.Env, VecEnv], store_raw_results, *args, **kwargs
    ):
        super().__init__(eval_env, *args, **kwargs)

        self.store_raw_results = store_raw_results
        if self.log_path:
            self.log_path = (
                self.log_path.replace("evaluations", "raw_evaluation_results")
                if store_raw_results
                else self.log_path.replace("evaluations", "evaluation_statistics")
            )

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
                    (
                        episode_rewards,
                        episode_lengths,
                    ) = self.generate_evaluation_statistics(
                        episode_rewards=episode_rewards, episode_lengths=episode_lengths
                    )
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
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
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
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
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
                (
                    self.evaluations_results[i],
                    self.evaluations_length[i],
                ) = self.generate_evaluation_statistics(
                    episode_rewards=self.evaluations_results[i],
                    episode_lengths=self.evaluations_length[i],
                )

            kwargs = {}
            # Save success log if present
            if len(self._is_success_buffer) > 0:
                self.evaluations_successes.append(self._is_success_buffer)
                kwargs = dict(successes=self.evaluations_successes)

            self.log_path = self.log_path.replace(
                "raw_evaluation_results", "evaluation_statistics"
            )
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
