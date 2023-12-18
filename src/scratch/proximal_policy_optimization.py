import sys
import time
from collections import defaultdict, Counter
import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
import logging
import hdbscan
#
# from cuml.cluster import hdbscan

from scipy import stats
from PIL import Image
from skimage.transform import resize

# import umap

from cuml.manifold import UMAP

# import cupy as cp
# import numba as nb

logging.captureWarnings(True)


class ProximalPolicyOptimization:
    """This class implements the Proximal Policy Optimization algorithm."""

    def __init__(self, ppo_configuration):
        """
        This class initializes the PPO parameters.

        :param ppo_configuration: Dictionary - Dictionary containing the PPO configuration.
        """

        self.device = (
            "cuda"
            if (
                    torch.cuda.is_available()
                    and ppo_configuration["device"] in ["cuda", "auto"]
            )
            else "cpu"
        )

        self.timesteps_per_batch = ppo_configuration["timesteps_per_batch"]
        self.max_timesteps_per_episode = ppo_configuration["max_timesteps_per_episode"]

        # Number of times to update actor/critic per iteration.
        self.number_of_epochs_per_iteration = ppo_configuration[
            "number_of_epochs_per_iteration"
        ]

        self.actor_learning_rate = ppo_configuration["actor_learning_rate"]
        self.critic_learning_rate = ppo_configuration["critic_learning_rate"]

        self.gamma = ppo_configuration["gamma"]
        self.clip = ppo_configuration["clip"]
        self.gae_lambda = ppo_configuration["gae_lambda"]
        self.minibatch_size = ppo_configuration["minibatch_size"]
        self.entropy_coefficient = ppo_configuration["entropy_coefficient"]
        self.target_kl = ppo_configuration["target_kl"]
        self.max_grad_norm = ppo_configuration["max_grad_norm"]

        self.render_environment_training = ppo_configuration[
            "render_environment_training"
        ]
        self.render_environment_evaluation = ppo_configuration[
            "render_environment_evaluation"
        ]
        self.save_freq = ppo_configuration["save_freq"]

        # TODO: Add a method for evaluation with metrics such as min, max, median, mode, std, quartiles
        #  of reward and episode timesteps both during training and testing.
        self.deterministic = False  # If we're testing, don't sample actions
        self.seed = ppo_configuration["seed"]

        self.environment = ppo_configuration["environment"]
        self.observation_space = self.environment.observation_space
        self.action_space = self.environment.action_space
        self.action_space_type = ppo_configuration["action_space_type"]
        self.force_image_observation = ppo_configuration["force_image_observation"]

        if self.force_image_observation:
            self.environment.reset()
            # observation = self.environment.render()[150:350, 200:400, 0] / 255
            # print(np.min(observation), np.max(observation))
            # plt.imshow(observation, interpolation="nearest")
            # plt.show()
            # sys.exit()
            self.observation_shape = resize(self.environment.render(), (80, 80)).shape[
                                     :2
                                     ]
            # self.observation_shape = self.environment.render().shape[:2]
            print("Init Observation Shape:", self.observation_shape)
        else:
            # self.observation_shape = self.observation_space.shape[0]

            # TODO: ATARI:
            # observation, _ = self.environment.reset()
            # plt.imshow(observation[33:193, :, 1], interpolation="nearest")
            # plt.show()
            # sys.exit()
            self.observation_shape = (80, 80)

        # TODO: Automatically handle these instead of passing in the "action_space_type" parameter.
        # Handling the different action spaces depending on the environment.
        if self.action_space_type == "discrete":
            self.action_shape = self.action_space.n
        elif self.action_space_type == "multi_discrete":
            self.action_shape = self.action_space.nvec

        self.stack_observations = ppo_configuration["stack_observations"]
        self.number_of_observations_to_stack = ppo_configuration[
            "number_of_observations_to_stack"
        ]

        self.policy = ppo_configuration["policy"]
        self.use_state_dependent_exploration = ppo_configuration[
            "use_state_dependent_exploration"
        ]

        if not self.stack_observations:
            self.actor = self.policy(
                self.observation_shape,
                self.action_shape,
                model="actor",
                # action_space_type=self.action_shape_type,
            )
            self.critic = self.policy(
                self.observation_shape,
                1,
                model="critic",
                # action_space_type=self.action_shape_type,
            )
            self.sde_actor = self.policy(
                self.observation_shape,
                self.action_shape,
                model="sde_actor",
                # action_space_type=self.action_shape_type,
            )
            self.sde_actor.load_state_dict(self.actor.state_dict())

        else:
            # # Stacking with a MLP Policy.
            # self.actor = self.policy_class(
            #     self.number_of_observations_to_stack * self.observation_shape,
            #     self.action_shape,
            #     model="actor",
            # )
            # self.critic = self.policy_class(
            #     self.number_of_observations_to_stack * self.observation_shape,
            #     1,
            #     model="critic",
            # )

            # Stacking with CNN Policy.
            self.actor = self.policy(
                self.number_of_observations_to_stack * self.observation_shape,
                self.action_shape,
                model="actor",
            )
            self.critic = self.policy(
                self.number_of_observations_to_stack * self.observation_shape,
                1,
                model="critic",
            )

            if self.use_state_dependent_exploration:
                self.sde_actor = self.policy(
                    self.number_of_observations_to_stack * self.observation_shape,
                    self.action_shape,
                    model="sde_actor",
                )
                self.sde_actor.load_state_dict(self.actor.state_dict())

            # # Stacking with an LSTM or Transformer Policy.
            # self.actor = self.policy_class(
            #     in_dim=self.observation_shape,
            #     out_dim=self.action_shape,
            #     model="actor",
            # )
            # self.critic = self.policy_class(
            #     in_dim=self.observation_shape,
            #     out_dim=1,
            #     model="critic",
            # )

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        # if self.use_state_dependent_exploration:
        #     self.sde_actor = self.sde_actor.to(self.device)

        # Initial Warmup for UMAP
        # self.umap = umap.UMAP(n_components=8)
        self.umap = UMAP(n_components=8)
        self.environment.reset()
        self.number_of_original_clustering_samples = 20_000
        warmup_observations = []
        raw_observation_list = []

        for i in range(self.number_of_original_clustering_samples):
            observation, reward, terminated, truncated, info = self.environment.step(
                self.action_space.sample()
            )
            # TODO: Compare the computation time and performance of resizing the images vs using the images directly.
            observation = resize(
                observation[33:193, :, 0] / 255,
                (80, 80),
                anti_aliasing=False,
            )

            if self.stack_observations:
                raw_observation_list.append(observation)
                observation = self.observation_stacker(raw_observation_list)

            warmup_observations.append(np.asarray(observation).flatten())

            done = terminated or truncated
            if done:
                self.environment.reset()

        self.umap.fit(np.asarray(warmup_observations))
        clusterer = hdbscan.HDBSCAN(
            # clusterer = HDBSCAN(
            # min_cluster_size=min(25, cluster_size),
            min_cluster_size=5,
            # max_cluster_size=250,  # Doesn't seem to effect the results much.
            min_samples=50,  # Use along with min_cluster_size.
            cluster_selection_epsilon=-0.0,  # Leave at 0 for more clusters.
            cluster_selection_method="leaf",  # Try "leaf" for more clusters. "eom" for less clusters.
            # leaf_size=10,  # Doesn't seem to effect the results much.
            metric="euclidean",  # Test
            # core_dist_n_jobs=32,  # Number of parallel jobs. (Increase to CPU count)
            prediction_data=True,
        )
        # self.batch_observations_test = cp.asarray(self.batch_observations_test)
        # self.batch_observations_test = nb.cuda.to_device(self.batch_observations_test)
        warmup_observations = self.umap.transform(np.asarray(warmup_observations))
        clustering_start = time.perf_counter()
        # clusterer.fit(self.batch_observations_test)
        clusterer.fit(np.asarray(warmup_observations))
        print(f"Clustering Time: {time.perf_counter() - clustering_start}")
        sys.exit()
        del warmup_observations, raw_observation_list

        self.environment.reset()
        self.batch_observations_test = []
        raw_observation_list = []
        # self.number_of_observations_to_stack = 1

        # observation = np.asarray(
        #     [
        #         resize(
        #             self.environment.reset()[0][33:193, :, 0] / 255,
        #             (80, 80),
        #             anti_aliasing=False,
        #         )
        #         for _ in range(4)
        #     ]
        # )
        # print(f"Observation Shape:", observation.shape)
        # observation = observation.flatten()
        # print(f"Observation Shape:", observation.shape)

        number_of_original_clustering_samples = 10_000
        for _ in range(number_of_original_clustering_samples):
            observation, reward, terminated, truncated, info = self.environment.step(
                self.action_space.sample()
            )
            # TODO: Compare the computation time and performance of resizing the images vs using the images directly.
            observation = resize(
                observation[33:193, :, 0] / 255,
                (80, 80),
                anti_aliasing=False,
            )

            if self.stack_observations:
                raw_observation_list.append(observation)
                observation = self.observation_stacker(raw_observation_list)
                if len(raw_observation_list) > self.number_of_observations_to_stack:
                    raw_observation_list = raw_observation_list[
                                           -self.number_of_observations_to_stack:
                                           ]

            self.batch_observations_test.append(np.asarray(observation).flatten())
            done = terminated or truncated
            if done:
                self.environment.reset()

        # Testing UMAP for dimensionality reduction:
        # reducer = umap.UMAP(n_components=8)
        reducer = UMAP(n_components=8)
        dr_start = time.perf_counter()
        reducer.fit(np.asarray(self.batch_observations_test))
        print(f"Dimensionality Reduction Time: {time.perf_counter() - dr_start}")
        batch_transform_start = time.perf_counter()
        embeddings = reducer.transform(np.asarray(self.batch_observations_test))
        print(f"Batch Transform Time: {time.perf_counter() - batch_transform_start}")

        self.batch_observations_test = [embedding for embedding in embeddings]

        # observation = observation.reshape(1, 80 * 80)
        # number_of_original_clustering_samples = 10
        overall_running_data_start = time.perf_counter()
        for i in range(90_000):
            # individual_transform_start = time.perf_counter()
            # self.batch_observations_test.append(
            #     reducer.transform(observation.reshape(1, 4 * 80 * 80)).ravel()
            # )
            # print(
            #     f"Individual Transform Time: {time.perf_counter() - individual_transform_start}"
            # )
            # if (i + 1) % 1000 == 0:
            #     print(i + 1)

            observation, reward, terminated, truncated, info = self.environment.step(
                self.action_space.sample()
            )
            # TODO: Compare the computation time and performance of resizing the images vs using the images directly.
            observation = resize(
                observation[33:193, :, 0] / 255,
                (80, 80),
                anti_aliasing=False,
            )

            if self.stack_observations:
                raw_observation_list.append(observation)
                observation = self.observation_stacker(raw_observation_list)
                if len(raw_observation_list) > self.number_of_observations_to_stack:
                    raw_observation_list = raw_observation_list[
                                           -self.number_of_observations_to_stack:
                                           ]

            self.batch_observations_test.append(np.asarray(observation).flatten())
            done = terminated or truncated
            if done:
                self.environment.reset()

            if (i + 1) % number_of_original_clustering_samples == 0:
                running_batch_transform_start = time.perf_counter()
                embeddings = reducer.transform(
                    np.asarray(
                        self.batch_observations_test[
                        +i + 1: number_of_original_clustering_samples + i + 1
                        ]
                    )
                )
                self.batch_observations_test[
                +i + 1: number_of_original_clustering_samples + i + 1
                ] = [embedding for embedding in embeddings]
                gc.collect()
                del embeddings
                print(
                    f"Running Batch Transform Time: {time.perf_counter() - running_batch_transform_start}"
                )

        # print(
        #     f"Overall Running Batch Time: {time.perf_counter() - overall_running_data_start}"
        # )
        # clusterer = hdbscan.HDBSCAN(
        #     # clusterer = HDBSCAN(
        #     # min_cluster_size=min(25, cluster_size),
        #     min_cluster_size=5,
        #     # max_cluster_size=250,  # Doesn't seem to effect the results much.
        #     min_samples=50,  # Use along with min_cluster_size.
        #     cluster_selection_epsilon=-0.0,  # Leave at 0 for more clusters.
        #     cluster_selection_method="leaf",  # Try "leaf" for more clusters. "eom" for less clusters.
        #     # leaf_size=10,  # Doesn't seem to effect the results much.
        #     metric="euclidean",  # Test
        #     # core_dist_n_jobs=32,  # Number of parallel jobs. (Increase to CPU count)
        #     prediction_data=True,
        # )
        #
        # # self.batch_observations_test = cp.asarray(self.batch_observations_test)
        # # self.batch_observations_test = nb.cuda.to_device(self.batch_observations_test)
        # clustering_start = time.perf_counter()
        # # clusterer.fit(self.batch_observations_test)
        # clusterer.fit(np.asarray(self.batch_observations_test))
        # print(f"Clustering Time: {time.perf_counter() - clustering_start}")
        #
        # self.environment.reset()
        # for i in range(10_000):
        #     observation, reward, terminated, truncated, info = self.environment.step(
        #         self.action_space.sample()
        #     )
        #     # TODO: Compare the computation time and performance of resizing the images vs using the images directly.
        #     observation = resize(
        #         observation[33:193, :, 0] / 255,
        #         (80, 80),
        #         anti_aliasing=False,
        #     )
        #
        #     if self.stack_observations:
        #         raw_observation_list.append(observation)
        #         observation = self.observation_stacker(raw_observation_list)
        #         if len(raw_observation_list) > self.number_of_observations_to_stack:
        #             raw_observation_list = raw_observation_list[
        #                 -self.number_of_observations_to_stack :
        #             ]
        #
        #         observation = reducer.transform(
        #             np.asarray(observation).reshape(1, 80 * 80 * 4)
        #         )
        #
        #     approx_predict_start = time.perf_counter()
        #     label = hdbscan.approximate_predict(clusterer, observation.reshape(1, 8))
        #     print(
        #         f"Approximate Predict Time: {time.perf_counter() - approx_predict_start}"
        #     )
        #
        # self.batch_observations_test = torch.tensor(
        #     self.batch_observations_test,
        #     # .reshape(
        #     # (
        #     #     -1,
        #     #     1,
        #     #     self.number_of_observations_to_stack,
        #     #     self.observation_shape[0],
        #     #     self.observation_shape[1],
        #     # )
        #     # ),
        #     dtype=torch.float,
        #     device=self.device,
        # )
        # print(self.batch_observations_test.shape)
        # t = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # r = torch.cuda.memory_reserved(0) / (1024**3)
        # a = torch.cuda.memory_allocated(0) / (1024**3)
        # f = (r - a) / (1024**3)  # free inside reserved
        #
        # print("Total GPU Memory:", round(t, 4))
        # print("Reserved GPU Memory:", round(r, 4))
        # print("Allocated GPU Memory:", round(a, 4))
        # print("Free GPU Memory:", round(f, 4), round(t - r, 4), round(t - a, 4))
        # from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex
        #
        # nvmlInit()
        # h = nvmlDeviceGetHandleByIndex(0)
        # info = nvmlDeviceGetMemoryInfo(h)
        # print(f"total    : {round(info.total / (1024**3), 4)}")
        # print(f"free     : {round(info.free / (1024**3), 4)}")
        # print(f"used     : {round(info.used / (1024**3), 4)}")
        # sys.exit()
        total_params = sum(p.numel() for p in self.actor.parameters())

        self.verbose = ppo_configuration["verbose"]

        if self.verbose:
            print(f"Number of parameters: {total_params}")

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        # self.actor_optim = Adam(self.actor.parameters())
        # self.critic_optim = Adam(self.critic.parameters())

        # TODO: ADD IN THE CODE TO HANDLE DIFFERENT OBSERVATION AND ACTION SPACES.
        # Initialize the covariance matrix used to query the actor for actions.
        self.cov_var = torch.full(size=(self.action_shape,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            "delta_t": time.time_ns(),
            "timesteps_counter": 0,  # timesteps so far
            "iterations_counter": 0,  # iterations so far
            "batch_lengths": [],  # episodic lengths in batch
            "batch_rewards": [],  # episodic returns in batch
            "actor_losses": [],  # losses of actor network in current iteration
            "lr": 0,
        }

        self.states_observed = []
        self.actions_taken = []
        self.reweighing_strength = 1
        self.cluster_labels_checkpoint = None
        self.previous_number_of_states_clustered = 0
        self.cluster_associated_actions = {}
        self.entire_data_clustered = True  # Flag for efficient SDE.

        self.total_timesteps = ppo_configuration["total_timesteps"]
        self.evaluate_performance_bool = ppo_configuration["evaluate_performance_bool"]
        self.number_of_evaluation_episodes = ppo_configuration[
            "number_of_evaluation_episodes"
        ]
        self.evaluate_performance_every = ppo_configuration[
            "evaluate_performance_every"
        ]
        self.training_episode_rewards = []
        self.training_episode_timesteps = []
        self.evaluation_episode_rewards = []
        self.evaluation_episode_timesteps = []
        self.global_timestep_counter = 0

    def state_dependent_exploration(self, action_probabilities):
        """
        This method...
        """

        if (
                len(self.states_observed) < 10
                or self.global_timestep_counter > self.total_timesteps * 0.5
        ):
            action_probabilities = action_probabilities.cpu().detach().numpy()
            return action_probabilities

        # TODO: TEST THE DIFFERENT APPROACHES TO REWEIGHING. ALSO TRY TO MAKE REWEIGHING STRENGTH DEPENDENT ON THE
        #  REWARDS.
        # Linear:
        # self.reweighing_strength = 1 * (
        #     (self.total_timesteps - len(self.states_observed)) / self.total_timesteps
        # )
        # Exponential Decay:
        self.reweighing_strength = 1 * (
                (0.01 / 1) ** (1 / (self.total_timesteps * 0.5))
        ) ** len(self.states_observed)
        # self.reweighing_strength = 1
        # Dumb way that decreases the reweighing strength rapidly.
        # self.reweighing_strength = self.reweighing_strength * (
        #     (self.total_timesteps - len(self.states_observed)) / self.total_timesteps
        # )
        action_probabilities = action_probabilities.cpu().detach().numpy()
        action_probabilities = {
            i: action_probabilities[i] for i in range(len(action_probabilities))
        }
        time_print = 300_000

        if (
                len(self.states_observed) < 100
                or len(self.states_observed)
                > 1.1 * self.previous_number_of_states_clustered
        ):
            # clustering_start_time = time.perf_counter()
            # cluster_size = (
            #     int(np.ceil(len(self.states_observed) / 10))
            #     if len(self.states_observed) < 100
            #     else len(self.states_observed)
            # )
            self.previous_number_of_states_clustered = len(self.states_observed)
            self.clusterer = hdbscan.HDBSCAN(
                # min_cluster_size=min(25, cluster_size),
                min_cluster_size=25,
                # max_cluster_size=250,  # Doesn't seem to effect the results much.
                min_samples=5,  # Use along with min_cluster_size.
                cluster_selection_epsilon=-0.0,  # Leave at 0 for more clusters.
                cluster_selection_method="eom",  # Try "leaf" for more clusters. "eom" for less clusters.
                leaf_size=10,  # Doesn't seem to effect the results much.
                metric="euclidean",  # Test
                core_dist_n_jobs=32,  # Number of parallel jobs. (Increase to CPU count)
                prediction_data=True,
            )
            # TODO: TESTING OUT CLUSTERING BASED ON THE OUTPUT OF THE SDE ACTOR NETWORK.
            # if self.global_timestep_counter % 10000 == 0:
            nn_obs_start = time.perf_counter()
            copy_state_dict_start = time.perf_counter()
            self.sde_actor.load_state_dict(self.actor.state_dict())
            if len(self.actions_taken) > time_print:
                print(
                    f"\n\n\n\n\nCopy State Dict: {time.perf_counter() - copy_state_dict_start}"
                )

            if self.stack_observations:
                if self.force_image_observation:
                    # print(
                    #     "stacked observation shape before:",
                    #     np.asarray(observation).shape,
                    # )
                    nn_observations = np.asarray(self.states_observed).reshape(
                        (
                            -1,
                            1,
                            self.number_of_observations_to_stack,
                            self.observation_shape[0],
                            self.observation_shape[1],
                        )
                    )
                else:
                    # nn_observations = np.asarray(self.states_observed).reshape(
                    #     (
                    #         -1,
                    #         self.number_of_observations_to_stack,
                    #         self.observation_shape,
                    #     )
                    # )

                    # TODO: Atari:
                    nn_obs_reshape = time.perf_counter()
                    nn_observations = np.asarray(self.states_observed).reshape(
                        (
                            -1,
                            1,
                            self.number_of_observations_to_stack,
                            self.observation_shape[0],
                            self.observation_shape[1],
                        )
                    )
                    if len(self.actions_taken) > time_print:
                        print(f"NN OBS Reshape: {time.perf_counter() - nn_obs_reshape}")
            else:
                nn_observations = np.asarray(self.states_observed)

            with torch.inference_mode():
                nn_obs_tensor = time.perf_counter()
                nn_observations = torch.tensor(
                    nn_observations, dtype=torch.float, device="cpu"
                )
                # nn_observations = torch.tensor(
                #     nn_observations, dtype=torch.float, device=self.device
                # )
                if len(self.actions_taken) > time_print:
                    print(f"NN OBS Tensor: {time.perf_counter() - nn_obs_tensor}")
                nn_forward_pass_start = time.perf_counter()
                nn_observations = self.sde_actor(nn_observations)
                if len(self.actions_taken) > time_print:
                    print(
                        f"NN Forward Pass: {time.perf_counter() - nn_forward_pass_start}"
                    )

                # torch.cuda.synchronize()
                nn_obs_numpy_start = time.perf_counter()
                nn_observations = nn_observations.numpy()
                # nn_observations = nn_observations.cpu().numpy()
                if len(self.actions_taken) > time_print:
                    print(f"NN OBS Numpy: {time.perf_counter() - nn_obs_numpy_start}")
                    print(f"NN OBS Overall: {time.perf_counter() - nn_obs_start}")
            # print(type(nn_observations), nn_observations.shape)

            # self.clusterer.fit(self.states_observed)

            # TODO: Outlier reduction and NN OBS
            clustering_start_time = time.perf_counter()
            self.clusterer.fit(nn_observations)
            if len(self.actions_taken) > time_print:
                print(
                    "Clustering Time:",
                    time.perf_counter() - clustering_start_time,
                )
            self.cluster_labels_checkpoint = self.clusterer.labels_
            self.entire_data_clustered = True
            del nn_observations
            print(
                f"Cluster Counts: {sorted(Counter(self.clusterer.labels_).values(), reverse=True)}"
            )
            print(
                f"Ordered Cluster Counts: {sorted(Counter(self.clusterer.labels_).items())}"
            )
            print(f"Total Number of Clusters: {len(np.unique(self.clusterer.labels_))}")
        else:
            # clustering_prediction_start_time = time.perf_counter()
            # self.clusterer.generate_prediction_data()
            nn_ob_start = time.perf_counter()
            with torch.inference_mode():
                if self.force_image_observation:
                    nn_observation = np.asarray(self.states_observed[-1:]).reshape(
                        (
                            -1,
                            1,
                            self.number_of_observations_to_stack,
                            self.observation_shape[0],
                            self.observation_shape[1],
                        )
                    )
                else:
                    # nn_observation = np.asarray(self.states_observed[-1:])

                    # TODO: Atari:
                    nn_observation = np.asarray(self.states_observed[-1:]).reshape(
                        (
                            -1,
                            1,
                            self.number_of_observations_to_stack,
                            self.observation_shape[0],
                            self.observation_shape[1],
                        )
                    )
                nn_observation = torch.tensor(
                    nn_observation, dtype=torch.float, device="cpu"
                )
                # nn_observation = torch.tensor(
                #     nn_observation, dtype=torch.float, device=self.device
                # )
                # nn_observation = self.sde_actor(torch.tensor(self.states_observed[-1:]))
                nn_observation = self.sde_actor(nn_observation)
                nn_observation = nn_observation.numpy()
                # nn_observation = nn_observation.cpu().numpy()
            if len(self.actions_taken) > time_print:
                print(f"NN OB Overall: {time.perf_counter() - nn_ob_start}")
            # print(
            #     "singular", type(nn_observation), nn_observation, nn_observation.shape
            # )

            # new_labels, _ = hdbscan.approximate_predict(
            #     self.clusterer, self.states_observed[-1:]
            # )
            clustering_prediction_start_time = time.perf_counter()
            new_labels, _ = hdbscan.approximate_predict(self.clusterer, nn_observation)
            if len(self.actions_taken) > time_print:
                print(
                    "Clustering Prediction Time:",
                    time.perf_counter() - clustering_prediction_start_time,
                )
            del nn_observation
            # print("New labels, _", new_labels, _)
            concatenation_start_time = time.perf_counter()
            # print(f"TEST: {type(self.clusterer.labels_), type(self.clusterer.labels_[0]), type(new_labels)}")
            self.clusterer.labels_ = np.concatenate(
                (self.clusterer.labels_, new_labels)
            )
            # self.clusterer.labels_ = np.concatenate(
            #     (self.clusterer.labels_, cp.asarray(new_labels))
            # )
            if len(self.actions_taken) > time_print:
                print(
                    "Cluster Label Concatenation Time:",
                    time.perf_counter() - concatenation_start_time,
                )
            # print("concataned labels,", self.clusterer.labels_)
            self.entire_data_clustered = False

        sde_start_time = time.perf_counter()
        if self.entire_data_clustered:
            # cluster_associate_actions_new_start_time = time.perf_counter()
            # Commenting for CuML
            # self.cluster_associated_actions = {
            #     key: {action: 0 for action in range(self.environment.action_space.n)}
            #     for key in np.unique(self.clusterer.labels_)
            # }
            self.cluster_associated_actions = {
                int(key): {
                    action: 0 for action in range(self.environment.action_space.n)
                }
                for key in np.unique(self.clusterer.labels_)
            }

            # for i, action_taken in enumerate(self.actions_taken):
            #     self.cluster_associated_actions[self.clusterer.labels_[i]][
            #         action_taken
            #     ] += 1
            for i, action_taken in enumerate(self.actions_taken):
                self.cluster_associated_actions[int(self.clusterer.labels_[i])][
                    action_taken
                ] += 1

            # if len(self.actions_taken) > 50_000:
            #     print(
            #         "CAA New Time:",
            #         time.perf_counter() - cluster_associate_actions_new_start_time,
            #     )
        else:
            # cluster_associate_actions_update_start_time = time.perf_counter()
            self.cluster_associated_actions[int(self.clusterer.labels_[-1])][
                self.actions_taken[-1]
            ] += 1

            # if len(self.actions_taken) > 50_000:
            #     print(
            #         "CAA Update Time:",
            #         time.perf_counter() - cluster_associate_actions_update_start_time,
            #     )
        # TODO: Lmao this should be the total for the cluster not the overall total.
        # total_actions = len(self.actions_taken)

        # keys = self.cluster_associated_actions[self.clusterer.labels_[-1]].keys()
        # values = self.cluster_associated_actions[self.clusterer.labels_[-1]].values()
        keys = self.cluster_associated_actions[int(self.clusterer.labels_[-1])].keys()
        values = self.cluster_associated_actions[
            int(self.clusterer.labels_[-1])
        ].values()
        total_actions = sum(list(values))

        # TODO: TESTING NOT REWEIGHING THE ACTION PROBABILITIES BASED ON THE TOTAL ACTION COUNT AND INIDIVIDUAL
        #  ACTION COUNT.
        if total_actions > 10000 and min(list(values)) > 1000:
            return list(action_probabilities.values())

        cluster_associated_actions_ratios = {
            key: value / total_actions for key, value in zip(keys, values)
        }
        sde_action_probabilities = {
            i: action_probabilities[i] for i in range(len(action_probabilities))
        }

        sorted_cluster_associated_actions_ratios = sorted(
            cluster_associated_actions_ratios.items(), key=lambda x: x[1], reverse=True
        )
        action_probability_differences = []

        for i in range(len(sorted_cluster_associated_actions_ratios)):
            action, caa_ratio = sorted_cluster_associated_actions_ratios[i]
            action_probability = action_probabilities[action]
            action_probability_difference = []
            for j in range(i + 1, len(sorted_cluster_associated_actions_ratios)):
                next_action, next_caa_ratio = sorted_cluster_associated_actions_ratios[
                    j
                ]
                action_probability_difference.append(
                    self.reweighing_strength
                    * (caa_ratio - next_caa_ratio)
                    * action_probability
                )

            non_zero_values = len(action_probability_difference)
            for value in action_probability_difference:
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

            # sde_action_probabilities[0] = 1 - sde_action_probabilities[1]
            # sde_action_probabilities[0] = (
            #     1
            #     - sde_action_probabilities[1]
            #     - sde_action_probabilities[2]
            #     - sde_action_probabilities[3]
            # )
        if len(self.actions_taken) > time_print:
            print("SDE Time:", time.perf_counter() - sde_start_time)
        # print(
        #     f"Variable Sizes:\n States Observed: {sys.getsizeof(self.states_observed)}\n, NN Observations: "
        #     f"{sys.getsizeof(nn_observations)}"
        # )
        # print(f"\n\n\n\n\n\n\nOriginal Action Probabilities: {action_probabilities}")
        # print(
        #     f"Cluster Associate Actions: {self.cluster_associated_actions[self.clusterer.labels_[-1]]}"
        # )
        # print(f"Cluster Associated Action Ratios: {cluster_associated_actions_ratios}")
        # print(f"Action Probability Differences: {action_probability_differences}")
        # print(f"SDE Action Probabilities: {sde_action_probabilities}")
        # gc.collect()
        return list(sde_action_probabilities.values())

    def learn(self, total_timesteps):
        """
        Train the actor and critic networks. Here is where the main PPO algorithm resides.

        :param total_timesteps - Integer - Total number of timesteps to train the actor/critic.
        """

        timesteps_counter = 0
        iterations_counter = 0

        while timesteps_counter < total_timesteps:  # ALG STEP 2
            # batch_generation_start_time = time.time()
            (
                batch_observations,
                batch_actions,
                batch_action_probabilities,
                batch_log_probabilities,
                batch_rewards,
                batch_lengths,
                batch_values,
                batch_dones,
            ) = self.rollout()  # ALG STEP 3
            # print(
            #     f"Batch Generation Time: {time.time() - batch_generation_start_time} seconds"
            # )

            # Calculate advantage using GAE
            A_k = self.calculate_gae(batch_rewards, batch_values, batch_dones)
            # print(
            #     f"Batch Observations Shape: {batch_observations.shape}\n Batch Observation:\n "
            #     f"{batch_observations[0]}"
            # )

            # TODO: USING THIS FOR IMAGE BASED ENVIRONMENTS. AUTOMATE THIS.
            # batch_observations = batch_observations.reshape()
            # TODO: TESTING USING TORCH NO GRAD FOR MEMORY SAVING
            with torch.inference_mode():
                self.critic = self.critic.to("cpu")
                batch_observations = batch_observations.to("cpu")
                A_k = A_k.to("cpu")
                V = self.critic(batch_observations).squeeze()
            batch_rtgs = A_k + V.detach()
            batch_rtgs = batch_rtgs.to(self.device)
            A_k = A_k.to(self.device)
            self.critic = self.critic.to(self.device)
            batch_observations = batch_observations.to(self.device)

            # Calculate how many timesteps we collected this batch
            timesteps_counter += np.sum(batch_lengths)

            # Increment the number of iterations
            iterations_counter += 1

            # Logging timesteps so far and iterations so far
            self.logger["timesteps_counter"] = timesteps_counter
            self.logger["iterations_counter"] = iterations_counter

            # self.evaluate_performance()

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs

            step = batch_observations.size(0)
            inds = np.arange(step)
            loss = []
            training_start_time = time.time()
            for _ in range(self.number_of_epochs_per_iteration):  # ALG STEP 6 & 7
                # Learning Rate Annealing
                frac = (timesteps_counter - 1.0) / total_timesteps
                new_lr_actor = self.actor_learning_rate * (1.0 - frac)
                new_lr_critic = self.critic_learning_rate * (1.0 - frac)

                # Make sure learning rate doesn't go below 0
                new_lr_actor = max(new_lr_actor, 0.0)
                new_lr_critic = max(new_lr_critic, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr_actor
                self.critic_optim.param_groups[0]["lr"] = new_lr_critic
                # Log learning rate
                self.logger["lr"] = new_lr_actor
                # self.logger["lr_critic"] = new_lr_critic

                # Mini-batch Update
                # TODO: Determine if we want to shuffle the data for LSTM and Transformer Policies.
                np.random.shuffle(inds)  # Shuffling the index
                # approx_kl = None

                for start in range(0, step, self.minibatch_size):
                    end = start + self.minibatch_size
                    idx = inds[start:end]
                    # Extract data at the sampled indices
                    mini_obs = batch_observations[idx]
                    mini_acts = batch_actions[idx]
                    mini_batch_action_probabilities = batch_action_probabilities[idx]
                    mini_log_prob = batch_log_probabilities[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    # Calculate V_phi and pi_theta(a_t | s_t) and entropy
                    V, curr_log_probs, entropy = self.evaluate(
                        mini_obs, mini_acts, mini_batch_action_probabilities
                    )

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    # NOTE: we just subtract the logs, which is the same as
                    # dividing the values and then canceling the log with e^log.
                    # For why we use log probabilities instead of actual probabilities,
                    # here's a great explanation:
                    # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                    # TL;DR makes gradient descent easier behind the scenes.
                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()

                    # Calculate surrogate losses.
                    surr1 = ratios * mini_advantage
                    surr2 = (
                            torch.clamp(ratios, 1 - self.clip, 1 + self.clip)
                            * mini_advantage
                    )

                    # Calculate actor and critic losses.
                    # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                    # the performance function, but Adam minimizes the loss. So minimizing the negative
                    # performance function maximizes it.
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    # Entropy Regularization
                    entropy_loss = entropy.mean()
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - self.entropy_coefficient * entropy_loss

                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # Gradient Clipping with given threshold
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm
                    )
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())

                # Approximating KL Divergence
                if approx_kl > self.target_kl:
                    break  # if kl aboves threshold

            # print(f"Training Time: {time.time() - training_start_time} seconds")
            # print(f"Trained Iteration {iterations_counter}")
            # Log actor loss
            avg_loss = sum(loss) / len(loss)
            self.logger["actor_losses"].append(avg_loss)
            if self.evaluate_performance_bool:
                self.evaluate_performance()
                print(self.evaluation_episode_rewards[-1])
            # Print a summary of our training so far

            # t = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # r = torch.cuda.memory_reserved(0) / (1024**3)
            # a = torch.cuda.memory_allocated(0) / (1024**3)
            # f = (r - a) / (1024**3)  # free inside reserved

            # print("Before GC Total GPU Memory:       ", round(t, 4))
            # print("Before GC Reserved GPU Memory:    ", round(r, 4))
            # print("Before GC Allocated GPU Memory:   ", round(a, 4))
            # print(
            #     "Before GC Free GPU Memory:        ",
            #     round(t - r, 4),
            # )
            # Save our model if it's time
            # if iterations_counter % self.save_freq == 0:
            #     torch.save(self.actor.state_dict(), "./ppo_actor.pth")
            #     torch.save(self.critic.state_dict(), "./ppo_critic.pth")
            # del batch_observations
            # del batch_actions
            # del batch_action_probabilities
            # del batch_log_probabilities
            # del batch_rewards
            # del batch_lengths
            # del batch_values
            # del batch_dones
            # del V
            # del A_k
            # del batch_rtgs
            # del actor_loss
            # del critic_loss
            # del entropy_loss
            # del mini_obs
            # del mini_acts
            # del mini_batch_action_probabilities
            # del mini_log_prob
            # del mini_rtgs
            # del mini_advantage
            # del logratios
            # del ratios
            # del approx_kl
            # del surr1
            # del surr2
            # gc.collect()
            # print("After GC Total GPU Memory:       ", round(t, 4))
            # print("After GC Reserved GPU Memory:    ", round(r, 4))
            # print("After GC Allocated GPU Memory:   ", round(a, 4))
            # print(
            #     "After GC Free GPU Memory:        ",
            #     round(t - r, 4),
            # )
            self._log_summary()
        print("\n\n\nIteration Rewards:")
        for iteration_reward in self.evaluation_episode_rewards:
            print(iteration_reward)

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []  # List to store computed advantages for each timestep

        # Iterate over each episode's rewards, values, and done flags
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []  # List to store advantages for the current episode
            last_advantage = 0  # Initialize the last computed advantage

            # Calculate episode advantage in reverse order (from last timestep to first)
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Calculate the temporal difference (TD) error for the current timestep
                    delta = (
                            ep_rews[t]
                            + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1])
                            - ep_vals[t]
                    )
                else:
                    # Special case at the boundary (last timestep)
                    delta = ep_rews[t] - ep_vals[t]

                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = (
                        delta
                        + self.gamma * self.gae_lambda * (1 - ep_dones[t]) * last_advantage
                )
                last_advantage = (
                    advantage  # Update the last advantage for the next timestep
                )
                advantages.insert(
                    0, advantage
                )  # Insert advantage at the beginning of the list

            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)

        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float, device=self.device)

    def observation_stacker(self, raw_observations):
        """
        This method stacks the raw observations as per "self.number_of_observations_to_stack".

        :param raw_observations: Integer - The number of observations we will stack.

        Return: stacked_observation: List - Stacked observation.
        """

        stacked_observation = []
        if len(raw_observations) < self.number_of_observations_to_stack:
            padded_observation = np.zeros(raw_observations[0].shape)
            # print(f"Padded observations shape: {padded_observation.shape}")
            for j in range(
                    self.number_of_observations_to_stack - len(raw_observations)
            ):
                stacked_observation.append(padded_observation)

            for x in range(len(raw_observations)):
                stacked_observation.append(raw_observations[x])

        else:
            start_index = len(raw_observations) - self.number_of_observations_to_stack
            end_index = len(raw_observations)

            for j in range(start_index, end_index):
                stacked_observation.append(raw_observations[j])

        return stacked_observation

    def rollout(self):
        """
        This method collects a batch of observations, actions, log probabilities, rewards, state values,
        and done flags.

        :return batch_observations - Tensor -
        :return batch_actions - Tensor
        :return batch_log_probabilities - Tensor
        :return batch_rewards -
        :return batch_lengths -
        :return batch_state_values -
        :return batch_dones -
        """

        # TODO: MOVE RAW OBSERVATIONS TO THE WHILE LOOP BEFORE EACH EPISODE. OPTIMIZED. USE DEQUE OR ARRAY WITH
        #  LENGTH EQUAL TO THE NUMBER OF OBSERVATIONS WE WANT TO STACK.
        # raw_observation_list = []
        batch_observations = []
        batch_actions = []
        batch_log_probabilities = []
        batch_rewards = []
        batch_lengths = []
        batch_state_values = []
        batch_dones = []

        # TEST FOR USING THE SDE ACTION PROBABILITIES IN THE EVALUATE METHOD:
        batch_action_probabilities = []
        batch_timesteps_counter = 0

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while batch_timesteps_counter < self.timesteps_per_batch:
            episode_rewards = []
            episode_values = []
            episode_dones = []

            observation, _ = self.environment.reset()

            # TODO: Atari:
            observation = resize(
                (observation[33:193, :, 0] / 255 - 0.5) * 2,
                (self.observation_shape[0], self.observation_shape[1]),
                anti_aliasing=False,
            )
            # initial_get_image_observation_start_time = time.time()
            if self.force_image_observation:
                # observation = self.environment.render()[:, :, 0]
                observation = resize(
                    (self.environment.render()[150:350, 200:400, 0] / 255 - 0.5) * 2,
                    (self.observation_shape[0], self.observation_shape[1]),
                    anti_aliasing=False,
                )
                # print("Environment reset observation shape", observation.shape)
            # print(
            #     f"Initial Get Image Observation Time: {time.time() - initial_get_image_observation_start_time}"
            # )
            done = False
            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            # for episode_timestep_counter in range(self.max_timesteps_per_episode):

            episode_timestep_counter = 0
            raw_observation_list = []
            while (
                    episode_timestep_counter < self.max_timesteps_per_episode and not done
            ):
                # If render is specified, render the environment
                # if self.render_environment_training:
                #     self.environment.render()
                # Track done flag of the current state
                # each_timestep_start_time = time.time()
                episode_dones.append(done)

                episode_timestep_counter += 1
                batch_timesteps_counter += 1
                self.global_timestep_counter += 1
                # if batch_timesteps_counter % 1000 == 0:
                #     print(f"\n{batch_timesteps_counter}:")
                #     t = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                #     r = torch.cuda.memory_reserved(0) / (1024**3)
                #     a = torch.cuda.memory_allocated(0) / (1024**3)
                #     f = (r - a) / (1024**3)  # free inside reserved
                #
                #     print("Total GPU Memory:       ", round(t, 4))
                #     print("Reserved GPU Memory:    ", round(r, 4))
                #     print("Allocated GPU Memory:   ", round(a, 4))
                #     print(
                #         "Free GPU Memory:        ",
                #         round(t - r, 4),
                #     )
                #     from pynvml import (
                #         nvmlInit,
                #         nvmlDeviceGetMemoryInfo,
                #         nvmlDeviceGetHandleByIndex,
                #     )
                #
                #     nvmlInit()
                #     h = nvmlDeviceGetHandleByIndex(0)
                #     info = nvmlDeviceGetMemoryInfo(h)
                #     print(f"Total GPU Memory    : {round(info.total / (1024**3), 4)}")
                #     print(f"Free GPU Memory     : {round(info.free / (1024**3), 4)}")
                #     print(f"Used GPU Memory     : {round(info.used / (1024**3), 4)}")

                # TODO: Move this out of the inner while loop to evaluate performance after the episode has ended?
                #  Maybe just use a flag? Also experiment with doing this evaluation each time after training the
                #  actor/critic. Which one makes more sense?
                # if (
                #     self.evaluate_performance_bool
                #     and self.global_timestep_counter % self.evaluate_performance_every
                #     == 0
                # ):
                #     self.evaluate_performance()

                # Track observations in this batch
                # observation_stacker_start_time = time.time()
                if self.stack_observations:
                    raw_observation_list.append(observation)
                    observation = self.observation_stacker(raw_observation_list)
                    if len(raw_observation_list) > self.number_of_observations_to_stack:
                        raw_observation_list = raw_observation_list[
                                               -self.number_of_observations_to_stack:
                                               ]

                    # TODO: Determine the sequence shape for the different networks automatically.
                    # Uncomment this and commetn the below if self.stack_observations condition for MLP stacked
                    # policies.
                    # observation = np.asarray(observation).flatten()
                # print("batch observations element shape", np.asarray(observation).shape)
                # print(
                #     f"\n\n\n\nObservation Stacker Time: {time.time() - observation_stacker_start_time} seconds"
                # )
                batch_observations.append(observation)

                # TODO: TESTING USING THE OUTPUT OF THE LAST NEURAL NETWORK LAYER AS THE OBSERVATION FOR CLUSTERING.
                #  HANDLE EVERYTHING FROM STACKING TO TENSORS.
                # State dependent exploration
                if self.use_state_dependent_exploration:
                    self.states_observed.append(np.asarray(observation).flatten())
                    # if (
                    #     self.global_timestep_counter
                    #     % self.number_of_original_clustering_samples
                    #     == 0
                    # ):
                    #     embeddings = self.umap.transform(
                    #         self.states_observed[
                    #             self.global_timestep_counter : self.number_of_original_clustering_samples
                    #             + self.global_timestep_counter
                    #         ]
                    #     )
                    #     self.states_observed[
                    #         self.global_timestep_counter : self.number_of_original_clustering_samples
                    #         + self.global_timestep_counter
                    #     ] = [embedding for embedding in embeddings]

                # print(
                #     f"Observation:\n {observation}\n\n Observation Shape: {observation.shape}"
                # )
                # observation_reshape_start_time = time.time()
                if self.stack_observations:
                    if self.force_image_observation:
                        # print(
                        #     "stacked observation shape before:",
                        #     np.asarray(observation).shape,
                        # )
                        observation = np.asarray(observation).reshape(
                            (
                                -1,
                                1,
                                self.number_of_observations_to_stack,
                                self.observation_shape[0],
                                self.observation_shape[1],
                            )
                        )
                    else:
                        # observation = np.asarray(observation).reshape(
                        #     (
                        #         -1,
                        #         self.number_of_observations_to_stack,
                        #         self.observation_shape,
                        #     )
                        # )

                        # TODO: Atari:
                        observation = np.asarray(observation).reshape(
                            (
                                -1,
                                1,
                                self.number_of_observations_to_stack,
                                self.observation_shape[0],
                                self.observation_shape[1],
                            )
                        )

                # print(
                #     f"Observation Reshape Time: {time.time() - observation_reshape_start_time} seconds"
                # )
                # print("Stacked Observation Shape:", observation.shape)
                observation = torch.tensor(
                    observation, dtype=torch.float, device=self.device
                )
                # print(
                #     f"\n\nTensor Observation:\n {observation}\n\n Tensor Observation Shape: {observation.shape}"
                # )

                # predict_action_start_time = time.time()
                # Calculate action and make a step in the env.
                action_probabilities, action, log_prob = self.get_action(observation)
                # action_probabilities, action, log_prob = [0.5, 0.5], 0, 0.5
                # print(
                #     f"Action Prediction Time: {time.time() - predict_action_start_time} seconds"
                # )

                # State dependent exploration
                if self.use_state_dependent_exploration:
                    # actions_taken_append_start = time.perf_counter()
                    self.actions_taken.append(action)
                    # print(
                    #     f"Time Actions Taken Append: {time.perf_counter() - actions_taken_append_start}"
                    # )
                with torch.inference_mode():
                    val = self.critic(observation)
                # val = val.detach()
                # val = np.asarray([1])
                observation, reward, terminated, truncated, _ = self.environment.step(
                    action
                )
                # get_image_observation_start_time = time.time()
                if self.force_image_observation:
                    # raw_obs_start_time = time.time()
                    # observation = self.environment.render()[:, :, 0]
                    # print(f"Raw Observation Time: {time.time() - raw_obs_start_time}")
                    # resize_obs_start_time = time.time()
                    # observation = resize(
                    #     observation,
                    #     (self.observation_shape[0], self.observation_shape[1]),
                    #     anti_aliasing=False,
                    # )
                    # print(
                    #     f"Resized Observation Time: {time.time() - resize_obs_start_time}"
                    # )
                    observation = resize(
                        (self.environment.render()[150:350, 200:400, 0] / 255 - 0.5)
                        * 2,
                        (self.observation_shape[0], self.observation_shape[1]),
                        anti_aliasing=False,
                    )

                    # print("Environment Step Observation Shape:", observation.shape)
                # TODO: Atari:
                observation = resize(
                    (observation[33:193, :, 0] / 255 - 0.5) * 2,
                    (self.observation_shape[0], self.observation_shape[1]),
                    anti_aliasing=False,
                )
                # print(
                #     f"Get Image Observation Time: {time.time() - get_image_observation_start_time}"
                # )
                done = terminated or truncated
                # Track recent reward, action, and action log probability
                episode_rewards.append(reward)
                episode_values.append(val.flatten())
                # batch_actions.append(int(action))
                batch_actions.append(action)
                batch_log_probabilities.append(log_prob)

                # TEST TO USE THE SDE ACTION PROBABILITIES IN THE EVALUATE METHOD:
                batch_action_probabilities.append(action_probabilities)

                # If the environment tells us the episode is terminated, break
                if done:
                    break
                # print(
                #     f"Each Timestep Time: {time.time() - each_timestep_start_time} seconds"
                # )
            # Track episodic lengths, rewards, state values, and done flags
            batch_lengths.append(
                episode_timestep_counter + 0
            )  # + 1 before while loop replaced for loop
            batch_rewards.append(episode_rewards)
            batch_state_values.append(episode_values)
            batch_dones.append(episode_dones)

        # Reshape data as tensors in the shape specified in function description, before returning
        # TODO: FOR IMAGE BASED ENVIRONMENTS.
        if self.force_image_observation:
            batch_observations = torch.tensor(
                np.asarray(batch_observations).reshape(
                    (
                        -1,
                        1,
                        self.number_of_observations_to_stack,
                        self.observation_shape[0],
                        self.observation_shape[1],
                    )
                ),
                dtype=torch.float,
                device=self.device,
            )
        else:
            # batch_observations = torch.tensor(
            #     np.asarray(batch_observations), dtype=torch.float, device=self.device
            # )

            # TODO: Atari:
            batch_observations = torch.tensor(
                np.asarray(batch_observations).reshape(
                    (
                        -1,
                        1,
                        self.number_of_observations_to_stack,
                        self.observation_shape[0],
                        self.observation_shape[1],
                    )
                ),
                dtype=torch.float,
                device=self.device,
            )
        batch_actions = torch.tensor(
            batch_actions, dtype=torch.float, device=self.device
        )
        # TEST TO USE THE SDE ACTION PROBABILITIES IN THE EVALUATE METHOD:
        batch_action_probabilities = torch.tensor(
            np.asarray(batch_action_probabilities),
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        batch_log_probabilities = torch.tensor(
            batch_log_probabilities, dtype=torch.float, device=self.device
        ).flatten()

        # # TEST MOVING TO GPU:
        # print("Batch Rews:", batch_rewards[0], type(batch_rewards), type(batch_rewards[0]))
        # batch_rewards = torch.tensor(batch_rewards).to(self.device)
        # print("Batch Lens:", batch_lengths[0], type(batch_lengths), type(batch_lengths[0]))
        # batch_lengths = torch.tensor(batch_lengths).to(self.device)
        # print("Batch Vals:", batch_state_values[0], type(batch_state_values), type(batch_state_values[0]))
        # batch_state_values = torch.tensor(batch_state_values).to(self.device)
        # print("Batch Dones:", batch_dones[0], type(batch_dones), type(batch_dones[0]))
        # batch_dones = torch.tensor(batch_dones).to(self.device)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger["batch_rewards"] = batch_rewards
        self.logger["batch_lengths"] = batch_lengths

        # Here, we return the batch_rewards instead of batch_rtgs for later calculation of GAE
        # time.sleep(5)
        # print("\n\nTotal GPU Memory:       ", round(t, 4))
        # print("Reserved GPU Memory:    ", round(r, 4))
        # print("Allocated GPU Memory:   ", round(a, 4))
        # print(
        #     "Free GPU Memory:        ",
        #     round(t - r, 4),
        # )
        # print(f"Total GPU Memory    : {round(info.total / (1024**3), 4)}")
        # print(f"Free GPU Memory     : {round(info.free / (1024**3), 4)}")
        # print(f"Used GPU Memory     : {round(info.used / (1024**3), 4)}")
        # print(
        #     len(batch_observations),
        #     len(batch_actions),
        #     len(batch_action_probabilities),
        #     len(batch_log_probabilities),
        #     len(batch_rewards),
        #     len(batch_lengths),
        #     len(batch_state_values),
        #     len(batch_dones),
        # )
        # time.sleep(5)
        # print(f"Total GPU Memory    : {round(info.total / (1024**3), 4)}")
        # print(f"Free GPU Memory     : {round(info.free / (1024**3), 4)}")
        # print(f"Used GPU Memory     : {round(info.used / (1024**3), 4)}")
        # sys.exit()

        return (
            batch_observations,
            batch_actions,
            batch_action_probabilities,
            batch_log_probabilities,
            batch_rewards,
            batch_lengths,
            batch_state_values,
            batch_dones,
        )

    def get_action(self, obs):
        """
        Queries an action from the actor network, should be called from rollout.

        Parameters:
            obs - the observation at the current timestep

        Return:
            action - the action to take, as a numpy array
            log_prob - the log probability of the selected action in the distribution
        """
        # # Query the actor network for a mean action
        # obs = torch.tensor(obs, dtype=torch.float)
        # mean = self.actor(obs)
        #
        # # Create a distribution with the mean action and std from the covariance matrix above.
        # # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # # https://www.youtube.com/watch?v=JjB58InuTqM
        # dist = MultivariateNormal(mean, self.cov_mat)
        #
        # # Sample an action from the distribution
        # action = dist.sample()
        #
        # # Calculate the log probability for that action
        # log_prob = dist.log_prob(action)

        # DISCRETE ACTION SPACE MODIFICATION
        # print("Observation:", obs, obs.shape)
        action_probabilities = self.actor(obs)

        if self.verbose:
            print(f"Original Action Probabilities: {action_probabilities}")
        if self.use_state_dependent_exploration:
            # sde_ap_start = time.perf_counter()
            action_probabilities = self.state_dependent_exploration(
                action_probabilities
            )
            if self.verbose:
                print(f"SDE Action Probabilities: {action_probabilities}")
            action_probabilities = torch.tensor(
                action_probabilities, device=self.device
            )
            # print(f"Total SDE AP Time: {time.perf_counter() - sde_ap_start}")

            # Memory management.
            # gc.collect()
            # torch.cuda.empty_cache()

        dist = Categorical(action_probabilities)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # If we're testing, just return the deterministic action. Sampling should only be for training
        # as our "exploration" factor.
        # if self.deterministic:
        #     return mean.detach().numpy(), 1
        if self.deterministic:
            return torch.max(action_probabilities).detach().numpy(), 1

        # Return the sampled action and the log probability of that action in our distribution
        # action = action.cpu().detach().numpy()
        action_probabilities = action_probabilities.cpu().detach().numpy()
        log_prob = log_prob.detach()

        return (
            action_probabilities,
            action.item(),
            log_prob,
        )

    def evaluate(self, batch_obs, batch_acts, batch_action_probabilities):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        :param batch_action_probabilities: TBD
        :param batch_obs - the observations from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of observation)
        :param batch_acts - the actions from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of action)
        # :param batch_rtgs - the rewards-to-go calculated in the most recently collected
        #                     batch as a tensor. Shape: (number of timesteps in batch)
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        # if batch_obs.size(0) == 1:
        #     V = self.critic(batch_obs)
        # else:
        V = self.critic(batch_obs).squeeze()

        # # Calculate the log probabilities of batch actions using most recent actor network.
        # # This segment of code is similar to that in get_action()
        # mean = self.actor(batch_obs)
        # dist = MultivariateNormal(mean, self.cov_mat)
        # log_probs = dist.log_prob(batch_acts)

        # TODO: FINALIZE TESTING AND USE THE BEST ONE.

        # TEST: WHAT IF WE USE THE SAME ACTION PROBABILITIES GIVEN BY SDE?
        if self.use_state_dependent_exploration:
            action_probabilities = self.actor(batch_obs)
            # action_probabilities.data = batch_action_probabilities
        else:
            action_probabilities = self.actor(batch_obs)
        dist = Categorical(action_probabilities)
        log_probs = dist.log_prob(batch_acts)
        # print(
        #     f"ORIGINAL Log Probs: "
        #     f"{len(log_probs), log_probs[0].requires_grad, log_probs.requires_grad, log_probs[0], log_probs}"
        # )
        # from torchviz import make_dot

        # make_dot(log_probs[0]).render("TEST LOG PROB 0", format="png")
        # make_dot(log_probs).render("TEST LOG PROB", format="png")
        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, dist.entropy()

    def _log_summary(self):
        """
        Print to stdout what we've logged so far in the most recent batch.

        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger["timesteps_counter"]
        i_so_far = self.logger["iterations_counter"]
        lr = self.logger["lr"]
        avg_ep_lens = np.mean(self.logger["batch_lengths"])
        avg_ep_rews = np.mean(
            [np.sum(ep_rews) for ep_rews in self.logger["batch_rewards"]]
        )
        # start_time = time.time_ns()
        avg_actor_loss = np.mean(
            [losses.float().mean().cpu() for losses in self.logger["actor_losses"]]
        )
        # print("CPU", avg_actor_loss, time.time_ns() - start_time)
        # start_time = time.time_ns()
        # avg_actor_loss = torch.mean(
        #     torch.tensor(
        #         [losses.float().mean() for losses in self.logger["actor_losses"]]
        #     )
        # )
        # print("GPU:", avg_actor_loss, time.time_ns() - start_time)

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(
            f"-------------------- Iteration #{i_so_far} --------------------",
            flush=True,
        )
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger["batch_lengths"] = []
        self.logger["batch_rewards"] = []
        self.logger["actor_losses"] = []

    def evaluate_performance(self):
        """This method evaluates our agent's performance."""

        episode_rewards = []
        episode_timesteps = []

        for i in range(self.number_of_evaluation_episodes):
            raw_observation_list = []
            state, _ = self.environment.reset()
            done = False
            episode_reward = 0
            episode_timestep = 0
            if self.force_image_observation:
                # observation = self.environment.render()[:, :, 0]
                state = resize(
                    (self.environment.render()[150:350, 200:400, 0] / 255 - 0.5) * 2,
                    (self.observation_shape[0], self.observation_shape[1]),
                    anti_aliasing=False,
                )

            # TODO: Atari:
            state = resize(
                (state[33:193, :, 0] / 255 - 0.5) * 2,
                (self.observation_shape[0], self.observation_shape[1]),
                anti_aliasing=False,
            )

            while not done:
                if self.stack_observations:
                    raw_observation_list.append(state)
                    state = self.observation_stacker(raw_observation_list)

                if self.stack_observations:
                    if self.force_image_observation:
                        state = np.asarray(state).reshape(
                            (
                                -1,
                                1,
                                self.number_of_observations_to_stack,
                                self.observation_shape[0],
                                self.observation_shape[1],
                            )
                        )
                    else:
                        # state = np.asarray(state).reshape(
                        #     (
                        #         -1,
                        #         self.number_of_observations_to_stack,
                        #         self.observation_shape,
                        #     )
                        # )

                        # TODO: Atari:
                        state = np.asarray(state).reshape(
                            (
                                -1,
                                1,
                                self.number_of_observations_to_stack,
                                self.observation_shape[0],
                                self.observation_shape[1],
                            )
                        )

                state = torch.tensor(state, dtype=torch.float, device=self.device)

                with torch.inference_mode():
                    action_probabilities = self.actor(state)
                action_probabilities = action_probabilities.cpu().detach().numpy()
                action = np.argmax(action_probabilities)

                state, reward, terminated, truncated, info = self.environment.step(
                    action
                )

                if self.force_image_observation:
                    # observation = self.environment.render()[:, :, 0]
                    state = resize(
                        (self.environment.render()[150:350, 200:400, 0] / 255 - 0.5)
                        * 2,
                        (self.observation_shape[0], self.observation_shape[1]),
                        anti_aliasing=False,
                    )

                # TODO: Atari:
                state = resize(
                    (state[33:193, :, 0] / 255 - 0.5) * 2,
                    (self.observation_shape[0], self.observation_shape[1]),
                    anti_aliasing=False,
                )

                episode_reward += reward
                episode_timestep += 1

                done = terminated or truncated
            episode_rewards.append(episode_reward)
            episode_timesteps.append(episode_timestep)

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

        min_timesteps = min(episode_timesteps)
        max_timesteps = max(episode_timesteps)
        mean_timesteps = np.mean(episode_timesteps)
        std_timesteps = np.std(episode_timesteps)
        median_timesteps = np.median(episode_timesteps)
        mode_timesteps, mode_timesteps_count = stats.mode(episode_timesteps)
        (
            one_percentile_timesteps,
            five_percentile_timesteps,
            ten_percentile_timesteps,
            twenty_five_percentile_timesteps,
            fifty_percentile_timesteps,
            seventy_five_percentile_timesteps,
            ninty_percentile_timesteps,
            ninty_five_percentile_timesteps,
            ninty_nine_percentile_timesteps,
        ) = np.percentile(a=episode_timesteps, q=[1, 5, 10, 25, 50, 75, 90, 95, 99])

        self.evaluation_episode_rewards.append(
            [
                self.global_timestep_counter,
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
        )
        self.evaluation_episode_timesteps.append(
            [
                self.global_timestep_counter,
                min_timesteps,
                max_timesteps,
                mean_timesteps,
                std_timesteps,
                median_timesteps,
                mode_timesteps,
                mode_timesteps_count,
                one_percentile_timesteps,
                five_percentile_timesteps,
                ten_percentile_timesteps,
                twenty_five_percentile_timesteps,
                fifty_percentile_timesteps,
                seventy_five_percentile_timesteps,
                ninty_percentile_timesteps,
                ninty_five_percentile_timesteps,
                ninty_nine_percentile_timesteps,
            ]
        )
