import copy
import sys

from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import MlpExtractor, NatureCNN, BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space

from gymnasium import spaces
import torch as th
from torch import nn
from typing import Callable, Tuple, Union, List, Dict, Type
from custom_callbacks import StateDependentExploration
import numpy as np
import json
from torchinfo import summary
import gymnasium as gym


class CustomMlpExtractor(MlpExtractor):
    """
    This class inherits the MlpExtractor from Stable-baselines3 and adds the functionality of getting the output of
    the last layer from the Actor Network to be used as features for clustering when working with complex observations.
    """

    def __init__(self, feature_dim: int, net_arch: Union[List[int], Dict[str, List[int]]],
                 activation_fn: Type[nn.Module], device: Union[th.device, str] = "auto", use_sde_actor: bool = False,
                 use_sde_critic: bool = False):
        """

        :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
        :param net_arch: The specification of the policy and value networks.
        :param activation_fn: The activation function to use for the networks.
        :param device: PyTorch device.
        :param use_sde_actor: Boolean indicating if we want to use the SDE Actor as the feature extractor for
            clustering.
        :param use_sde_critic: Boolean indicating if we want to use the SDE Actor as the feature extractor for
            clustering.
        """
        super().__init__(feature_dim, net_arch, activation_fn, device)

        if use_sde_actor:
            self.sde_actor = copy.deepcopy(self.policy_net)
            self.sde_actor.to(device=device)
            self.sde_actor.load_state_dict(self.policy_net.state_dict())

        if use_sde_critic:
            self.sde_critic = copy.deepcopy(self.value_net)
            self.sde_critic.to(device=device)
            self.sde_critic.load_state_dict(self.value_net.state_dict())

    def forward_sde_actor(self, features: th.Tensor) -> th.Tensor:
        return self.sde_actor(features)

    def forward_sde_critic(self, features: th.Tensor) -> th.Tensor:
        return self.sde_critic(features)


class CustomNatureCNN(NatureCNN):
    """
    This class inherits the NatureCNN from Stable-baselines3 and adds the functionality of getting the output of
    the last layer from the Actor Network to be used as features for clustering when working with complex observations.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 32, normalized_image: bool = False,
                 use_sde_actor: bool = False, use_sde_critic: bool = False) -> None:
        super().__init__(observation_space, features_dim, normalized_image)

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU()
        )

        if use_sde_actor:
            self.sde_actor = copy.deepcopy(self.linear(self.cnn))
            self.sde_actor.to(device="cuda")
            self.sde_actor.load_state_dict(self.self.linear(self.cnn).state_dict())

        if use_sde_critic:
            self.sde_critic = copy.deepcopy(self.self.linear(self.cnn))
            self.sde_critic.to(device="cuda")
            self.sde_critic.load_state_dict(self.self.linear(self.cnn).state_dict())

    def forward_sde_actor(self, features: th.Tensor) -> th.Tensor:
        return self.sde_actor(features)

    def forward_sde_critic(self, features: th.Tensor) -> th.Tensor:
        return self.sde_critic(features)


# class CustomNatureCNN(BaseFeaturesExtractor):
#     """
#     CNN from DQN Nature paper:
#         Mnih, Volodymyr, et al.
#         "Human-level control through deep reinforcement learning."
#         Nature 518.7540 (2015): 529-533.
#
#     :param observation_space:
#     :param features_dim: Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     :param normalized_image: Whether to assume that the image is already normalized
#         or not (this disables dtype and bounds checks): when True, it only checks that
#         the space is a Box and has 3 dimensions.
#         Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
#     """
#
#     def __init__(
#             self,
#             observation_space: gym.Space,
#             features_dim: int = 8,
#             normalized_image: bool = False,
#     ) -> None:
#         assert isinstance(observation_space, spaces.Box), (
#             "NatureCNN must be used with a gym.spaces.Box ",
#             f"observation space, not {observation_space}",
#         )
#         super().__init__(observation_space, features_dim)
#         # We assume CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper
#         assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
#             "You should use NatureCNN "
#             f"only with images not with {observation_space}\n"
#             "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
#             "If you are using a custom environment,\n"
#             "please check it using our env checker:\n"
#             "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
#             "If you are using `VecNormalize` or already normalized channel-first images "
#             "you should pass `normalize_images=False`: \n"
#             "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
#         )
#         n_input_channels = observation_space.shape[0]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#
#         # Compute shape by doing one forward pass
#         with th.no_grad():
#             n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
#
#         self.linear = nn.Sequential(
#             nn.Linear(n_flatten, 512),
#             nn.ReLU(),
#             nn.Linear(512, features_dim),
#             nn.ReLU()
#         )
#
#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.linear(self.cnn(observations))


class CustomActorCriticPolicy(ActorCriticCnnPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            features_extractor_class=CustomNatureCNN,
            *args,
            **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        results_generator_configuration = open('results_generator_configuration.json')
        results_generator_configuration = json.load(results_generator_configuration)

        self.use_state_dependent_exploration = results_generator_configuration["use_state_dependent_exploration"]

        if self.use_state_dependent_exploration:
            state_dependent_exploration_configuration = open('state_dependent_exploration_configuration.json')
            state_dependent_exploration_configuration = json.load(state_dependent_exploration_configuration)

            if (state_dependent_exploration_configuration["feature_extractor"] == "SDE_Actor" or
                    state_dependent_exploration_configuration["feature_extractor"] == "SDE_Critic"):
                state_dependent_exploration_configuration["feature_extractor"] = self.extract_features

            state_dependent_exploration_configuration["action_space"] = self.action_space

            self.state_dependent_exploration = StateDependentExploration(
                state_dependent_exploration_configuration=state_dependent_exploration_configuration)
            # self.state_dependent_exploration = StateDependentExploration(
            #     action_space=self.action_space, verbose=0, features_extractor=self.extract_features, device=self.device
            # )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).

        # if state_dependent_exploration_configuration["feature_extractor"] == "SDE_Actor":
        self.mlp_extractor = CustomMlpExtractor(
            self.features_dim,
            # net_arch=dict(pi=[64, 64, 512], vf=[64, 64]),
            net_arch=[],
            # net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def forward(
            self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        # print(f"Observation Shape: {obs.size()}")
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        print(f"Latent Pi: {type(latent_pi), latent_pi.size(),}")
        print(f"Features: {type(features), features.size()}")
        # self.mlp_extractor.forward_sde_actor(features)
        # sde_features = self.features_extractor_class.forward_sde_actor(features)
        # print(f"SDE Features: {type(sde_features), sde_features.size()}")

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        print(f"Latent VF: {type(latent_vf), latent_vf.size(),}")
        print(f"Values: {type(values), values.size(),}")

        policy_net_modules = th.nn.ModuleList(self.mlp_extractor.policy_net.modules())
        print(f"Policy Net Modules: {type(policy_net_modules), len(policy_net_modules), policy_net_modules}")
        policy_net_modules = th.nn.ModuleList(self.mlp_extractor.value_net.modules())
        print(f"Value Net Modules: {type(policy_net_modules), len(policy_net_modules), policy_net_modules}")

        print(f"\n\nSummary Policy: {summary(self.mlp_extractor.policy_net, (4, 84, 84))}")
        print(f"\n\nSummary Value: {summary(self.mlp_extractor.value_net, (4, 84, 84))}")
        print(self.mlp_extractor.policy_net)
        for name, param in self.mlp_extractor.policy_net.parameters():
            print(name, param)
        sys.exit()

        # print(f"\nOG Action Probabilities: {distribution.distribution.probs.squeeze(), type(distribution)}")
        if (self.use_state_dependent_exploration and 100 < self.state_dependent_exploration.num_timesteps <
                self.state_dependent_exploration.reweighing_duration *
                self.state_dependent_exploration.locals[
                    "total_timesteps"]):
            # print(self.state_dependent_exploration.num_timesteps)
            action_probabilities_ = (
                self.state_dependent_exploration.reweigh_action_probabilities(
                    distribution.distribution.probs.squeeze(),
                    share_features_extractor=self.share_features_extractor, neural_network=self.mlp_extractor
                )
            )
            # try:
            actions = [np.random.choice(self.action_space.n,
                                        p=action_probabilities) for action_probabilities in action_probabilities_]

            # except ValueError:
            #     print(
            #         f"\nOG Action Probabilities: {distribution.distribution.probs.squeeze(), sum(distribution.distribution.probs.squeeze())}")
            #     print(f"Updated Action Probabilities: {action_probabilities, sum(action_probabilities)}")
            #     print("Cluster Associated Actions:", self.state_dependent_exploration.cluster_associated_actions[
            #         self.state_dependent_exploration.clusterer.labels_[-1]])
            #     sys.exit()
            actions = th.tensor(
                actions, device=self.state_dependent_exploration.model.device
            )
            # print(f"SDE Actions: {actions, actions.size(), actions.requires_grad}")
            # actions = distribution.get_actions(deterministic=deterministic)
            # print(f"OG Actions: {actions, actions.size(), actions.requires_grad}")
        else:
            actions = distribution.get_actions(deterministic=deterministic)
        # print(f"Original Action: {actions}")
        # print(f"Updated Action Probabilities: {action_probabilities, updated_distribution.probs}")
        # print(f"Updated Action: {action}")
        try:
            log_prob = distribution.log_prob(actions)
        except ValueError:
            print(actions.size(), distribution.distribution.probs.size(),
                  distribution.distribution.probs.squeeze().size(), len(action_probabilities_))
            log_prob = distribution.log_prob(actions)

        # log_prob = updated_distribution.log_prob(actions)
        # print(f"Original Log Probability: {distribution.log_prob(action)}")
        # print(f"Updated Log Probability: {updated_distribution.log_prob(action)}")
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob
