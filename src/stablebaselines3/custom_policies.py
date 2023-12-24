import copy
import sys

from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import (
    MlpExtractor,
    FlattenExtractor,
    NatureCNN,
)
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
    the last layer from the Actor/Critic Network to be used as features for clustering.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        use_sde_actor: bool = False,
        use_sde_critic: bool = False,
    ):
        """
        :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
        :param net_arch: The specification of the policy and value networks.
        :param activation_fn: The activation function to use for the networks.
        :param device: PyTorch device.
        :param use_sde_actor: Boolean indicating if we want to use the SDE Actor as the feature extractor for
            clustering.
        :param use_sde_critic: Boolean indicating if we want to use the SDE Critic as the feature extractor for
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

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 8,
        normalized_image: bool = False,
        use_sde_actor: bool = True,
        use_sde_critic: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim, normalized_image)

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

        if use_sde_actor:
            self.sde_actor_cnn = copy.deepcopy(self.cnn)
            self.sde_actor_linear = copy.deepcopy(self.linear)
            self.sde_actor_cnn.load_state_dict(self.cnn.state_dict())
            self.sde_actor_linear.load_state_dict(self.linear.state_dict())
            self.sde_actor = nn.Sequential(self.sde_actor_cnn, self.sde_actor_linear)

            self.sde_actor.to(device="cpu")
            # print("CNN Keys:", self.cnn.state_dict().keys())
            # print("Linear Keys:", self.linear.state_dict().keys())
            # print("SDE Actor Keys:", self.sde_actor.state_dict().keys())
            # temp = {self.cnn.state_dict() | self.linear.state_dict()}
            # print("Combined Keys:", temp.keys())
            # self.sde_actor.load_state_dict(
            #     state_dict=self.linear.state_dict() | self.linear.state_dict(),
            #     strict=False,
            # )
            # print(f"\n\nSummary SDE Actor: {summary(self.sde_actor, (10, 4, 84, 84))}")
            # sys.exit()

        if use_sde_critic:
            self.sde_critic = copy.deepcopy(self.self.linear(self.cnn))
            self.sde_critic.to(device="cuda")
            self.sde_critic.load_state_dict(self.self.linear(self.cnn).state_dict())

    def forward_sde_actor(self, features: th.Tensor) -> th.Tensor:
        return self.sde_actor(features)

    def forward_sde_critic(self, features: th.Tensor) -> th.Tensor:
        return self.sde_critic(features)


class CustomActorCriticPolicy(ActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        # features_extractor_class=CustomNatureCNN,
        *args,
        **kwargs,
    ):
        self.state_dependent_exploration_configuration = open(
            "state_dependent_exploration_configuration.json"
        )
        self.state_dependent_exploration_configuration = json.load(
            self.state_dependent_exploration_configuration
        )
        if is_image_space(observation_space) and self.state_dependent_exploration_configuration["feature_extractor"] == "Latent Policy":
            print("Using CustomNatureCNN feature extractor.")
            features_extractor_class = CustomNatureCNN
        elif is_image_space(observation_space) and not self.state_dependent_exploration_configuration["feature_extractor"] == "Latent Policy":
            print("Using the NatureCNN feature extractor.")
            features_extractor_class = NatureCNN
        else:
            features_extractor_class = FlattenExtractor
        # if self.verbose >= 2:
        # print(
        #     "Feature Extractor Class:",
        #     features_extractor_class,
        #     type(features_extractor_class),
        # )

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        results_generator_configuration = open("results_generator_configuration.json")
        results_generator_configuration = json.load(results_generator_configuration)

        self.use_state_dependent_exploration = results_generator_configuration[
            "use_state_dependent_exploration"
        ]

        if self.use_state_dependent_exploration:
            # self.state_dependent_exploration_configuration = open(
            #     "state_dependent_exploration_configuration.json"
            # )
            # self.state_dependent_exploration_configuration = json.load(
            #     self.state_dependent_exploration_configuration
            # )

            # TODO: Update this.
            # if (
            #     self.state_dependent_exploration_configuration["feature_extractor"]
            #     == "SDE_Actor"
            #     or self.state_dependent_exploration_configuration["feature_extractor"]
            #     == "SDE_Critic"
            # ):
            #     self.state_dependent_exploration_configuration[
            #         "feature_extractor"
            #     ] = self.extract_features

            self.state_dependent_exploration_configuration[
                "action_space"
            ] = self.action_space

            self.state_dependent_exploration = StateDependentExploration(
                state_dependent_exploration_configuration=self.state_dependent_exploration_configuration
            )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        print(
            "\nTesting if observation space is image space:",
            "\nObservation Space:",
            self.observation_space,
            "\nIs Image Space",
            is_image_space(self.observation_space),
        )
        print(
            f"Type self.features_dim: {type(self.features_dim)}, self.features_dim: {self.features_dim}"
        )

        if self.state_dependent_exploration_configuration[
            "feature_extractor"
        ] == "Latent Policy" and is_image_space(
            observation_space=self.observation_space
        ):
            print("CustomNatureCNN and Blank MLPExtractor.")
            self.mlp_extractor = CustomMlpExtractor(
                self.features_dim,
                # net_arch=self.net_arch,  # Default MLP Feature Extractor
                # net_arch=dict(pi=[64, 64, 512], vf=[64, 64]),  # If we want to use a custom MLP Feature Extractor
                net_arch=[],  # We use a blank network architecture if we use NatureCNN (or any other CNN).
                activation_fn=self.activation_fn,
                device=self.device,
            )
        elif self.state_dependent_exploration_configuration[
            "feature_extractor"
        ] == "Latent Policy" and not is_image_space(
            observation_space=self.observation_space
        ):
            print("Default MLPExtractor for non-image based observation space.")
            self.mlp_extractor = CustomMlpExtractor(
                self.features_dim,
                net_arch=self.net_arch,  # Default MLP Feature Extractor
                # net_arch=dict(pi=[64, 64, 512], vf=[64, 64]),  # If we want to use a custom MLP Feature Extractor
                # net_arch=[],  # We use a blank network architecture if we use NatureCNN (or any other CNN).
                activation_fn=self.activation_fn,
                device=self.device,
                use_sde_actor=True,
            )
        else:
            print("This is the default MLP Extractor for environments with vector observation space or image based "
                  "environments when latent policy is chosen as the feature extractor for SDE.")
            self.mlp_extractor = MlpExtractor(
                self.features_dim,
                net_arch=self.net_arch,
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
        # print(f"Custom Policies Observation Shape: {obs.size()}")
        # print("Custom Policies Obs:", obs)

        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # print(f"Latent Pi: {type(latent_pi), latent_pi.size(), latent_pi}")
        # print(f"Features: {type(features), features.size(), features}")
        # # self.mlp_extractor.forward_sde_actor(features)
        # sde_features = self.features_extractor.forward_sde_actor(obs / 255)
        # print(f"SDE Features: {type(sde_features), sde_features.size(), sde_features}")

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        # print(f"Latent VF: {type(latent_vf), latent_vf.size(),}")
        # print(f"Values: {type(values), values.size(),}")

        # policy_net_modules = th.nn.ModuleList(self.mlp_extractor.policy_net.modules())
        # print(
        #     f"Policy Net Modules: {type(policy_net_modules), len(policy_net_modules), policy_net_modules}"
        # )
        # policy_net_modules = th.nn.ModuleList(self.mlp_extractor.value_net.modules())
        # print(
        #     f"Value Net Modules: {type(policy_net_modules), len(policy_net_modules), policy_net_modules}"
        # )
        #
        # print(
        #     f"\n\nSummary Policy: {summary(self.mlp_extractor.policy_net, (4, 84, 84))}"
        # )
        # print(
        #     f"\n\nSummary Value: {summary(self.mlp_extractor.value_net, (4, 84, 84))}"
        # )
        # print(self.mlp_extractor.policy_net)
        # for name, param in self.mlp_extractor.policy_net.parameters():
        #     print(name, param)
        # sys.exit()
        # if deterministic:
        #     print(self.state_dependent_exploration.num_timesteps, deterministic)

        if (
            self.use_state_dependent_exploration
            and 100
            < self.state_dependent_exploration.num_timesteps
            < self.state_dependent_exploration.reweighing_duration
            * self.state_dependent_exploration.locals["total_timesteps"]
        ):
            # print(self.state_dependent_exploration.num_timesteps, deterministic)
            action_probabilities_ = (
                self.state_dependent_exploration.reweigh_action_probabilities(
                    distribution.distribution.probs.squeeze(),
                    share_features_extractor=self.share_features_extractor,
                    neural_network=self.features_extractor,
                )
            )
            # TODO: Uncomment the above block and comment the block below after testing CartPole-v1 with a single
            #  training environment.
            # action_probabilities_ = (
            #     self.state_dependent_exploration.reweigh_action_probabilities(
            #         distribution.distribution.probs,
            #         share_features_extractor=self.share_features_extractor,
            #         neural_network=self.features_extractor,
            #     )
            # )

            # TODO: Replicating my scratch implementation:
            action_probabilities_ = th.tensor(
                action_probabilities_, device=self.state_dependent_exploration.device
            )
            updated_distribution = th.distributions.Categorical(
                probs=action_probabilities_
            )
            actions = updated_distribution.sample()
            log_prob = updated_distribution.log_prob(actions)

            # print("SDE AP FROM Callback:", action_probabilities_)
            # print("SDE Updated Distribution:", updated_distribution.probs)

            # actions = [
            #     np.random.choice(self.action_space.n, p=action_probabilities)
            #     for action_probabilities in action_probabilities_
            # ]
            #
            # actions = th.tensor(
            #     actions, device=self.state_dependent_exploration.model.device
            # )

        else:
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
        # print(f"Original Action: {actions}")
        # print(f"Updated Action Probabilities: {action_probabilities_, updated_distribution_.probs}")
        # print(f"Updated Action: {action}")

        # log_prob = updated_distribution.log_prob(actions)

        # if (
        #     self.use_state_dependent_exploration
        #     and 100
        #     < self.state_dependent_exploration.num_timesteps
        #     < self.state_dependent_exploration.reweighing_duration
        #     * self.state_dependent_exploration.locals["total_timesteps"]
        # ):
        # print(
        #     f"\nOG Action Probabilities: {distribution.distribution.probs.squeeze(), type(distribution)}"
        # )
        # print("SDE Action Probabilities:", action_probabilities_)
        # actions_ = distribution.get_actions(deterministic=deterministic)
        # print(f"OG Actions: {actions_, actions_.size(), actions_.requires_grad}")
        # print(f"SDE Actions: {actions, actions.size(), actions.requires_grad}")
        # print(f"Original Log Probability: {distribution.log_prob(actions)}")
        # print(f"Updated Log Probability: {updated_distribution.log_prob(actions)}")
        # log_prob = updated_distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob
