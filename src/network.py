import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math


class MLPPolicy(nn.Module):
    """
    A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """

    def __init__(self, in_dim, out_dim, model, action_space_type=None):
        """
        Initialize the network and set up the layers.

        Parameters:
            in_dim - input dimensions as an int
            out_dim - output dimensions as an int

        Return:
            None
        """
        super(MLPPolicy, self).__init__()
        # print("Input Dimensions:", in_dim)

        self.action_space_type = action_space_type

        self.layer1 = nn.Linear(in_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 8)
        self.layer4 = nn.Linear(8, out_dim)

        self.model = model

    def forward(self, x):
        """
        Runs a forward pass on the neural network.

        Parameters:
            obs - observation to pass as input

        Return:
            output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))

        # actor_output = F.softmax(self.layer3(activation2), dim=-1)
        # critic_output = self.layer3(activation2)

        # print(
        #     f"\n\nModel: {self.model}\nActor Output: {actor_output}\nActor Output Shape: {actor_output.shape} "
        #     f"\nCritic Output: {critic_output}\nCritic Output Shape: {critic_output.shape}"
        # )

        if self.model == "actor":
            return F.softmax(self.layer4(x), dim=-1)
        elif self.model == "sde_actor":
            return x
        else:
            return self.layer4(x)


class CNNPolicy(nn.Module):
    """
    A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """

    def __init__(self, in_dim, out_dim, model, action_space_type=None):
        """
        Initialize the network and set up the layers.

        Parameters:
            in_dim - input dimensions as an int
            out_dim - output dimensions as an int

        Return:
            None
        """
        super(CNNPolicy, self).__init__()
        # print("Input Dimensions:", in_dim)

        self.action_space_type = action_space_type
        # TODO: Try Conv2D
        # THESE PARAMETERS WORK WITH 80, 120 IMAGES.
        self.conv1 = nn.Conv3d(
            in_channels=1, out_channels=32, stride=(1, 2, 2), kernel_size=(4, 10, 10)
        )
        self.conv2 = nn.Conv3d(
            in_channels=32, out_channels=64, stride=(1, 2, 2), kernel_size=(1, 10, 10)
        )
        self.conv3 = nn.Conv3d(
            in_channels=64, out_channels=16, stride=(1, 2, 2), kernel_size=(1, 4, 4)
        )
        self.fc1 = nn.Linear(16 * 1 * 6 * 6, 128)
        # self.conv1 = nn.Conv3d(
        #     in_channels=1, out_channels=32, stride=(1, 1, 1), kernel_size=(4, 9, 9)
        # )
        # self.conv2 = nn.Conv3d(
        #     in_channels=32, out_channels=64, stride=(1, 2, 2), kernel_size=(1, 10, 10)
        # )
        # self.conv3 = nn.Conv3d(
        #     in_channels=64, out_channels=16, stride=(1, 2, 2), kernel_size=(1, 2, 2)
        # )
        # self.pool = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        # self.fc1 = nn.Linear(16 * 1 * 6 * 16, 64)
        # self.fc2 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, out_dim)

        self.model = model

    def forward(self, x):
        """
        Runs a forward pass on the neural network.

        Parameters:
            obs - observation to pass as input

        Return:
            output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # actor_output = F.softmax(self.fc2(x), dim=-1).squeeze()
        # critic_output = self.fc2(x).squeeze()

        # print(
        #     f"\n\nModel: {self.model}\nActor Output: {actor_output}\nActor Output Shape: {actor_output.shape} "
        #     f"\nCritic Output: {critic_output}\nCritic Output Shape: {critic_output.shape}"
        # )
        #
        if self.model == "actor":
            return F.softmax(self.fc4(x), dim=-1).squeeze()
        elif self.model == "sde_actor":
            return x
        else:
            return self.fc4(x).squeeze()


class LSTMPolicy(nn.Module):
    def __init__(self, in_dim, out_dim, model, hidden_size=64, num_layers=1):
        super(LSTMPolicy, self).__init__()
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_dim)
        self.model = model

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            device=x.device
        )
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            device=x.device
        )

        out, _ = self.lstm(x, (h0, c0))

        # out = self.fc(out[:, -1, :])
        actor_output = F.softmax(self.fc(out[:, -1, :]), dim=-1).squeeze()
        critic_output = self.fc(out[:, -1, :]).squeeze(1)

        # print(
        #     f"\n\nModel: {self.model}\nActor Output: {actor_output}\nActor Output Shape: {actor_output.shape} "
        #     f"\nCritic Output: {critic_output}\nCritic Output Shape: {critic_output.shape}"
        # )

        if self.model == "actor":
            return actor_output
        else:
            return critic_output
        # return out


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        # Note: Vocab size should be the maximum sequence length. i.e., number of observations to stack.
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # print("Positional Encoding Class Called")
        pe = torch.zeros(vocab_size, d_model)
        # print(f"PE Initialized: {pe}")
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        # print(f"Position: {position}")
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # # print(f"Div Term: {div_term}")
        pe[:, 0::2] = torch.sin(position * div_term)
        # # print(f"PE Sin {pe}")
        pe[:, 1::2] = torch.cos(position * div_term)
        # print(f"PE Cos: {pe}")
        pe = pe.unsqueeze(0) / 100
        # print(f"PE Unsqueezed: {pe}")
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print(f"Input to the Positional Encoding: {x}")
        # print(f"X size 1:", x.size(1))
        # print(f"X Size: {x.size()}")
        x = x + self.pe[:, : x.size(1), :]
        # print("Positional Encoding Output:", x[0], x.shape)
        return self.dropout(x)


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        model,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        dropout=0.0,
        embedding_dimension=64,
    ):
        super().__init__()

        self.model = model

        # # This comes from the paper where the dimensionality of the Query, Key, and Value vector is dk=dv=d_model/h
        # assert in_dim % nhead == 0

        self.dense_layer = nn.Linear(in_dim, embedding_dimension)

        # self.embedding_layer = nn.Embedding(
        #     num_embeddings=in_dim, embedding_dim=embedding_dimension
        # )
        self.self_attention_pooling = SelfAttentionPooling(embedding_dimension)

        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dimension,
            dropout=dropout,
            vocab_size=8,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        # self.classifier = nn.Linear(64, 2)
        self.d_model = embedding_dimension
        self.output_layer = nn.Linear(embedding_dimension, out_dim)

    def forward(self, observation):
        # Convert observation to tensor if it's a numpy array
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float)

        observation = self.dense_layer(observation)

        # This is from the paper: "In the embedding layers, we multiply those weights by square root of d_model."
        # print(f"\n\nOriginal Observation:, {observation, observation.shape}")
        observation = observation * math.sqrt(self.d_model)

        # print(
        #     f"Observation after multiplying by the square root of d_model: {observation}"
        # )

        observation = self.pos_encoder(observation)
        # print(
        # f"Observation after positional encoding: {observation, observation.shape}"
        # )
        # print("Huh")
        # sys.exit()
        # print("\n\n\n\n\nSystem Exit")
        observation = self.transformer_encoder(observation)
        # print("Output Encoder:", observation.shape)
        # TODO: Instead of taking the average along the sequence length dimension, we can use a self attention
        #  pooling layer which might produce better results.
        # observation = observation.mean(dim=1)
        observation = self.self_attention_pooling(observation)

        actor_output = F.softmax(self.output_layer(observation), dim=-1).squeeze()
        # actor_output = F.softmax(self.output_layer(observation), dim=1).squeeze()
        critic_output = self.output_layer(observation).squeeze(1)

        # print(
        #     f"\n\nModel: {self.model}\nActor Output: {actor_output}\nActor Output Shape: {actor_output.shape} "
        #     f"\nCritic Output: {critic_output}\nCritic Output Shape: {critic_output.shape}"
        # )

        if self.model == "actor":
            return actor_output
        else:
            return critic_output
