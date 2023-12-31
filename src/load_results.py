import numpy as np

import pandas as pd
from src.settings import results_directory

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 1000)

configuration_name = "SDE"
environment_id = "BoxingNoFrameskip-v4"
current_time = "12-29-2023 12-24-47"
log_path = f"{results_directory}/{environment_id}/{configuration_name}/{current_time}"
number_of_training_runs = 1
per_training_run_evaluation_statistics = [
    np.load(f"{log_path}/training_run_{i + 1}/raw_evaluation_results.npz")
    for i in range(number_of_training_runs)
]

timesteps = per_training_run_evaluation_statistics[0]["timesteps"]
rewards = [
    mean_reward[2]
    for mean_reward in per_training_run_evaluation_statistics[0]["results"]
]
episode_lengths = [
    mean_episode_length[2]
    for mean_episode_length in per_training_run_evaluation_statistics[0]["ep_lengths"]
]

df = pd.DataFrame(
    {"Timesteps": timesteps, "Rewards": rewards, "Episode Lengths": episode_lengths}
)
print(df.head(100))
# print("Timesteps:", per_training_run_evaluation_statistics[0]["timesteps"])
# print("Rewards:", [mean_reward[2] for mean_reward in per_training_run_evaluation_statistics[0]["results"]])
# print("Episode Lengths:", [mean_episode_length[2] for mean_episode_length in per_training_run_evaluation_statistics[0]["ep_lengths"]])
