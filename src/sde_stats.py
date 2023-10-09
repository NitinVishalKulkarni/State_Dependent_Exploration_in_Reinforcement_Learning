import pandas as pd
import numpy as np
from collections import defaultdict

# sde = pd.read_csv("state_dependent_exploration_large.csv")
sde = pd.read_csv("state_dependent_exploration_lunar_lander.csv")
sde_stats = pd.DataFrame()
default_exploration_stats = pd.DataFrame()


mins = defaultdict(list)
maxs = defaultdict(list)
means = defaultdict(list)
stds = defaultdict(list)
medians = defaultdict(list)


for j in range(len(sde)):
    for i in range(10):
        mins[f"{(j + 1) * 5000}"].append(sde[f"Run {i + 1} Min Reward"].iloc[j])
        maxs[f"{(j + 1) * 5000}"].append(sde[f"Run {i + 1} Max Reward"].iloc[j])
        means[f"{(j + 1) * 5000}"].append(sde[f"Run {i + 1} Mean Reward"].iloc[j])
        stds[f"{(j + 1) * 5000}"].append(sde[f"Run {i + 1} Std Reward"].iloc[j])
        medians[f"{(j + 1) * 5000}"].append(sde[f"Run {i + 1} Median Reward"].iloc[j])

print(mins)
sde_stats["Timesteps"] = [(j + 1) * 5000 for j in range(len(mins))]
sde_stats["Min Min Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Max Min Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Mean Min Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Std Min Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Median Min Reward"] = [np.NAN for _ in range(len(mins))]

sde_stats["Min Max Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Max Max Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Mean Max Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Std Max Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Median Max Reward"] = [np.NAN for _ in range(len(mins))]

sde_stats["Min Mean Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Max Mean Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Mean Mean Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Std Mean Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Median Mean Reward"] = [np.NAN for _ in range(len(mins))]

sde_stats["Min Std Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Max Std Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Mean Std Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Std Std Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Median Std Reward"] = [np.NAN for _ in range(len(mins))]

sde_stats["Min Median Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Max Median Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Mean Median Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Std Median Reward"] = [np.NAN for _ in range(len(mins))]
sde_stats["Median Median Reward"] = [np.NAN for _ in range(len(mins))]

for i in range(len(mins)):
    sde_stats["Min Min Reward"].iloc[i] = np.min(mins[f"{(i + 1) * 5000}"])
    sde_stats["Max Min Reward"].iloc[i] = np.max(mins[f"{(i + 1) * 5000}"])
    sde_stats["Mean Min Reward"].iloc[i] = np.mean(mins[f"{(i + 1) * 5000}"])
    sde_stats["Std Min Reward"].iloc[i] = np.std(mins[f"{(i + 1) * 5000}"])
    sde_stats["Median Min Reward"].iloc[i] = np.median(mins[f"{(i + 1) * 5000}"])

    sde_stats["Min Max Reward"].iloc[i] = np.min(maxs[f"{(i + 1) * 5000}"])
    sde_stats["Max Max Reward"].iloc[i] = np.max(maxs[f"{(i + 1) * 5000}"])
    sde_stats["Mean Max Reward"].iloc[i] = np.mean(maxs[f"{(i + 1) * 5000}"])
    sde_stats["Std Max Reward"].iloc[i] = np.std(maxs[f"{(i + 1) * 5000}"])
    sde_stats["Median Max Reward"].iloc[i] = np.median(maxs[f"{(i + 1) * 5000}"])

    sde_stats["Min Mean Reward"].iloc[i] = np.min(means[f"{(i + 1) * 5000}"])
    sde_stats["Max Mean Reward"].iloc[i] = np.max(means[f"{(i + 1) * 5000}"])
    sde_stats["Mean Mean Reward"].iloc[i] = np.mean(means[f"{(i + 1) * 5000}"])
    sde_stats["Std Mean Reward"].iloc[i] = np.std(means[f"{(i + 1) * 5000}"])
    sde_stats["Median Mean Reward"].iloc[i] = np.median(means[f"{(i + 1) * 5000}"])

    sde_stats["Min Std Reward"].iloc[i] = np.min(stds[f"{(i + 1) * 5000}"])
    sde_stats["Max Std Reward"].iloc[i] = np.max(stds[f"{(i + 1) * 5000}"])
    sde_stats["Mean Std Reward"].iloc[i] = np.mean(stds[f"{(i + 1) * 5000}"])
    sde_stats["Std Std Reward"].iloc[i] = np.std(stds[f"{(i + 1) * 5000}"])
    sde_stats["Median Std Reward"].iloc[i] = np.median(stds[f"{(i + 1) * 5000}"])

    sde_stats["Min Median Reward"].iloc[i] = np.min(medians[f"{(i + 1) * 5000}"])
    sde_stats["Max Median Reward"].iloc[i] = np.max(medians[f"{(i + 1) * 5000}"])
    sde_stats["Mean Median Reward"].iloc[i] = np.mean(medians[f"{(i + 1) * 5000}"])
    sde_stats["Std Median Reward"].iloc[i] = np.std(medians[f"{(i + 1) * 5000}"])
    sde_stats["Median Median Reward"].iloc[i] = np.median(medians[f"{(i + 1) * 5000}"])

print(sde_stats.head(100))
sde_stats.to_csv(
    "lunar_lander_state_dependent_exploration_condensed_results.csv", index=False
)
