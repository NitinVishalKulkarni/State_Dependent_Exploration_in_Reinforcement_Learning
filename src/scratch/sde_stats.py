import pandas as pd
import numpy as np
from collections import defaultdict

# sde = pd.read_csv("state_dependent_exploration_large.csv")
sde = pd.read_csv("./Mountain Car/New folder/SDE_MountainCar_10_19_2023.csv")
sde_stats = pd.DataFrame()
default_exploration_stats = pd.DataFrame()
# sde.fillna(0, inplace=True)

timesteps = defaultdict(list)
mins = defaultdict(list)
maxs = defaultdict(list)
means = defaultdict(list)
stds = defaultdict(list)
medians = defaultdict(list)


for j in range(len(sde)):
    for i in range(7):
        timesteps[f"{(j + 1) * 1}"].append(sde[f"Run {i + 1} Timesteps"].iloc[j])
        mins[f"{(j + 1) * 1}"].append(sde[f"Run {i + 1} Min Reward"].iloc[j])
        maxs[f"{(j + 1) * 1}"].append(sde[f"Run {i + 1} Max Reward"].iloc[j])
        means[f"{(j + 1) * 1}"].append(sde[f"Run {i + 1} Mean Reward"].iloc[j])
        stds[f"{(j + 1) * 1}"].append(sde[f"Run {i + 1} Std Reward"].iloc[j])
        medians[f"{(j + 1) * 1}"].append(sde[f"Run {i + 1} Median Reward"].iloc[j])

sde_stats["Min Timesteps"] = [np.NAN for _ in range(len(timesteps))]
sde_stats["Max Timesteps"] = [np.NAN for _ in range(len(timesteps))]
sde_stats["Mean Timesteps"] = [np.NAN for _ in range(len(timesteps))]
sde_stats["Std Timesteps"] = [np.NAN for _ in range(len(timesteps))]
sde_stats["Median Timesteps"] = [np.NAN for _ in range(len(timesteps))]

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
    sde_stats["Min Timesteps"].iloc[i] = np.nanmin(timesteps[f"{(i + 1) * 1}"])
    sde_stats["Max Timesteps"].iloc[i] = np.nanmax(timesteps[f"{(i + 1) * 1}"])
    sde_stats["Mean Timesteps"].iloc[i] = np.nanmean(timesteps[f"{(i + 1) * 1}"])
    sde_stats["Std Timesteps"].iloc[i] = np.nanstd(timesteps[f"{(i + 1) * 1}"])
    sde_stats["Median Timesteps"].iloc[i] = np.nanmedian(timesteps[f"{(i + 1) * 1}"])

    sde_stats["Min Min Reward"].iloc[i] = np.nanmin(mins[f"{(i + 1) * 1}"])
    sde_stats["Max Min Reward"].iloc[i] = np.nanmax(mins[f"{(i + 1) * 1}"])
    sde_stats["Mean Min Reward"].iloc[i] = np.nanmean(mins[f"{(i + 1) * 1}"])
    sde_stats["Std Min Reward"].iloc[i] = np.nanstd(mins[f"{(i + 1) * 1}"])
    sde_stats["Median Min Reward"].iloc[i] = np.nanmedian(mins[f"{(i + 1) * 1}"])

    sde_stats["Min Max Reward"].iloc[i] = np.nanmin(maxs[f"{(i + 1) * 1}"])
    sde_stats["Max Max Reward"].iloc[i] = np.nanmax(maxs[f"{(i + 1) * 1}"])
    sde_stats["Mean Max Reward"].iloc[i] = np.nanmean(maxs[f"{(i + 1) * 1}"])
    sde_stats["Std Max Reward"].iloc[i] = np.nanstd(maxs[f"{(i + 1) * 1}"])
    sde_stats["Median Max Reward"].iloc[i] = np.nanmedian(maxs[f"{(i + 1) * 1}"])

    sde_stats["Min Mean Reward"].iloc[i] = np.nanmin(means[f"{(i + 1) * 1}"])
    sde_stats["Max Mean Reward"].iloc[i] = np.nanmax(means[f"{(i + 1) * 1}"])
    sde_stats["Mean Mean Reward"].iloc[i] = np.nanmean(means[f"{(i + 1) * 1}"])
    sde_stats["Std Mean Reward"].iloc[i] = np.nanstd(means[f"{(i + 1) * 1}"])
    sde_stats["Median Mean Reward"].iloc[i] = np.nanmedian(means[f"{(i + 1) * 1}"])

    sde_stats["Min Std Reward"].iloc[i] = np.nanmin(stds[f"{(i + 1) * 1}"])
    sde_stats["Max Std Reward"].iloc[i] = np.nanmax(stds[f"{(i + 1) * 1}"])
    sde_stats["Mean Std Reward"].iloc[i] = np.nanmean(stds[f"{(i + 1) * 1}"])
    sde_stats["Std Std Reward"].iloc[i] = np.nanstd(stds[f"{(i + 1) * 1}"])
    sde_stats["Median Std Reward"].iloc[i] = np.nanmedian(stds[f"{(i + 1) * 1}"])

    sde_stats["Min Median Reward"].iloc[i] = np.nanmin(medians[f"{(i + 1) * 1}"])
    sde_stats["Max Median Reward"].iloc[i] = np.nanmax(medians[f"{(i + 1) * 1}"])
    sde_stats["Mean Median Reward"].iloc[i] = np.nanmean(medians[f"{(i + 1) * 1}"])
    sde_stats["Std Median Reward"].iloc[i] = np.nanstd(medians[f"{(i + 1) * 1}"])
    sde_stats["Median Median Reward"].iloc[i] = np.nanmedian(medians[f"{(i + 1) * 1}"])

print(sde_stats.head(60))
sde_stats.to_csv("./Mountain Car/New Folder/SDE_MC.csv", index=False)
