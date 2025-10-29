import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import seaborn as sns
import numpy as np
import argparse 
from matplotlib.colors import LinearSegmentedColormap

parser = argparse.ArgumentParser(description="Analyze simulation results")
parser.add_argument("sim_folder", type=str, help="Path to the simulation folder")
parser.add_argument("out_folder", type=str, help="Path to save the output plots")
parser.add_argument("--title", type=str, default="Simulation Results", help="Title of the plot")
args = parser.parse_args()

os.makedirs(args.out_folder, exist_ok=True)

folder = args.sim_folder

metrics = []

info_file = f"{folder}/info.json"
with open(info_file, "r") as f:
    info = json.load(f)

for filename in os.listdir(folder):
    if filename.endswith(".metrics"):
        with open(os.path.join(folder, filename), "r") as f:
            metrics.append(json.load(f))



del f
del filename
del folder


def extract_contrib(contributors, pid):
    # fiter all text and keep numbers
    pid = pid.strip().replace("ds-", "")
    pid = int(pid)

    for contrib in contributors:
        if contrib["client"][0] == pid:
            return contrib
    return {}

flat_metrics = [
    {
        "version": m["version"],
        "accuracy": m["accuracy"],
        "partitionID": pid,
        "partitionAccuracy": acc,
        "trainSamples": extract_contrib(m["contributors"], pid).get("train_samples", 0),
        # calculate variance over in info["partition_nsamples_per_class"][pid]
        "partitionClassVariance": np.var(info["partition_nsamples_per_class"][pid]),
    }
    for m in metrics
    for pid, acc in m["groups"].items()
]

# partitions = [int(pid.strip().replace("ds-", "")) for pid in info["partition_nsamples"].keys()]
partitions = list(info["partition_nsamples"].keys())
n_partitions = len(partitions)


df = pd.DataFrame(flat_metrics)
df["partitionID"] = df["partitionID"].astype("category")
df = df.sort_values("version")

df["partitionVariance"] = df["partitionID"].map(
    df.groupby("partitionID", observed=False)["partitionAccuracy"].var()
)
df["partitionVariance"] = pd.to_numeric(df["partitionVariance"], errors="coerce")

df['samples'] = df['partitionID'].map(info['partition_nsamples'])
df['totalTrainSamples'] = (
    df.groupby('partitionID', observed=False)['trainSamples']
      .transform('sum')
)
# get top 5 less variance partitions
top_low_variance = (
    df.drop_duplicates(subset="partitionID")
      .nsmallest(5, "partitionVariance")
)

# Get top 5 highest variance partitions (unique partitionID)
top_high_variance = (
    df.drop_duplicates(subset="partitionID")
      .nlargest(5, "partitionVariance")
)

top_high_contribs = (
    df.drop_duplicates(subset="partitionID")
      .nlargest(5, "totalTrainSamples")
)

top_low_contribs = (
    df.drop_duplicates(subset="partitionID")
      .nsmallest(5, "totalTrainSamples")
)

# plot only highest variance partition accuracy
thv_df = df[df["partitionID"].isin(top_high_variance["partitionID"])]
thv_df = thv_df[['version', 'partitionAccuracy']]
thv_df = thv_df.rename(columns={"partitionAccuracy": "accuracy"})
thv_df = thv_df.groupby("version").agg({
    "accuracy": "mean"
}).reset_index()

tlv_df = df[df["partitionID"].isin(top_low_variance["partitionID"])]
tlv_df = tlv_df[['version', 'partitionAccuracy']]
tlv_df = tlv_df.rename(columns={"partitionAccuracy": "accuracy"})
tlv_df = tlv_df.groupby("version").agg({
    "accuracy": "mean"
}).reset_index()

<<<<<<< HEAD
=======
def weighted_std(group):
    weights = group['totalTrainSamples']
    values = group['partitionAccuracy']
    weighted_mean = np.average(values, weights=weights)
    weighted_var = np.average((values - weighted_mean) ** 2, weights=weights)
    return np.sqrt(weighted_var)  # Std dev is sqrt of variance
>>>>>>> 25b7ed3811d98b386a209d894ad37beaaa567912

agg_df = df.copy()
agg_df = agg_df[["version", "partitionAccuracy", 'totalTrainSamples']]
agg_df = agg_df.rename(columns={"partitionAccuracy": "accuracy"})
agg_df['waccuracy'] = agg_df['accuracy'] * agg_df['totalTrainSamples']
agg_df['accuracy_std'] = agg_df['accuracy']
<<<<<<< HEAD
=======
agg_df['wstd'] = agg_df['accuracy']

>>>>>>> 25b7ed3811d98b386a209d894ad37beaaa567912
agg_df = agg_df.groupby("version").agg({
    "accuracy": "mean",
    "accuracy_std": 'std',
    "waccuracy": "sum",
    "totalTrainSamples": "sum"
}).reset_index()
agg_df["waccuracy"] = agg_df["waccuracy"] / agg_df["totalTrainSamples"]

<<<<<<< HEAD
=======
wstd_series = df.groupby('version').apply(weighted_std)
agg_df['wstd'] = wstd_series.values

>>>>>>> 25b7ed3811d98b386a209d894ad37beaaa567912
max_idx = agg_df["accuracy"].idxmax()
max_version = agg_df.loc[max_idx, "version"]
max_acc = agg_df.loc[max_idx, "accuracy"]

<<<<<<< HEAD
# PLOTS
def plot(**opts):
    plt.figure(figsize=(15, 6))
=======
wmax_idx = agg_df["waccuracy"].idxmax()
wmax_version = agg_df.loc[wmax_idx, "version"]
wmax_acc = agg_df.loc[wmax_idx, "waccuracy"]

# PLOTS
def plot(**opts):
    plt.figure(figsize=(4*2, 3*2))
>>>>>>> 25b7ed3811d98b386a209d894ad37beaaa567912

    if opts.get('w_accuracy', False):
        plt.plot(agg_df["version"], agg_df["accuracy"])
    if opts.get('w_waccuracy', False):
        plt.plot(agg_df["version"], agg_df["waccuracy"], label="Weighted Accuracy", linestyle="--")
<<<<<<< HEAD
=======
        
>>>>>>> 25b7ed3811d98b386a209d894ad37beaaa567912
    if opts.get('w_thv', False):
        plt.plot(thv_df["version"], thv_df["accuracy"], label="Top 5 High Variance Partitions", linestyle=":")
    if opts.get('w_tlv', False):
        plt.plot(tlv_df["version"], tlv_df["accuracy"], label="Top 5 Low Variance Partitions", linestyle="--")
    if opts.get('w_variance', False):
        plt.fill_between(agg_df["version"], agg_df["accuracy"] - agg_df["accuracy_std"], agg_df["accuracy"] + agg_df["accuracy_std"], alpha=0.2)
<<<<<<< HEAD
    if opts.get('w_max', False):
        plt.axhline(y=max_acc, color="red", linestyle="--", label=f"Max Accuracy: {max_acc:.2f}")
        plt.scatter(max_version, max_acc, color="red", s=100, zorder=5)
        plt.text(
            agg_df["version"].iloc[-1],
            max_acc,
            f"{max_acc:.2f}",
            color="red",
            va="bottom",
            ha="right",
        )
=======
        # plt.fill_between(agg_df["version"], 
        #             agg_df["waccuracy"] - agg_df["wstd"], 
        #             agg_df["waccuracy"] + agg_df["wstd"], 
        #             alpha=0.2, color="blue", label="± Weighted Std")
            
    if opts.get('w_max', False):
        plt.axhline(y=max_acc, color="red", linestyle="--", label=f"Max Accuracy: {max_acc:.2f}")
        # plt.scatter(max_version, max_acc, color="red", s=100, zorder=5)
        # plt.text(
        #     agg_df["version"].iloc[-1],
        #     max_acc,
        #     f"{max_acc:.2f}",
        #     color="red",
        #     va="bottom",
        #     ha="right",
        # )

    if opts.get('w_wmax', False):
        plt.axhline(y=wmax_acc, color="green", linestyle="--", label=f"Max Weighted Accuracy: {wmax_acc:.2f}")
        # plt.scatter(wmax_version, wmax_acc, color="green", s=100, zorder=5)
        # plt.text(
        #     agg_df["version"].iloc[-1],
        #     wmax_acc,
        #     f"{wmax_acc:.2f} (w)",
        #     color="green",
        #     va="top",
        #     ha="right",
        # )
>>>>>>> 25b7ed3811d98b386a209d894ad37beaaa567912
        
    if opts.get('wacc_rand', 0):
        acc_bp = df.groupby(['version', 'partitionID'])['partitionAccuracy'].mean().reset_index()
        # print(acc_bp.head())

        # Create a color map for each partition
        red_to_grey = LinearSegmentedColormap.from_list(
            "red_grey", ["red", "grey"]
        )

        colors = red_to_grey(np.linspace(0, 1, len(partitions)))

        # sort partitions by nsamples
        sparts = list(partitions)
        partitions.sort(key=lambda x: info["partition_nsamples"][x])

        for pid, color in zip(sparts, colors):
            fds = acc_bp[acc_bp['partitionID'] == pid]
            
            if fds.empty:
                continue
                
            
            plt.plot(fds['version'], fds['partitionAccuracy'], label=f"Partition {pid}", color=color)
            
        

    if args.title:
        plt.suptitle(args.title)
        
    plt.xlabel("Version")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Version")
<<<<<<< HEAD
    
    plt.legend()

plot(w_variance=True, w_waccuracy=True, w_accuracy=True, w_thv=True, w_tlv=True, w_max=True)
plt.savefig(f"{args.out_folder}/accuracy_vs_version_full.png")

plot(w_accuracy=True, w_variance=True)
=======
    plt.ylim(0, 1)
    
    plt.legend()

plot(w_variance=True, w_waccuracy=True, w_accuracy=True, w_thv=True, w_tlv=True, w_max=True, w_wmax=True)
plt.savefig(f"{args.out_folder}/accuracy_vs_version_full.png")

plot(w_accuracy=True, w_variance=True, w_waccuracy=True, w_max=True, w_wmax=True)
>>>>>>> 25b7ed3811d98b386a209d894ad37beaaa567912
plt.savefig(f"{args.out_folder}/accuracy_vs_version.png")

plot(wacc_rand=20)
plt.savefig(f"{args.out_folder}/accuracy_vs_version_random.png")
