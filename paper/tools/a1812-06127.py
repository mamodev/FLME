import os
import shutil
from timeline.timeline_gen import fedprox_timeline, save_exp, set_save_folder

timeline_folder = ".experiments/a1812-06127/timelines"

if os.path.exists(timeline_folder):
    shutil.rmtree(timeline_folder)

set_save_folder(timeline_folder)

# save_exp(fedprox_timeline(part=30, drp=0,  lr=0.01, seed=42), "part30_drp0.0_lr0.01_seed42.json")
# save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=42), "part30_drp0.5_lr0.01_seed42.json")
# save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=42), "part30_drp0.9_lr0.01_seed42.json")

# save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=42, allow_partial_part=True), "part30_drp0.5_lr0.03_mu0_seed42.json")
# save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=42, allow_partial_part=True), "part30_drp0.9_lr0.03_mu0_seed42.json")

# save_exp(fedprox_timeline(part=30, drp=.0, lr=0.01, seed=42, allow_partial_part=True, mu=1), "part30_drp0.0_lr0.03_mu1_seed42.json")
# save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=42, allow_partial_part=True, mu=1), "part30_drp0.5_lr0.03_mu1_seed42.json")
# save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=42, allow_partial_part=True, mu=1), "part30_drp0.9_lr0.03_mu1_seed42.json")


# # MNIST
# save_exp(fedprox_timeline(part=1000, drp=.0, lr=0.03, seed=42), "part1000_drp0.0_lr0.03_seed42.json")
# save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=42), "part1000_drp0.5_lr0.03_seed42.json")
# save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=42), "part1000_drp0.9_lr0.03_seed42.json")

# save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=42, allow_partial_part=True), "part1000_drp0.5_lr0.03_mu0_seed42.json")
# save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=42, allow_partial_part=True), "part1000_drp0.9_lr0.03_mu0_seed42.json")

# save_exp(fedprox_timeline(part=1000, drp=.0, lr=0.03, seed=42, allow_partial_part=True, mu=1), "part1000_drp0.0_lr0.03_mu1_seed42.json")
# save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=42, allow_partial_part=True, mu=1), "part1000_drp0.5_lr0.03_mu1_seed42.json")
# save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=42, allow_partial_part=True, mu=1), "part1000_drp0.9_lr0.03_mu1_seed42.json")

# # Added experiments 
# # save_exp(fedprox_timeline(part=30, cpa=30, drp=0,  lr=0.01, seed=42), "part30_DISTR_lr0.01_seed42.json")
# # save_exp(fedprox_timeline(part=1000, cpa=1000, drp=0, lr=0.03, seed=42), "part1000_DISTR_lr0.03_seed42.json")


# save_exp(fedprox_timeline(part=30, drp=0,  lr=0.01, seed=33), "part30_drp0.0_lr0.01_seed33.json")
# save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=33), "part30_drp0.5_lr0.01_seed33.json")
# save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=33), "part30_drp0.9_lr0.01_seed33.json")

# save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=33, allow_partial_part=True), "part30_drp0.5_lr0.03_mu0_seed33.json")
# save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=33, allow_partial_part=True), "part30_drp0.9_lr0.03_mu0_seed33.json")

# save_exp(fedprox_timeline(part=30, drp=.0, lr=0.01, seed=33, allow_partial_part=True, mu=1), "part30_drp0.0_lr0.03_mu1_seed33.json")
# save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=33, allow_partial_part=True, mu=1), "part30_drp0.5_lr0.03_mu1_seed33.json")
# save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=33, allow_partial_part=True, mu=1), "part30_drp0.9_lr0.03_mu1_seed33.json")


# # MNIST
# save_exp(fedprox_timeline(part=1000, drp=.0, lr=0.03, seed=33), "part1000_drp0.0_lr0.03_seed33.json")
# save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=33), "part1000_drp0.5_lr0.03_seed33.json")
# save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=33), "part1000_drp0.9_lr0.03_seed33.json")

# save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=33, allow_partial_part=True), "part1000_drp0.5_lr0.03_mu0_seed33.json")
# save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=33, allow_partial_part=True), "part1000_drp0.9_lr0.03_mu0_seed33.json")

# save_exp(fedprox_timeline(part=1000, drp=.0, lr=0.03, seed=33, allow_partial_part=True, mu=1), "part1000_drp0.0_lr0.03_mu1_seed33.json")
# save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=33, allow_partial_part=True, mu=1), "part1000_drp0.5_lr0.03_mu1_seed33.json")
# save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=33, allow_partial_part=True, mu=1), "part1000_drp0.9_lr0.03_mu1_seed33.json")



synth_timelines = [
    # "part30_drp0.0_lr0.01_seed42.json",
    # "part30_drp0.5_lr0.01_seed42.json",
    # "part30_drp0.9_lr0.01_seed42.json",
    # # "part30_DISTR_lr0.01_seed42.json",
    # "part30_drp0.5_lr0.03_mu0_seed42.json",
    # "part30_drp0.9_lr0.03_mu0_seed42.json",
    # "part30_drp0.0_lr0.03_mu1_seed42.json",
    # "part30_drp0.5_lr0.03_mu1_seed42.json",
    # "part30_drp0.9_lr0.03_mu1_seed42.json",
    
    # "part30_drp0.0_lr0.01_seed33.json",
    # "part30_drp0.5_lr0.01_seed33.json",
    # "part30_drp0.9_lr0.01_seed33.json",
    # "part30_drp0.5_lr0.03_mu0_seed33.json",
    # "part30_drp0.9_lr0.03_mu0_seed33.json",
    # "part30_drp0.0_lr0.03_mu1_seed33.json",
    # "part30_drp0.5_lr0.03_mu1_seed33.json",
    # "part30_drp0.9_lr0.03_mu1_seed33.json",
    
    save_exp(fedprox_timeline(part=30, drp=0,  lr=0.01, seed=42), "part30_drp0.0_lr0.01_seed42.json"),
    save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=42), "part30_drp0.5_lr0.01_seed42.json"),
    save_exp(fedprox_timeline(part=30, cpa=30, epochs=1, batch_size=512, drp=0,  lr=0.01, seed=42), "part30_all_lr0.01_seed42.json"),

    # save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=42), "part30_drp0.9_lr0.01_seed42.json"),
    # save_exp(fedprox_timeline(part=30, drp=.0, lr=0.01, seed=42, allow_partial_part=True, mu=1), "part30_drp0.0_lr0.03_mu1_seed42.json"),
    save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=42, allow_partial_part=True, mu=1), "part30_drp0.5_lr0.03_mu1_seed42.json"),
    # save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=42, allow_partial_part=True, mu=1), "part30_drp0.9_lr0.03_mu1_seed42.json"),
    ]

sims = {
    'a1812-06127_synth_iid':    synth_timelines,
    'a1812-06127_synth_11':     synth_timelines,
    'a1812-06127_synth_0505':   synth_timelines,
    'a1812-06127_synth_00':     synth_timelines,

    'a1812-06127_synth_iid_nosk':    synth_timelines,
    'a1812-06127_synth_11_nosk':     synth_timelines,
    'a1812-06127_synth_0505_nosk':   synth_timelines,
    'a1812-06127_synth_00_nosk':     synth_timelines,

    'a1812-06127_synth_iid_nqsk':    synth_timelines,
    'a1812-06127_synth_11_nqsk':     synth_timelines,
    'a1812-06127_synth_0505_nqsk':   synth_timelines,
    'a1812-06127_synth_00_nqsk':     synth_timelines,

    
    'a1812-06127_mnist': [
        save_exp(fedprox_timeline(part=1000, drp=.0, lr=0.03, seed=42), "part1000_drp0.0_lr0.03_seed42.json"),
        save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=42), "part1000_drp0.5_lr0.03_seed42.json"),
        save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=42), "part1000_drp0.9_lr0.03_seed42.json"),

        save_exp(fedprox_timeline(part=1000, drp=.0, lr=0.03, seed=42, allow_partial_part=True, mu=1), "part1000_drp0.0_lr0.03_mu1_seed42.json"),
        save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=42, allow_partial_part=True, mu=1), "part1000_drp0.5_lr0.03_mu1_seed42.json"),
        save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=42, allow_partial_part=True, mu=1), "part1000_drp0.9_lr0.03_mu1_seed42.json"),

        save_exp(fedprox_timeline(part=1000, drp=.0, lr=0.03, seed=42, cpa=300), "part1000_drp0.0_lr0.03_cpa300_seed42.json"),
        save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=42, cpa=300), "part1000_drp0.5_lr0.03_cpa300_seed42.json"),
        save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=42, cpa=300), "part1000_drp0.9_lr0.03_cpa300_seed42.json"),

        save_exp(fedprox_timeline(part=1000, drp=.0, lr=0.03, seed=42, cpa=300, allow_partial_part=True, mu=1), "part1000_drp0.0_lr0.03_cpa300_mu1_seed42.json"),
        save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=42, cpa=300, allow_partial_part=True, mu=1), "part1000_drp0.5_lr0.03_cpa300_mu1_seed42.json"),
        save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=42, cpa=300, allow_partial_part=True, mu=1), "part1000_drp0.9_lr0.03_cpa300_mu1_seed42.json"),

    #     # "part1000_DISTR_lr0.03_seed42.json",
        
    #     "part1000_drp0.0_lr0.03_seed42.json",
    #     "part1000_drp0.5_lr0.03_seed42.json",
    #     "part1000_drp0.9_lr0.03_seed42.json",
    #     "part1000_drp0.5_lr0.03_mu0_seed42.json",
    #     "part1000_drp0.9_lr0.03_mu0_seed42.json",
    #     "part1000_drp0.0_lr0.03_mu1_seed42.json",
    #     "part1000_drp0.5_lr0.03_mu1_seed42.json",
    #     "part1000_drp0.9_lr0.03_mu1_seed42.json",
        
          
    #     "part1000_drp0.0_lr0.03_seed33.json",
    #     "part1000_drp0.5_lr0.03_seed33.json",
    #     "part1000_drp0.9_lr0.03_seed33.json",
    #     "part1000_drp0.5_lr0.03_mu0_seed33.json",
    #     "part1000_drp0.9_lr0.03_mu0_seed33.json",
    #     "part1000_drp0.0_lr0.03_mu1_seed33.json",
    #     "part1000_drp0.5_lr0.03_mu1_seed33.json",
    #     "part1000_drp0.9_lr0.03_mu1_seed33.json",
    ]
}

# create makefile
Makefile = "ROOT := ../..\n"
Makefile += "DS_FOLDER := data\n"
Makefile += "WPD_FLAG := --worker-per-device 1"
Makefile += """
ifneq ($(WORKER_PER_DEVICE),)
    WPD_FLAG := --worker-per-device $(WORKER_PER_DEVICE)
endif
"""

AllRule = "all: "
AllPlots = "all-plots: "
AllRuns = "all-runs: "

for dataset, sim_files in sims.items():
    
    sim_name = dataset.replace('a1812-06127_', '')
    Makefile += f"\n######### {sim_name} #########\n\n"

    ds_path = f"$(ROOT)/$(DS_FOLDER)/{dataset}"

    all_deps = []
    sim_deps = []
    plot_deps = []

    for file in sim_files:
        fmt_file = file.replace('.json', '')

        params = fmt_file.split('_')
        # split params by first number
        param_dict = {}
        for param in params:
            if any(char.isdigit() for char in param):
                split_idx = next(i for i, char in enumerate(param) if char.isdigit())
                key = param[:split_idx]
                value = param[split_idx:]
                param_dict[key] = value
            else:
                param_dict[param] = None

        params_str = ', '.join([f"{k}={v}" for k, v in param_dict.items()])


        for n in range(1):
            unique_id =   f"{sim_name}-{fmt_file}-{n + 1}"
            repo_folder = f"runs/{sim_name}/{fmt_file}/{n + 1}"
            plot_folder = f"plots/{sim_name}/{fmt_file}/{n + 1}"

            Makefile += f"{repo_folder}: \n"
            Makefile += f"\tpython3 $(ROOT)/tools/simulation/timeline.py"
            Makefile += f" --net logistic"
            Makefile += f" --repo-folder {repo_folder}"
            Makefile += f" --ds-folder {ds_path}"
            Makefile += f" --timeline timelines/{file}"
            Makefile += f" $(WPD_FLAG)\n"

            Makefile += "\n"

            Makefile += f"{repo_folder}/info.json: \n"
            Makefile += f"\tpython3 $(ROOT)/tools/simulation/analyze.py"
            Makefile += f" --net logistic"
            Makefile += f" --sim-dir {repo_folder}"
            Makefile += f" --dataset-dir {ds_path}"
            Makefile += "\n\n"

            Makefile += f"{plot_folder}: $(ROOT)/tools/analyses/sim_plots.py\n"
            Makefile += f"\tpython3 $(ROOT)/tools/analyses/sim_plots.py"
            Makefile += f" {repo_folder}"
            Makefile += f" {plot_folder}"
            Makefile += f" --title '{sim_name}  ({params_str})'"
            Makefile += "\n\n"

            Makefile += f"{unique_id}-run: "
            Makefile += f" {repo_folder}"
            Makefile += f" {repo_folder}/info.json"
            Makefile += "\n\n"

            sim_deps.append(f"{unique_id}-run")
            plot_deps.append(f"{plot_folder}")

            Makefile += f"{unique_id}-all:"
            Makefile += f" {repo_folder}"
            Makefile += f" {repo_folder}/info.json"
            Makefile += f" {plot_folder}"
            Makefile += "\n\n"

            all_deps.append(f"{unique_id}-all")

        
    
    Makefile += "\n"

    Makefile += f"{sim_name}: "
    for dep in all_deps:
        Makefile += f"{dep} "
    Makefile += "\n\n"

    Makefile += f"{sim_name}-runs: "
    for dep in sim_deps:
        Makefile += f"{dep} "
    Makefile += "\n\n"

    Makefile += f"{sim_name}-plots: "
    for dep in plot_deps:
        Makefile += f"{dep} "
    Makefile += "\n\n"

    AllPlots += f"{sim_name}-plots "
    AllRuns += f"{sim_name}-runs "
    AllRule += f"{sim_name} "

Makefile += AllPlots + "\n\n"
Makefile += AllRuns + "\n\n"
Makefile += AllRule + "\n\n"
Makefile += "\n\n"
    
with open(".experiments/a1812-06127/Makefile", "w") as f:
    f.write(Makefile)