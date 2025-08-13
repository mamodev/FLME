from timeline.timeline_gen import fedprox_timeline, save_exp, set_save_folder

timeline_folder = ".experiments/a1812-06127/timelines"
set_save_folder(timeline_folder)

save_exp(fedprox_timeline(part=30, drp=0,  lr=0.01, seed=42), "part30_drp0.0_lr0.01_seed42.json")
save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=42), "part30_drp0.5_lr0.01_seed42.json")
save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=42), "part30_drp0.9_lr0.01_seed42.json")

save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=42, allow_partial_part=True), "part30_drp0.5_lr0.03_mu0_seed42.json")
save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=42, allow_partial_part=True), "part30_drp0.9_lr0.03_mu0_seed42.json")

save_exp(fedprox_timeline(part=30, drp=.0, lr=0.01, seed=42, allow_partial_part=True, mu=1), "part30_drp0.0_lr0.03_mu1_seed42.json")
save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=42, allow_partial_part=True, mu=1), "part30_drp0.5_lr0.03_mu1_seed42.json")
save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=42, allow_partial_part=True, mu=1), "part30_drp0.9_lr0.03_mu1_seed42.json")


# MNIST
save_exp(fedprox_timeline(part=1000, drp=.0, lr=0.03, seed=42), "part1000_drp0.0_lr0.03_seed42.json")
save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=42), "part1000_drp0.5_lr0.03_seed42.json")
save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=42), "part1000_drp0.9_lr0.03_seed42.json")

save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=42, allow_partial_part=True), "part1000_drp0.5_lr0.03_mu0_seed42.json")
save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=42, allow_partial_part=True), "part1000_drp0.9_lr0.03_mu0_seed42.json")

save_exp(fedprox_timeline(part=1000, drp=.0, lr=0.03, seed=42, allow_partial_part=True, mu=1), "part1000_drp0.0_lr0.03_mu1_seed42.json")
save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=42, allow_partial_part=True, mu=1), "part1000_drp0.5_lr0.03_mu1_seed42.json")
save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=42, allow_partial_part=True, mu=1), "part1000_drp0.9_lr0.03_mu1_seed42.json")

# Added experiments 
# save_exp(fedprox_timeline(part=30, cpa=30, drp=0,  lr=0.01, seed=42), "part30_DISTR_lr0.01_seed42.json")
# save_exp(fedprox_timeline(part=1000, cpa=1000, drp=0, lr=0.03, seed=42), "part1000_DISTR_lr0.03_seed42.json")


save_exp(fedprox_timeline(part=30, drp=0,  lr=0.01, seed=33), "part30_drp0.0_lr0.01_seed33.json")
save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=33), "part30_drp0.5_lr0.01_seed33.json")
save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=33), "part30_drp0.9_lr0.01_seed33.json")

save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=33, allow_partial_part=True), "part30_drp0.5_lr0.03_mu0_seed33.json")
save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=33, allow_partial_part=True), "part30_drp0.9_lr0.03_mu0_seed33.json")

save_exp(fedprox_timeline(part=30, drp=.0, lr=0.01, seed=33, allow_partial_part=True, mu=1), "part30_drp0.0_lr0.03_mu1_seed33.json")
save_exp(fedprox_timeline(part=30, drp=.5, lr=0.01, seed=33, allow_partial_part=True, mu=1), "part30_drp0.5_lr0.03_mu1_seed33.json")
save_exp(fedprox_timeline(part=30, drp=.9, lr=0.01, seed=33, allow_partial_part=True, mu=1), "part30_drp0.9_lr0.03_mu1_seed33.json")


# MNIST
save_exp(fedprox_timeline(part=1000, drp=.0, lr=0.03, seed=33), "part1000_drp0.0_lr0.03_seed33.json")
save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=33), "part1000_drp0.5_lr0.03_seed33.json")
save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=33), "part1000_drp0.9_lr0.03_seed33.json")

save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=33, allow_partial_part=True), "part1000_drp0.5_lr0.03_mu0_seed33.json")
save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=33, allow_partial_part=True), "part1000_drp0.9_lr0.03_mu0_seed33.json")

save_exp(fedprox_timeline(part=1000, drp=.0, lr=0.03, seed=33, allow_partial_part=True, mu=1), "part1000_drp0.0_lr0.03_mu1_seed33.json")
save_exp(fedprox_timeline(part=1000, drp=.5, lr=0.03, seed=33, allow_partial_part=True, mu=1), "part1000_drp0.5_lr0.03_mu1_seed33.json")
save_exp(fedprox_timeline(part=1000, drp=.9, lr=0.03, seed=33, allow_partial_part=True, mu=1), "part1000_drp0.9_lr0.03_mu1_seed33.json")



synth_timelines = [
    "part30_drp0.0_lr0.01_seed42.json",
    "part30_drp0.5_lr0.01_seed42.json",
    "part30_drp0.9_lr0.01_seed42.json",
    # "part30_DISTR_lr0.01_seed42.json",
    "part30_drp0.5_lr0.03_mu0_seed42.json",
    "part30_drp0.9_lr0.03_mu0_seed42.json",
    "part30_drp0.0_lr0.03_mu1_seed42.json",
    "part30_drp0.5_lr0.03_mu1_seed42.json",
    "part30_drp0.9_lr0.03_mu1_seed42.json",
    
    "part30_drp0.0_lr0.01_seed33.json",
    "part30_drp0.5_lr0.01_seed33.json",
    "part30_drp0.9_lr0.01_seed33.json",
    "part30_drp0.5_lr0.03_mu0_seed33.json",
    "part30_drp0.9_lr0.03_mu0_seed33.json",
    "part30_drp0.0_lr0.03_mu1_seed33.json",
    "part30_drp0.5_lr0.03_mu1_seed33.json",
    "part30_drp0.9_lr0.03_mu1_seed33.json",
]

sims = {
    'a1812-06127_synth_iid':    synth_timelines,
    'a1812-06127_synth_00':     synth_timelines,
    'a1812-06127_synth_11':     synth_timelines,
    'a1812-06127_synth_0505':   synth_timelines,
    
    'a1812-06127_mnist': [
        # "part1000_DISTR_lr0.03_seed42.json",
        
        "part1000_drp0.0_lr0.03_seed42.json",
        "part1000_drp0.5_lr0.03_seed42.json",
        "part1000_drp0.9_lr0.03_seed42.json",
        "part1000_drp0.5_lr0.03_mu0_seed42.json",
        "part1000_drp0.9_lr0.03_mu0_seed42.json",
        "part1000_drp0.0_lr0.03_mu1_seed42.json",
        "part1000_drp0.5_lr0.03_mu1_seed42.json",
        "part1000_drp0.9_lr0.03_mu1_seed42.json",
        
          
        "part1000_drp0.0_lr0.03_seed33.json",
        "part1000_drp0.5_lr0.03_seed33.json",
        "part1000_drp0.9_lr0.03_seed33.json",
        "part1000_drp0.5_lr0.03_mu0_seed33.json",
        "part1000_drp0.9_lr0.03_mu0_seed33.json",
        "part1000_drp0.0_lr0.03_mu1_seed33.json",
        "part1000_drp0.5_lr0.03_mu1_seed33.json",
        "part1000_drp0.9_lr0.03_mu1_seed33.json",
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

for dataset, sim_files in sims.items():
    
    sim_name = dataset.replace('a1812-06127_', '')
    Makefile += f"\n######### {sim_name} #########\n\n"
        
    for file in sim_files:
        for n in range(3):
            prefix = '' if n == 0 else f"{n+1}-"
            fmt_file = file.replace('.json', '')
            Makefile += f"{prefix}{sim_name}-{fmt_file}: \n"
            Makefile += f"\tpython3 $(ROOT)/tools/simulation/timeline.py"
            Makefile += f" --net logistic"
            Makefile += f" --repo-folder {prefix}{sim_name}-{fmt_file}"
            Makefile += f" --ds-folder $(ROOT)/$(DS_FOLDER)/{dataset}"
            Makefile += f" --timeline timelines/{file}"
            Makefile += f" $(WPD_FLAG)\n"
            Makefile += "\n"

            Makefile += f"{prefix}{sim_name}-{fmt_file}/info.json: \n"
            Makefile += f"\tpython3 $(ROOT)/tools/simulation/analyze.py"
            Makefile += f" --net logistic"
            Makefile += f" --sim-dir {prefix}{sim_name}-{fmt_file}"
            Makefile += f" --dataset-dir $(ROOT)/$(DS_FOLDER)/{dataset}"
            Makefile += "\n\n"

            Makefile += f"plots/{prefix}{sim_name}-{fmt_file}: $(ROOT)/tools/analyses/sim_plots.py\n"
            Makefile += f"\tpython3 $(ROOT)/tools/analyses/sim_plots.py"
            Makefile += f" {prefix}{sim_name}-{fmt_file}"
            Makefile += f" plots/{prefix}{sim_name}-{fmt_file}"
            Makefile += "\n\n"

            Makefile += f"{prefix}{sim_name}-{fmt_file}-all:"
            Makefile += f" {prefix}{sim_name}-{fmt_file}"
            Makefile += f" {prefix}{sim_name}-{fmt_file}/info.json"
            Makefile += f" plots/{prefix}{sim_name}-{fmt_file}"
            Makefile += "\n\n"
        
    
    Makefile += "\n"    
    Makefile += f"{sim_name}: "
    for dep in sim_files:
        fmt_dep = dep.replace('.json', '')
        for n in range(3):
            prefix = '' if n == 0 else f"{n+1}-"
            Makefile += f"{prefix}{sim_name}-{fmt_dep}-all "
    Makefile += "\n\n"
    
    AllRule += f"{sim_name} "


Makefile += AllRule + "\n\n"

Makefile += "\n\n"

    
with open(".experiments/a1812-06127/Makefile", "w") as f:
    f.write(Makefile)