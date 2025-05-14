import os
import argparse
import json
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

def analyze_folder(args_tuple):
    folder, root, ds_root = args_tuple
    info_path = os.path.join(root, folder, "info.json")
    if not os.path.exists(info_path):
        return (folder, "Skipping because info.json does not exist")

    try:
        with open(info_path) as f:
            info = json.load(f)
    except Exception as e:
        return (folder, f"[Error] Failed to load info.json: {e}")

    seed = info.get("seed", 0)
    if "seed" not in info:
        msg = f"[WARNING] does not have a seed, defaulting to 0"
    else:
        msg = None

    if "dataset" not in info:
        return (folder, "[Error] does not have a dataset, skipping")

    dataset = info["dataset"]
    ds_folder = os.path.join(ds_root, dataset)
    if not os.path.exists(ds_folder):
        return (folder, "[Error] does not have a dataset folder, skipping")

    # Remove .metrics files
    metrics_files = glob.glob(os.path.join(root, folder, "*.metrics"))
    for mfile in metrics_files:
        try:
            os.remove(mfile)
        except Exception as e:
            return (folder, f"[Error] Failed to remove {mfile}: {e}")

    try:
        import analyze
        analyze.analyze(
            os.path.join(root, folder),
            ds_folder,
            seed,
        )
    except Exception as e:
        return (folder, f"[Error] failed to analyze: {e}")

    return (folder, "analyzed successfully" if msg is None else f"{msg}, analyzed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reanalyze the data")
    parser.add_argument("--root", type=str, default=".simulations")
    parser.add_argument("--ds_root", type=str, default=".splits")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    folders_in_root = [
        f for f in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, f))
    ]

    tasks = [(folder, args.root, args.ds_root) for folder in folders_in_root]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_folder = {executor.submit(analyze_folder, t): t[0] for t in tasks}
        for idx, future in enumerate(as_completed(future_to_folder)):
            folder = future_to_folder[future]
            try:
                folder, result = future.result()
                print(f"[{idx+1}/{len(tasks)}] {folder}: {result}")
            except Exception as exc:
                print(f"[{idx+1}/{len(tasks)}] {folder}: generated an exception: {exc}")
