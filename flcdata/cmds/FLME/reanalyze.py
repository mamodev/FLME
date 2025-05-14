



if __name__ == "__main__":
    import os
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Reanalyze the data")
    parser.add_argument("--root", type=str, default=".simulations")

    args = parser.parse_args()

    folders_in_root = [
        f for f in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, f))
    ]

    for fidx, folder in enumerate(folders_in_root):
        info_path = os.path.join(args.root, folder, "info.json")
        if not os.path.exists(info_path):
            print(f"Skipping {folder} because info.json does not exist")
            continue

        info = json.load(open(info_path))
        seed = info.get("seed", None)
        if seed is None:
            print(f"[WARNING] {folder} does not have a seed, defaulting to 0")
            seed = 0

        
        if "dataset" not in info:
            print(f"[Error] {folder} does not have a dataset, skipping")
            continue

        dataset = info["dataset"]
        ds_folder = os.path.join(".splits", dataset)
        if not os.path.exists(ds_folder):
            print(f"[Error] {folder} does not have a dataset folder, skipping")
            continue

        try:
            os.system(f"rm {os.path.join(args.root, folder)}/*.metrics")

            import analyze  
            analyze.analyze(
                os.path.join(args.root, folder),
                ds_folder,
                seed,
            )
        except Exception as e:
            print(f"[Error] {folder} failed to analyze: {e}")
            continue
        
        print(f"[{fidx+1}/{len(folders_in_root)}] {folder} analyzed successfully")