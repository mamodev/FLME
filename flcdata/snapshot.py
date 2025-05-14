#!/usr/bin/env python3

import os
import argparse

parser = argparse.ArgumentParser(description="Create or restore a snapshot of the current state.")

parser.add_argument(
    "--restore",
    action="store_true",
    help="Restore the snapshot instead of creating one.",
)

args = parser.parse_args()


folders = [
    ".data",
    ".pipelines_logs",
    ".splits",
    ".timelines",
    ".simulations",
    ".assets",
    ".notes",
]

snampshot_name = input("Enter the name of the snapshot: ")
if not snampshot_name:
    print("Snapshot name cannot be empty.")
    exit(1)


snampshot_root = ".snapshots"
snampshot_path = os.path.join(snampshot_root, snampshot_name)

if not args.restore:
    if os.path.exists(snampshot_name):
        print(f"Snapshot {snampshot_name} already exists.")
        exit(1)

    os.makedirs(snampshot_path, exist_ok=True)

    for folder in folders:
        if os.path.exists(folder):
            res = os.system(f"mv {folder} {snampshot_path}")
            if res != 0:
                print(f"Failed to move {folder} to {snampshot_path}.")
        else:
            print(f"Folder {folder} does not exist. Skipping.")

    print(f"Snapshot {snampshot_name} created.")

else:
    if not os.path.exists(snampshot_path):
        print(f"Snapshot {snampshot_name} does not exist.")
        exit(1)

    for folder in folders:
        if os.path.exists(folder):
            print(f"Folder {folder} already exists. Skipping.")
        else:
            res = os.system(f"mv {snampshot_path}/{folder} .")
            if res != 0:
                print(f"Failed to move {snampshot_path}/{folder} to .")
            else:
                print(f"Restored {folder} from snapshot.")

    os.rmdir(snampshot_path)
    print(f"Snapshot {snampshot_name} restored.")

