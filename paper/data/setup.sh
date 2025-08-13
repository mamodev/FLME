#!/bin/bash 
set -e
# Usage <script.sh> <command> (default: install)
# where command is either install or clean 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

if [ "$1" != "clean" ] && [ "$1" != "install" ] && [ -n "$1" ]; then
    echo "Error: Invalid command '$1'."
    echo "Usage: $0 <command> (default: install)"
    echo "Where command is either 'install' or 'clean'."
    exit 1
fi

if [ "$1" == "clean" ]; then
    echo "Running clean command"
    rm -rf "$SCRIPT_DIR/.gh-litian96-fedprox"
    rm -rf "$SCRIPT_DIR/a1812-06127_synth00"
    rm -rf "$SCRIPT_DIR/a1812-06127_synth11"
    exit 0
fi

required_packages=("git" "python3")
for package in "${required_packages[@]}"; do
    if ! command -v $package &> /dev/null; then
        echo "$package could not be found, please install $package first."
        exit 1
    fi
done

required_python_libs=("numpy" "torch" "torchvision")
for lib in "${required_python_libs[@]}"; do
    if ! python3 -c "import $lib" &> /dev/null; then
        echo "Python library $lib could not be found, please install $lib first."
        exit 1
    fi
done

echo ".torch_datasets" > "$SCRIPT_DIR/.gitignore"

# check if there is folder .gh-litian96-fedprox
echo ".gh-litian96-fedprox" >> "$SCRIPT_DIR/.gitignore"
if [ -d "$SCRIPT_DIR/.gh-litian96-fedprox" ]; then
    echo "Folder .gh-litian96-fedprox exists"
else
    echo "Folder .gh-litian96-fedprox does not exist"
    git clone --depth 1 https://github.com/litian96/FedProx.git "$SCRIPT_DIR/.gh-litian96-fedprox"
fi

# fedprox paper
synth_ds=("synthetic_0_0" "synthetic_1_1" "synthetic_0.5_0.5" "synthetic_iid")
synth_ds_dest=("a1812-06127_synth_00" "a1812-06127_synth_11" "a1812-06127_synth_0505" "a1812-06127_synth_iid")

for i in "${!synth_ds[@]}"; do
    src="${synth_ds[$i]}"
    dest="${synth_ds_dest[$i]}"
    echo "$dest" >> "$SCRIPT_DIR/.gitignore"

    if [ -d "$SCRIPT_DIR/$dest" ]; then
        echo "Folder $dest exists, skipping dataset setup."
    else
        cp -r "$SCRIPT_DIR/.gh-litian96-fedprox/data/$src/data" "$SCRIPT_DIR/$dest"
        python3 $SCRIPT_DIR/a1812-06127-adapt.py "$SCRIPT_DIR/$dest/test/mytest.json" "$SCRIPT_DIR/$dest/"
        python3 $SCRIPT_DIR/a1812-06127-adapt.py "$SCRIPT_DIR/$dest/train/mytrain.json" "$SCRIPT_DIR/$dest/"
        rm -rf "$SCRIPT_DIR/$dest/test"
        rm -rf "$SCRIPT_DIR/$dest/train"
        mv "$SCRIPT_DIR/$dest/mytest.npz" "$SCRIPT_DIR/$dest/test.npz"
        mv "$SCRIPT_DIR/$dest/mytrain.npz" "$SCRIPT_DIR/$dest/train.npz"
        python3 $SCRIPT_DIR/merge.py "$SCRIPT_DIR/$dest/test.npz" "$SCRIPT_DIR/$dest/train.npz" "$SCRIPT_DIR/$dest/data.npz"
        echo "Dataset $dest has been set up."
    fi
done

echo "a1812-06127_mnist" >> "$SCRIPT_DIR/.gitignore"
if [ ! -d "$SCRIPT_DIR/a1812-06127_mnist" ]; then
    echo "Creating dataset a1812-06127_mnist"
    mkdir -p "$SCRIPT_DIR/a1812-06127_mnist"
    cd "$SCRIPT_DIR/a1812-06127_mnist"
    python3 "$SCRIPT_DIR/a1812-06127-mnist-patch.py"
    cd "$SCRIPT_DIR"
    echo "Dataset a1812-06127_mnist has been set up."

    # output  FOLDER/data/test/<some_unkown_name>.json => FOLDER/test.npz
    # output  FOLDER/data/train/<some_unkown_name>.json => FOLDER/train.npz
    test_json=$(find "$SCRIPT_DIR/a1812-06127_mnist/data/test" -name "*.json" | head -n 1)
    train_json=$(find "$SCRIPT_DIR/a1812-06127_mnist/data/train" -name "*.json" | head -n 1)
    if [ -z "$test_json" ] || [ -z "$train_json" ]; then
        echo "Unexpected error: JSON files not found as output of a1812-06127-mnist-patch.py"
        exit 1
    fi

    mv "$test_json" "$SCRIPT_DIR/a1812-06127_mnist/test.json"
    mv "$train_json" "$SCRIPT_DIR/a1812-06127_mnist/train.json"

    python3 $SCRIPT_DIR/a1812-06127-adapt.py "$SCRIPT_DIR/a1812-06127_mnist/test.json" "$SCRIPT_DIR/a1812-06127_mnist/"
    python3 $SCRIPT_DIR/a1812-06127-adapt.py "$SCRIPT_DIR/a1812-06127_mnist/train.json" "$SCRIPT_DIR/a1812-06127_mnist/"
    python3 "$SCRIPT_DIR/merge.py" "$SCRIPT_DIR/a1812-06127_mnist/test.npz" "$SCRIPT_DIR/a1812-06127_mnist/train.npz" "$SCRIPT_DIR/a1812-06127_mnist/data.npz"
    rm -rf "$SCRIPT_DIR/a1812-06127_mnist/data" "$SCRIPT_DIR/a1812-06127_mnist/train.json" "$SCRIPT_DIR/a1812-06127_mnist/test.json"

    echo "Dataset a1812-06127_mnist successfully processed."
fi