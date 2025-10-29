#!/bin/bash
set -e  

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

ENSURE_REMOVED=("$SCRIPT_DIR/a1812-06127_synth_iid_nosk" "$SCRIPT_DIR/a1812-06127_synth_00_nosk" "$SCRIPT_DIR/a1812-06127_synth_0505_nosk" "$SCRIPT_DIR/a1812-06127_synth_11_nosk")
for dir in "${ENSURE_REMOVED[@]}"; do
    if [ -d "$dir" ]; then
        echo "Removing existing directory $dir"
        rm -rf "$dir"
    fi
done

python3 $SCRIPT_DIR/a1812-06127-synth-nosk.py $SCRIPT_DIR/a1812-06127_synth_iid_nosk   --iid  --samples_per_user=280
python3 $SCRIPT_DIR/a1812-06127-synth-nosk.py $SCRIPT_DIR/a1812-06127_synth_00_nosk    --alpha=0.0 --beta=0.0 --samples_per_user=280
python3 $SCRIPT_DIR/a1812-06127-synth-nosk.py $SCRIPT_DIR/a1812-06127_synth_0505_nosk  --alpha=0.5 --beta=0.5 --samples_per_user=280
python3 $SCRIPT_DIR/a1812-06127-synth-nosk.py $SCRIPT_DIR/a1812-06127_synth_11_nosk    --alpha=1.0 --beta=1.0 --samples_per_user=280

python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_iid_nosk/test.json $SCRIPT_DIR/a1812-06127_synth_iid_nosk/
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_iid_nosk/train.json $SCRIPT_DIR/a1812-06127_synth_iid_nosk/
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_00_nosk/test.json $SCRIPT_DIR/a1812-06127_synth_00_nosk/
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_00_nosk/train.json $SCRIPT_DIR/a1812-06127_synth_00_nosk/
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_0505_nosk/test.json $SCRIPT_DIR/a1812-06127_synth_0505_nosk/
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_0505_nosk/train.json $SCRIPT_DIR/a1812-06127_synth_0505_nosk/
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_11_nosk/test.json $SCRIPT_DIR/a1812-06127_synth_11_nosk/   
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_11_nosk/train.json $SCRIPT_DIR/a1812-06127_synth_11_nosk/

rm -f $SCRIPT_DIR/a1812-06127_synth_iid_nosk/test.json
rm -f $SCRIPT_DIR/a1812-06127_synth_iid_nosk/train.json
rm -f $SCRIPT_DIR/a1812-06127_synth_00_nosk/test.json
rm -f $SCRIPT_DIR/a1812-06127_synth_00_nosk/train.json
rm -f $SCRIPT_DIR/a1812-06127_synth_0505_nosk/test.json
rm -f $SCRIPT_DIR/a1812-06127_synth_0505_nosk/train.json
rm -f $SCRIPT_DIR/a1812-06127_synth_11_nosk/test.json
rm -f $SCRIPT_DIR/a1812-06127_synth_11_nosk/train.json


python3 $SCRIPT_DIR/merge.py $SCRIPT_DIR/a1812-06127_synth_iid_nosk/train.npz $SCRIPT_DIR/a1812-06127_synth_iid_nosk/test.npz $SCRIPT_DIR/a1812-06127_synth_iid_nosk/data.npz
python3 $SCRIPT_DIR/merge.py $SCRIPT_DIR/a1812-06127_synth_00_nosk/train.npz $SCRIPT_DIR/a1812-06127_synth_00_nosk/test.npz $SCRIPT_DIR/a1812-06127_synth_00_nosk/data.npz
python3 $SCRIPT_DIR/merge.py $SCRIPT_DIR/a1812-06127_synth_0505_nosk/train.npz $SCRIPT_DIR/a1812-06127_synth_0505_nosk/test.npz $SCRIPT_DIR/a1812-06127_synth_0505_nosk/data.npz
python3 $SCRIPT_DIR/merge.py $SCRIPT_DIR/a1812-06127_synth_11_nosk/train.npz $SCRIPT_DIR/a1812-06127_synth_11_nosk/test.npz $SCRIPT_DIR/a1812-06127_synth_11_nosk/data.npz

echo "All datasets generated and merged successfully."
