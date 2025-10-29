#!/bin/bash
set -e  

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

ENSURE_REMOVED=("$SCRIPT_DIR/a1812-06127_synth_iid_nqsk" "$SCRIPT_DIR/a1812-06127_synth_00_nqsk" "$SCRIPT_DIR/a1812-06127_synth_0505_nqsk" "$SCRIPT_DIR/a1812-06127_synth_11_nqsk")
for dir in "${ENSURE_REMOVED[@]}"; do
    if [ -d "$dir" ]; then
        echo "Removing existing directory $dir"
        rm -rf "$dir"
    fi
done

python3 $SCRIPT_DIR/a1812-06127-synth-nqsk.py $SCRIPT_DIR/a1812-06127_synth_iid_nqsk   --iid  --samples_per_user=280
python3 $SCRIPT_DIR/a1812-06127-synth-nqsk.py $SCRIPT_DIR/a1812-06127_synth_00_nqsk    --alpha=0.0 --beta=0.0 --samples_per_user=280
python3 $SCRIPT_DIR/a1812-06127-synth-nqsk.py $SCRIPT_DIR/a1812-06127_synth_0505_nqsk  --alpha=0.5 --beta=0.5 --samples_per_user=280
python3 $SCRIPT_DIR/a1812-06127-synth-nqsk.py $SCRIPT_DIR/a1812-06127_synth_11_nqsk    --alpha=1.0 --beta=1.0 --samples_per_user=280

python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_iid_nqsk/test.json $SCRIPT_DIR/a1812-06127_synth_iid_nqsk/
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_iid_nqsk/train.json $SCRIPT_DIR/a1812-06127_synth_iid_nqsk/
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_00_nqsk/test.json $SCRIPT_DIR/a1812-06127_synth_00_nqsk/
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_00_nqsk/train.json $SCRIPT_DIR/a1812-06127_synth_00_nqsk/
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_0505_nqsk/test.json $SCRIPT_DIR/a1812-06127_synth_0505_nqsk/
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_0505_nqsk/train.json $SCRIPT_DIR/a1812-06127_synth_0505_nqsk/
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_11_nqsk/test.json $SCRIPT_DIR/a1812-06127_synth_11_nqsk/   
python3 $SCRIPT_DIR/a1812-06127-adapt.py $SCRIPT_DIR/a1812-06127_synth_11_nqsk/train.json $SCRIPT_DIR/a1812-06127_synth_11_nqsk/

rm -f $SCRIPT_DIR/a1812-06127_synth_iid_nqsk/test.json
rm -f $SCRIPT_DIR/a1812-06127_synth_iid_nqsk/train.json
rm -f $SCRIPT_DIR/a1812-06127_synth_00_nqsk/test.json
rm -f $SCRIPT_DIR/a1812-06127_synth_00_nqsk/train.json
rm -f $SCRIPT_DIR/a1812-06127_synth_0505_nqsk/test.json
rm -f $SCRIPT_DIR/a1812-06127_synth_0505_nqsk/train.json
rm -f $SCRIPT_DIR/a1812-06127_synth_11_nqsk/test.json
rm -f $SCRIPT_DIR/a1812-06127_synth_11_nqsk/train.json


python3 $SCRIPT_DIR/merge.py $SCRIPT_DIR/a1812-06127_synth_iid_nqsk/train.npz $SCRIPT_DIR/a1812-06127_synth_iid_nqsk/test.npz $SCRIPT_DIR/a1812-06127_synth_iid_nqsk/data.npz
python3 $SCRIPT_DIR/merge.py $SCRIPT_DIR/a1812-06127_synth_00_nqsk/train.npz $SCRIPT_DIR/a1812-06127_synth_00_nqsk/test.npz $SCRIPT_DIR/a1812-06127_synth_00_nqsk/data.npz
python3 $SCRIPT_DIR/merge.py $SCRIPT_DIR/a1812-06127_synth_0505_nqsk/train.npz $SCRIPT_DIR/a1812-06127_synth_0505_nqsk/test.npz $SCRIPT_DIR/a1812-06127_synth_0505_nqsk/data.npz
python3 $SCRIPT_DIR/merge.py $SCRIPT_DIR/a1812-06127_synth_11_nqsk/train.npz $SCRIPT_DIR/a1812-06127_synth_11_nqsk/test.npz $SCRIPT_DIR/a1812-06127_synth_11_nqsk/data.npz

echo "All datasets generated and merged successfully."
