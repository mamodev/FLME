
set -e

python3 generate_iid.py
python3 adapt.py test.json ../../.splits/fp-iid/
python3 adapt.py train.json ../../.splits/fp-iid/
python3 ../../lib/merge.py ../../.splits/fp-iid/test.npz ../../.splits/fp-iid/train.npz  ../../.data/fp-iid.npz


python3 generate_synthetic_0_0.py
python3 adapt.py test.json ../../.splits/fp-0-0/
python3 adapt.py train.json ../../.splits/fp-0-0/
python3 ../../lib/merge.py ../../.splits/fp-0-0/test.npz ../../.splits/fp-0-0/train.npz  ../../.data/fp-0-0.npz

python3 generate_synthetic_1_1.py
python3 adapt.py test.json ../../.splits/fp-1-1/
python3 adapt.py train.json ../../.splits/fp-1-1/
python3 ../../lib/merge.py ../../.splits/fp-1-1/test.npz ../../.splits/fp-1-1/train.npz  ../../.data/fp-1-1.npz

python3 generate_synthetic_0.5_0.5.py
python3 adapt.py test.json ../../.splits/fp-0.5-0.5/
python3 adapt.py train.json ../../.splits/fp-0.5-0.5/
python3 ../../lib/merge.py ../../.splits/fp-0.5-0.5/test.npz ../../.splits/fp-0.5-0.5/train.npz  ../../.data/fp-0.5-0.5.npz


# generate merged dataset (for ui compatibility)




rm test.json
rm train.json

