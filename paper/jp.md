jupyter notebook --no-browser --NotebookApp.token='' --NotebookApp.password='' --ip=127.0.0.1 --port=8888


python3 tools/simulation/timeline.py --timeline sync.json --ds-folder data/cubeds --repo-folder cubeds-sync --worker-per-device 5


python3 data/train_split.py --input_file data/cubeds/data.npz --output_dir  data/cubeds/ 

python3 tools/simulation/analyze.py --net simple --sim-dir cubeds-sync3 --dataset-dir data/cubeds/


python3 tools/simulation/serial_timeline.py --net simple --timeline sync.json --ds-folder data/cubeds --repo-folder cubeds-sync3 --nuke-repo


python3 tools/simulation/timeline.py --net logistic  --timeline sync.json --ds-folder data/a1812-06127_synth_11/ --repo-folder s11 --worker-per-device 5 --nuke-repo


python3 tools/simulation/analyze.py --net logistic --sim-dir s11 --dataset-dir data/a1812-06127_synth_11/

# edprox-exp-drp05.json 
python3 tools/simulation/timeline.py --net logistic  --timeline fedprox-exp-drp05.json --ds-folder data/a1812-06127_synth_11/ --repo-folder fp-drp05-s11 --worker-per-device 5 --nuke-repo
python3 tools/simulation/analyze.py --net logistic --sim-dir fp-drp05-s11 --dataset-dir data/a1812-06127_synth_11/


python3 tools/simulation/timeline.py --net logistic  --timeline fedprox-exp-p1000-drp05.json --ds-folder data/a1812-06127_mnist/ --repo-folder fp-drp05-s11 --worker-per-device 5 --nuke-repo

python3 tools/simulation/analyze.py --net logistic --sim-dir fp-drp05-s11 --dataset-dir  data/a1812-06127_mnist/ 
