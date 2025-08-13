python tm-server.py --timeline=.timelines/ss.json --repo-folder=.simulations/sim-ss-data --ds-folder=.splits/data --nuke-repo --lr-range 0.1 0.1


python remote.py run "kill \$(pgrep -f client.py)" all && python remote.py run "rm flcdata/tm-client.py" && python remote.py upload tm-client.py flcdata/ all   

python remote.py runprog "cd flcdata && source venv/bin/activate && python3 tm-client.py --host mamodeh.ddns.net --port 6969 --name @host" all

python remote.py upload tm-client.py flcdata/ all   


# upload new version

python remote.py run "kill \$(pgrep -f client.py)" all; python remote.py run "rm flcdata/tm-client.py" all; python remote.py upload tm-client.py flcdata/ all   


python3 cmds/FLME/timeline.py --timeline .timelines/sync.json --repo-folder=.simulations/prova --ds-folder=.splits/data/ --nuke-repo


python3 tools/simulation/timeline.py --timeline sync.json  --ds-folder=data/a1812-06127_synth_00 --repo-folder=prova --nuke-repo