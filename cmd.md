python tm-server.py --timeline=.timelines/ss.json --repo-folder=.simulations/sim-ss-data --ds-folder=.splits/data --nuke-repo --lr-range 0.1 0.1


python remote.py run "kill \$(pgrep -f client.py)" all && python remote.py run "rm flcdata/tm-client.py" && python remote.py upload tm-client.py flcdata/ all   

python remote.py runprog "cd flcdata && source venv/bin/activate && python3 tm-client.py --host mamodeh.ddns.net --port 6969 --name @host" all

python remote.py upload tm-client.py flcdata/ all   


# upload new version

python remote.py run "kill \$(pgrep -f client.py)" all; python remote.py run "rm flcdata/tm-client.py" all; python remote.py upload tm-client.py flcdata/ all   
