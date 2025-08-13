#!/bin/bash
set -e 

# Usage compress-sim.sh <sim_dir>
tar -cjf archive.tar.bz2 -C mydir .
tar -cJf archive.tar.xz -C mydir .
tar --zstd -cf archive.tar.zst -C mydir .