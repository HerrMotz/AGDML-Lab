#!/usr/bin/env bash

python3 split.py

for i in $(find . -type f -name "split_file_*")
do
  sbatch run.sh "$i"
done
