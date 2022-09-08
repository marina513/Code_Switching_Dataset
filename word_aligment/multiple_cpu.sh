#!/bin/bash

#SBATCH -J ASR_CS
#SBATCH --error=/home/mmaher/LM_CS/word_aligment/Error/%j%x.err # error file
#SBATCH --output=/home/mmaher/LM_CS/word_aligment/Error/%j%x.out # output log file
#SBATCH --time=24:00:00 # 24 hours of wall time
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=3 #number of tasks per node
#SBATCH --partition=batch # gpu partition



for f in "/home/mmaher/LM_CS/split/0-1K/"*
do
    echo "Processing $f dir..."
    python aligment.py  $f  &

done
wait
