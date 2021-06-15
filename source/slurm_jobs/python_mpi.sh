#!/bin/bash
# Author: Saul Pierotti
# Mail: saulpierotti.bioinfo@gmail.com
# Last updated: 24/04/2021

# my project id (from projinfo)
#SBATCH -A SNIC2020-5-300

# Send stderr of my program into <jobid>.error
#SBATCH --error=logs/%J.error

# Send stdout of my program into <jobid>.output
#SBATCH --output=logs/%J.output

mkdir -p logs
ml purge >/dev/null 2>&1
ml load foss
ml load Python/3.8.6
source "$HOME/projects/virtualenvs/random_search/bin/activate"
mpiexec python "$@"
