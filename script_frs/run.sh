#!/bin/bash

#SBATCH --partition=geo #glados12,glados16
#SBATCH --nodelist=g25
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --array=1-324
#SBATCH --output="slurmout/slurm-%A_%a.out"

timestr=`sed -n "${SLURM_ARRAY_TASK_ID}p" times_detections_2021.list | awk -F, '{print $1}'`

echo "Time: ${timestr}"
python write_events.py ${timestr}
