#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH -p geo,glados12,glados16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --array=1-200
#SBATCH --output="slurmout/slurm-%A_%a.out"

day=$(sed -n "${SLURM_ARRAY_TASK_ID}p" dates.list | awk '{print $1}')
echo "DAY: ${day}"

python detect.py $day
#python make_templates.py $day
#python get_full_template.py $day
