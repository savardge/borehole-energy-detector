#!/bin/bash
#SBATCH -p geo,glados12,glados16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --array=1-16
#SBATCH --output="outslurm/slurm-%A_%a.out"

day=$(sed -n "${SLURM_ARRAY_TASK_ID}p" dates.list | awk '{print $1}')
echo "DAY: ${day}"

python detector.py $day
#python make_templates.py $day
#python get_full_template.py $day
