#!/bin/bash

#SBATCH -p geo,glados12,glados16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
###SBATCH --array=4-3845 
#SBATCH --array=350,351,352,353,354,355,356,357,438,439,440,441,442,443,444,445,446,453,455,463,464,465,466,475,476,477,478,479,480,481,517,518,519,520,521,522,523,524,525,526,595,605,606,607,608,609,610,611,612,624,638,639,640,654,655,656,657,658,659,660,808,809,810,811,812,813,814,815,816,817,838,839,856,859,860,867,870,871,872,873,957,958,959,960,961,962,963,964,965,966,1091,1092,1093,1101,1102,1104,1131,1132,1133,1134,3705,3780,3781,3782,3783,3784,3795,3831,3832,3834,3835,3836,3837,3838,3840,3841
#SBATCH --output="outslurm/slurm-%A_%a.out"

DETTIME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" times_deep_events.list)

echo "File: $DETTIME"

python extract_waveforms.py "${DETTIME}"