#!/bin/bash 

#SBATCH --job-name=LRSE_data
#SBATCH --array=1-4
#SBATCH -o output_LRSE_data_%a.txt
#SBATCH -e error_LRSE_data_%a.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liuw@mit.edu

#SBATCH -t 4-00:00:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=500GB
#SBATCH --partition=sched_mit_sloan_interactive_r8

# Should use module load mit/matlab/2022a with eosloan.mit.edu/eofe7.mit.edu/eofe8.mit.edu
# Should use module load mit/matlab/matlab-2022a with eofe4.mit.edu
module load mit/matlab/matlab-2022a
matlab -nodesktop -nodisplay -nosplash < gen_data.m ${SLURM_ARRAY_TASK_ID}