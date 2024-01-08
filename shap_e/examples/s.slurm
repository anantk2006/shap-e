#!/bin/sh

#!/bin/bash
#SBATCH --job-name=varsize
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.40gb:1

#SBATCH --output=varsize.out
#SBATCH --error=varsize.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=30G
#SBATCH --time=0-00:5:00


## Load the relevant modules needed for the job
## Hopper
module load gnu9/9.3.0
module load cudnn/8.0.4.30-11.0-gi
module load python/3.8.6-ff
module load anaconda3
echo started2
cd /home/akhande/Syslab/shap-e/shap_e/examples
conda init
source ~/.bashrc
conda activate syslab
python3 poc.py prompt="a backpack" --second=" with a water bottle on the side"