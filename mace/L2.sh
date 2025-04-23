#!/bin/bash
#SBATCH --job-name=L2
#SBATCH --output=output_2.log
#SBATCH --error=error_2.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=32G

module load cuda/12.0
source ~/.bashrc
conda activate mace

start=$(date +%s)

mace_run_train \
    --name="MACE_ZrO2_L2" \
    --train_file="../data/train.xyz" \
    --valid_fraction=0.05 \
    --test_file="../data/test.xyz" \
    --valid_batch_size=20 \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{40:-3349.0, 8:-1029.2809654211628}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o + 128x2e' \
    --r_max=5.0 \
    --batch_size=10 \
    --max_num_epochs=1500 \
    --stage_two \
    --start_stage_two=1200 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
    --eval_interval=10 \
    --patience=100 \
    --plot_frequency=10

end=$(date +%s)
duration=$((end - start))
echo "total time: $duration seconds"

