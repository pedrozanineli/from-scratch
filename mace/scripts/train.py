import sys
import os
import subprocess
import numpy as np

train_data_path = '/home/p.zanineli/work/from-scratch/data/train.xyz' 
test_data_path = '/home/p.zanineli/work/from-scratch/data/test.xyz'

size="small"

model_name=f"mace-fs-{size}"

from codecarbon import EmissionsTracker
tracker = EmissionsTracker(output_dir='emissions',output_file=f'{size}.csv',allow_multiple_runs=True)
tracker.start()

if not os.path.exists(model_name+'.model'):

    # logfile = open(f'train_{size}.log','w')

    subprocess.run([
        "python",
        "run_train.py",
        f"--name={size}",
        f"--train_file={train_data_path}",
        f"--test_file={test_data_path}",
        f"--valid_fraction=0.05",
        "--E0s=average",
        "--energy_weight=1.0",
        "--forces_weight=10.0",

        # "--hidden_irreps=128x0e",
        "--hidden_irreps=128x0e + 128x1o",
        
        "--r_max=6",
        "--batch_size=5",
        "--max_num_epochs=200",
        "--device=cuda",
        "--ema",
        "--ema_decay=0.99",
        "--amsgrad",
        "--eval_interval=10",
        "--patience=100",
        "--forces_key=forces",
        "--energy_key=energy"])

tracker.stop()
