import glob
import os
import shutil

from definitions import RUNS_DIR

ppo_dir = os.path.join(RUNS_DIR, "ppo")
step_dirs = glob.glob(os.path.join(ppo_dir, '*/*/Step_*'))
print(len(step_dirs))
count = 0
for dir in step_dirs:
    stepcount = int(dir.split("Step_")[-1])
    if stepcount % 500000 != 0:
        count += 1
        shutil.rmtree(dir)
print(count)
