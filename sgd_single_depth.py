#! python

#$ -N kedro

# execute from current directory
#$ -cwd

# Preserve environment variables
#$ -V

# Provide path to python executable
#$ -S /home/stenger/smaxxhome/anaconda3/envs/pydiver/bin/python

# Merge error and out
#$ -j yes

# serial queue
#$ -q taranis-gpu1.q
# -q teutates.q
# -q grannus.q

# Path for output
#$ -o /data.bmp/heart/DataAnalysis/2020_3DExMedSurfaceToDepth/FromSurface2DepthFinal/logs/jobs

# job array of length 1
#$ -t 1:2


import os
import numpy as np


def main():
    print(os.environ['SGE_TASK_ID'])
    SGE_TASK_ID = int(os.environ['SGE_TASK_ID']) - 1

    depth = str(SGE_TASK_ID)
    time_steps = 32

    os.system(f"python train.py -depth {depth}")
    #os.system(f"python train.py --depth {depth} --time_steps {time_steps}")


if __name__ == '__main__':
    main()
