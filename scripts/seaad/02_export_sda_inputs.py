"""02_export_sda_inputs.py: SEA-AD SDA input and the SLURM array script for the ten SDA runs.

Stacks the ten `<cell type>_rpm_normalized.csv` tables (cell types in directory-listing order) into
`sda/rna.txt` (donor x gene x cell type), with `sda/md_samples.txt`, `sda/genes.txt` and
`sda/cell_types.txt`, and writes `sda/matd.sh`: ten SDA v1.1 runs (800 components requested,
3,000 iterations, saved every 1,500). The record's `cell_types.txt` gives the order used. Paths: `_paths.py`;
the SDA executable is G_ROS_SDA_BINARY (default `sda`) and the cluster folder written into `matd.sh`
is G_ROS_CLUSTER_WORKDIR (default G_ROS_SEAAD_DIR), as in scripts/05_export_sda_inputs.py.

From the original analysis (dlpfc1a_export_data.py, 2023). Changes: paths (the original wrote the
cluster paths and executable into `matd.sh` verbatim) and the unused natsort import removed.
"""
import numpy as np
import scipy as sp
import pandas as pd
from os import path, makedirs, listdir, environ
from _paths import SEAAD_DIR
SDA_BINARY = environ.get('G_ROS_SDA_BINARY', 'sda')  # SDA v1.1 executable
CLUSTER_WORKDIR = environ.get('G_ROS_CLUSTER_WORKDIR', SEAAD_DIR)  # folder on the cluster that holds sda/


save_directory = SEAAD_DIR + "/"

# Lists to collect DataFrames, genes, samples, and cell types
dfs = []
genes = []
samples = []
cell_types = []

# Now get  Iterate over all the files in the directory
to_concat = []
start = True
for file_name in listdir(save_directory):
    if "_rpm_normalized.csv" in file_name:  # To ensure we only process the rpm normalized files
        cell_type = file_name.replace('_rpm_normalized.csv', '').replace('_', ':')
        df = pd.read_csv(save_directory + file_name, index_col=0)
    
        print(cell_type)
        cell_types.append(cell_type)
        to_concat.append(df)
    
        if start:
            samples = df.index.tolist()
            genes = df.columns.tolist()
            start = False
        else:
            if samples != df.index.tolist():
                print('samples not the same')
                break
            if genes != df.columns.tolist():
                print('genes not the same')
                break

# Concatenate DataFrames
concatenated_df = pd.concat(to_concat, axis=0)

# Wrtie it out and see what happens
directory = save_directory + 'sda/'

if not path.exists(directory):
    makedirs(directory)

concatenated_df.to_csv(directory + 'rna.txt', header=False, index=False, sep=' ', na_rep='NA')

f = open(directory + 'md_samples.txt', 'w')
for samp in samples:
    f.write(samp + '\n')
f.close()

f = open(directory + 'genes.txt', 'w')
for gene in genes:
    f.write(gene + '\n')
f.close()

f = open(directory + 'cell_types.txt', 'w')
for cell in cell_types:
    f.write(cell + '\n')
f.close()

cores = '10'

# Sort the cluster commands out
f = open(directory + 'matd.sh', "w")

f.write('#!/bin/bash\n\n')

f.write('#SBATCH -A combat.prj\n')
f.write('#SBATCH -J dlpfc\n')
f.write('#SBATCH -o dlpfc-%j.out\n')
f.write('#SBATCH --array 1-10\n')
f.write('#SBATCH -c ' + cores + '\n')
f.write('#SBATCH -p long\n')
f.write('#SBATCH -e dlpfc-%j.err\n\n')

f.write('echo "#######################################################################"\n')
f.write('echo `date`: Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} \n')
f.write('echo "Run on host: $(hostname)\n')
f.write('echo "Operating system: $(uname -s)\n')
f.write('echo "Username: $(whoami)\n')
f.write('echo "Started at: $(date)\n')
f.write('echo "#######################################################################"\n\n')

f.write('mkdir ' + CLUSTER_WORKDIR + '/sda/results${SLURM_ARRAY_TASK_ID}\n')
f.write('export OPENMP_NUM_THREADS=' + cores + '\n')

to_write = SDA_BINARY + ' --data '

to_write = to_write + CLUSTER_WORKDIR + '/sda/rna.txt '

to_write = to_write + '--N '+str(len(samples))+' --out ' + CLUSTER_WORKDIR + '/sda/results${SLURM_ARRAY_TASK_ID} --num_openmp_threads ' + cores + ' --num_blocks 8 --eigen_parallel true --num_comps 800 --save_freq 1500 --remove_zero_comps true --max_iter 3000 --ignore_missing true \n\n'

f.write(to_write)

f.write('echo "#########################################################################"\n')
f.write('echo "Finished at: "`date`\n')
f.write('echo "#########################################################################"\n')
f.write('exit 0')

f.close()
