#!/bin/bash
#SBATCH --job-name=run_A9
#SBATCH --output=logs/run_A9_%j.out
#SBATCH --error=logs/run_A9_%j.err
#SBATCH --cpus-per-task=1

julia $BASEDIR/run_A9.jl -r $REPS --d_size $DSIZE --ndemes $ND --Nmmin $NM_MIN --Nmmax $NM_MAX --Nm_is_log --Nm_grid_size $NM_GRID_SIZE --slurm --cost $COST --bmin $B_MIN --bmax $B_MAX --B_grid_size $B_GRID_SIZE --output $OUTPUT