#!/usr/bin/env julia
#SBATCH --output=logs/%x_%j.out     # Stdout goes to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err      # Stderr goes to logs/jobname_jobid.err
#SBATCH --partition=genoa-std-mem   # Queue to submit to
#SBATCH --ntasks=4                  # Number of tasks (usually one per process)
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --mem=256G                  # Memory allocation
#SBATCH --time=00:05:00             # Maximum runtime (hh:mm:ss)
#SBATCH --qos=normal                 

using Distributed, SlurmClusterManager


addprocs(SlurmManager())
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere include("run_experiments.jl")

main()
