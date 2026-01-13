#!/usr/bin/env julia
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=genoa-std-mem
#SBATCH --ntasks=4
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=00:05:00
#SBATCH --qos=normal

using Distributed, SlurmClusterManager

addprocs(SlurmManager())
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere include("run_experiments.jl")

main()
