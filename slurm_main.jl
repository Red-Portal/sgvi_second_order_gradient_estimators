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

using Serialization
using Distributed, SlurmClusterManager

rootdir = @__DIR__

addprocs(SlurmManager(), topology=:master_worker, dir=rootdir)

@everywhere using Pkg
@everywhere Pkg.activate(".")

include("run_experiments.jl")
main()
