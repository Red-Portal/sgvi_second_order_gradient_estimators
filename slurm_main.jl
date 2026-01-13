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

ENV["PATH"] = @__DIR__

addprocs(SlurmManager())

@everywhere using Pkg
@everywhere Pkg.activate(ENV["PATH"])
@everywhere include(joinpath(ENV["PATH"], "run_experiments.jl"))
main()
