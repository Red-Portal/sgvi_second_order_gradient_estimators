
using Distributed, SlurmClusterManager


addprocs(SlurmManager())
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere include("run_experiments.jl")

main()
