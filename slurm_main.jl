#!/usr/bin/env julia --project=.

using Distributed
using LoggingExtras

const problem  = ENV["SLURM_JOB_NAME"][4:end]
const num_cpus = parse(Int, ENV["SLURM_CPUS_PER_TASK"])

@info("Connect workers")
exeflags = ["--project=$(ENV["SLURM_SUBMIT_DIR"])", "--threads=1"]
addprocs(num_cpus; exeflags)
@info("Connect workers - done")

@info("Setting up paths")
@everywhere cd(ENV["SLURM_SUBMIT_DIR"])
@everywhere using Pkg
@everywhere Pkg.activate(".")
@info("Setting up paths - done")

@info("Loading main script")
@everywhere include(joinpath(ENV["SLURM_SUBMIT_DIR"], "run_experiments.jl"))
@info("Loading main script - done")

function main()
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    rng = Random123.Philox4x(UInt64, seed, 8)

    n_iters      = 4001
    n_thin       = 100
    n_reps       = 16
    logstepsizes =
        [(logstepsize = logstepsize,) for logstepsize in range(-8, 0; step=0.2)]
    algorithms   = [(algorithm = "WVI",), (algorithm = "BBVI",), (algorithm = "NGVI",)]
    orders       = [(order = 1,), (order = 2,)]
    keys         = [(key = key,) for key in 1:n_reps]

    @info("Load models")
    @suppress load_model(problem)
    @info("Load models - done")

    @info("Run experiments")
    configs = Iterators.product(orders, logstepsizes, algorithms, keys) |> collect
    configs = reshape(configs, :)
    configs = map(x -> merge(x...), configs)

    fname = "data/raw/$(problem).jld2"

    if isfile(fname)
        @info("File $(fname) is already present. Skipping experiment.")
        return
    end
    dfs = @showprogress pmap(configs) do config
        (; key, logstepsize, algorithm, order) = config

        rng_local = deepcopy(rng)
        Random123.set_counter!(rng_local, key)
            
        alg = if algorithm == "WVI"
            KLMinWassFwdBwd(; n_samples=8, stepsize=10^logstepsize)
        elseif algorithm == "BBVI"
            KLMinProxRepGradDescentGaussian(; n_samples=8, stepsize=10^logstepsize)
        elseif algorithm == "NGVI"
            KLMinNaturalGradDescent(; n_samples=4, stepsize=10^logstepsize)
        end

        xs, ts, ys = run_experiment(rng, problem, order, alg, n_iters, n_thin)
        df = DataFrame(
            key=key,
            iteration=xs,
            elbo=ys,
            time=ts,
            algorithm=algorithm,
            order=order,
            problem=problem,
            logstepsize=logstepsize
        )
        GC.gc()
        df
    end
    df = vcat(dfs...)
    JLD2.save(fname, "data", df)
end

main()
exit()
