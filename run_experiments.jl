
using AdvancedVI
using Base.Iterators
using DataFrames
using Distributed
using Distributions, LinearAlgebra
using JLD2
using LogDensityProblems, StanLogDensityProblems, PosteriorDB
using Random, Random123
using Suppressor

include("capability_adjusted.jl")
include("klminproxrepgraddescentgaussian.jl")
include("klminwassfwdbwd_patch.jl")
include("optimize_patch.jl")

function LoadStanProblem(
    post::PosteriorDB.Posterior, path::AbstractString; force::Bool=false, kwargs...
)
    model = PosteriorDB.model(post)
    data = PosteriorDB.load(PosteriorDB.dataset(post), String)
    lib = joinpath(path, "$(model.name)_model.so")
    if isfile(lib)
        return StanLogDensityProblems.StanProblem(lib, data; kwargs...)
    else
        stan_file = PosteriorDB.path(PosteriorDB.implementation(model, "stan"))
        stan_file_new = joinpath(path, basename(stan_file))
        cp(stan_file, stan_file_new; force=force)
        return StanLogDensityProblems.StanProblem(stan_file_new, data; kwargs...)
    end
end

function load_model(name)
    if !isdir(".stan")
        mkdir(".stan")
    end
    pdb = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, name)
    prob = LoadStanProblem(
        post, ".stan/"; force=true, nan_on_error=true, make_args=["STAN_THREADS=true"]
    )
    return prob
end

function run_experiment(rng, prob_name, order, alg, n_iters, n_thin)
    prob = @suppress load_model(prob_name)
    prob = CapabilityAdjusted(prob, LogDensityProblems.LogDensityOrder{order}())
    d    = d = LogDensityProblems.dimension(prob)
    q    = FullRankGaussian(zeros(d), LowerTriangular(Matrix{Float64}(0.34 * I, d, d)))

    t_begin = time()
    function callback(; rng, iteration, q, kwargs...)
        if (mod(iteration, n_thin) == 1) || (iteration == n_iters)
            elbo =
                -estimate_objective(
                    rng, RepGradELBO(2^10; entropy=MonteCarloEntropy()), q, prob
                )
            return (elbo_avg=elbo, elapsed=time() - t_begin)
        else
            nothing
        end
    end

    try
        _, info, _ = optimize(rng, alg, n_iters, prob, q; callback, show_progress=false)
        xs = 1:n_thin:n_iters
        ts = [i.elapsed for i in info[xs]]
        ys = [i.elbo_avg for i in info[xs]]
        return xs, ts, ys
    catch 
        xs = 1:n_thin:n_iters
        ts = fill(0.0, length(xs))
        ys = fill(-Inf, length(xs))
        return xs, ts, ys
    end
end

function main()
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    rng = Random123.Philox4x(UInt64, seed, 8)

    n_iters      = 5000
    n_thin       = 100
    n_reps       = 8
    problems     = [
        "dogs-dogs",
        "surgical_data-surgical_model",
        "rats_data-rats_model",
        "bones_data-bones_model",
        "butterfly-multi_occupancy",
        "GLMM_Poisson_data-GLMM_Poisson_model",
        "pilots-pilots",
        "nes2000-nes",
        #"election88-election88_full",
        "hudson_lynx_hare-lotka_volterra",
        "loss_curves-losscurve_sislob",
        "gp_pois_regr-gp_pois_regr",
        "rstan_downloads-prophet",
        "bball_drive_event_1-hmm_drive_1",
        "uk_drivers-state_space_stochastic_level_stochastic_seasonal",
        "radon_mn-radon_hierarchical_intercept_centered",
        "three_men1-ldaK2",
        "sat-hier_2pl",
        "science_irt-grsm_latent_reg_irt",
        "timssAusTwn_irt-gpcm_latent_reg_irt",
    ]
    logstepsizes =
        [(logstepsize = logstepsize,) for logstepsize in range(-8, -2; step=0.25)]
    algorithms   = [(algorithm = "WVI",), (algorithm = "BBVI",),]
    orders       = [(order = 1,), (order = 2,)]
    keys         = [(key = key,) for key in 1:n_reps]

    @info("Load models")
    @showprogress for problem in problems
        @suppress load_model(problem)
    end
    @info("Load models - done")

    @info("Run experiments")
    configs = Iterators.product(orders, logstepsizes, algorithms, keys) |> collect
    configs = reshape(configs, :)
    configs = map(x -> merge(x...), configs)

    for problem in problems
        fname = "data/raw/$(problem).jld2"

        if isfile(fname)
            @info("File $(fname) is already present. Skipping experiment.")
            continue
        end
        dfs = @showprogress pmap(configs) do config
            (; key, logstepsize, algorithm, order) = config

            rng_local = deepcopy(rng)
            Random123.set_counter!(rng_local, key)
            
            alg = if algorithm == "WVI"
                KLMinWassFwdBwd(; n_samples=8, stepsize=10^logstepsize)
            else
                KLMinProxRepGradDescentGaussian(; n_samples=8, stepsize=10^logstepsize)
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
    @info("Run experiments - done")
end
