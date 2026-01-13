
using LogDensityProblems, StanLogDensityProblems, PosteriorDB
using AdvancedVI
using Random, Random123
using Distributions, LinearAlgebra
using Plots, StatsPlots

include("capability_adjusted.jl")
include("klminproxrepgraddescentgaussian.jl")
include("klminwassfwdbwd_patch.jl")
include("klminnaturalgraddescent_patch.jl")
include("optimize_patch.jl")

function main()
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    rng = Random123.Philox4x(UInt64, seed, 8)
    Random123.set_counter!(rng, 2)
    Random.seed!(rand(rng, UInt64))

    #model_name = "loss_curves-losscurve_sislob"
    #model_name = "bball_drive_event_1-hmm_drive_1"

    #model_name = "gp_pois_regr-gp_pois_regr"
    #model_name = "GLMM_Poisson_data-GLMM_Poisson_model"
    #model_name = "dogs-dogs"
    #model_name = "radon_mn-radon_hierarchical_intercept_centered"
    #model_name = "GLMM_data-GLMM1_model"
    model_name = "seeds_data-seeds_model"
    #model_name = "loss_curves-losscurve_sislob"
    #model_name = "butterfly-multi_occupancy"

    if !isdir(".stan")
        mkdir(".stan")
    end
    pdb = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, model_name)
    prob = StanProblem(
        post, ".stan/"; force=true, nan_on_error=true, make_args=["STAN_THREADS=true"]
    )
    d = LogDensityProblems.dimension(prob)

    cap = LogDensityProblems.LogDensityOrder{2}()
    prob = CapabilityAdjusted(prob, cap)

    q = FullRankGaussian(zeros(d), LowerTriangular(Matrix{Float64}(0.34 * I, d, d)))

    n_thin = 30
    t_begin = time()
    function callback(; rng, iteration, q, kwargs...)
        if mod(iteration, n_thin) == 1
            elbo =
                -estimate_objective(
                    rng, RepGradELBO(2^10; entropy=MonteCarloEntropy()), q, prob
                )
            return (elbo_avg=elbo, elapsed=time() - t_begin)
        else
            nothing
        end
    end

    #alg = KLMinWassFwdBwd(; n_samples=10, stepsize=1e-8)
    alg = KLMinProxRepGradDescentGaussian(; n_samples=10, stepsize=1e-8)
    #alg = KLMinNaturalGradDescent(; n_samples=10, stepsize=1e-8)

    q, info, _ = optimize(rng, alg, 1000, prob, q; callback, show_progress=true)

    xs = 1:n_thin:length(info)
    ts = xs #[i.elapsed for i in info[xs]]
    ys = [i.elbo_avg for i in info[xs]]
    display(Plots.plot!(ts, ys; ylims=(quantile(ys, 0.05), Inf)))
    return info
end
