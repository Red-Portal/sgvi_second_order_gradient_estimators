
using AdvancedVI
using AdvancedVI: subsample, init, step, AbstractVariationalAlgorithm, AbstractSubsampling
using LinearAlgebra
using Distributions
using LogDensityProblems

@kwdef struct KLMinProxRepGradDescentGaussian{Sub<:Union{Nothing,<:AbstractSubsampling}} <:
              AbstractVariationalAlgorithm
    stepsize::Float64
    n_samples::Int = 1
    subsampling::Sub = nothing
end

struct KLMinProxRepGradDescentGaussianState{Q,P,S,GradBuf,ScaleBuf}
    q::Q
    prob::P
    iteration::Int
    sub_st::S
    grad_buf::GradBuf
    scale_buf::ScaleBuf
end

function AdvancedVI.init(
    rng::Random.AbstractRNG,
    alg::KLMinProxRepGradDescentGaussian,
    q_init::MvLocationScale{<:LowerTriangular,<:Normal,L},
    prob,
) where {L}
    sub = alg.subsampling
    n_dims = LogDensityProblems.dimension(prob)
    capability = LogDensityProblems.capabilities(typeof(prob))
    if capability < LogDensityProblems.LogDensityOrder{1}()
        throw(
            ArgumentError(
                "`KLMinProxRepGradDescentGaussian` requires at least first-order differentiation capability. The capability of the supplied `LogDensityProblem` is $(capability).",
            ),
        )
    end
    sub_st = isnothing(sub) ? nothing : init(rng, sub)
    grad_buf = Vector{eltype(q_init.location)}(undef, n_dims)
    scale_buf = Matrix{eltype(q_init.location)}(undef, n_dims, n_dims)
    return KLMinProxRepGradDescentGaussianState(
        q_init, prob, 0, sub_st, grad_buf, scale_buf
    )
end

AdvancedVI.output(::KLMinProxRepGradDescentGaussian, state) = state.q

function location_and_scale_grad!(
    rng::Random.AbstractRNG,
    q::MvLocationScale{<:LinearAlgebra.AbstractTriangular,<:Normal,L},
    n_samples::Int,
    grad_buf::AbstractVector{T},
    hess_buf::AbstractMatrix{T},
    prob,
) where {T<:Real,L}
    logπ_avg = zero(T)
    fill!(grad_buf, zero(T))
    fill!(hess_buf, zero(T))

    if LogDensityProblems.capabilities(typeof(prob)) ≤
        LogDensityProblems.LogDensityOrder{1}()
        # First-order-only: use Stein/Price identity for the Hessian
        #
        #   E_{z ~ N(m, CC')} ∇2 log π(z)
        #   = E_{z ~ N(m, CC')} (CC')^{-1} (z - m) ∇ log π(z)T
        #   = E_{u ~ N(0, I)} C' \ (u ∇ log π(z)T) .
        # 
        # Algorithmically, draw u ~ N(0, I), z = C u + m, where C = q.scale.
        # Accumulate A = E[ u ∇ log π(z)T ], then map back: H = C \ A.
        d = LogDensityProblems.dimension(prob)
        u = randn(rng, T, d, n_samples)
        m, C = q.location, q.scale
        z = C*u .+ m
        for b in 1:n_samples
            zb, ub = view(z, :, b), view(u, :, b)
            logπ, ∇logπ = LogDensityProblems.logdensity_and_gradient(prob, zb)
            logπ_avg += logπ/n_samples

            rdiv!(∇logπ, n_samples)
            ∇logπ_div_nsamples = ∇logπ

            grad_buf[:] .+= ∇logπ_div_nsamples
            hess_buf[:, :] .+= ∇logπ_div_nsamples*ub'
        end
        return logπ_avg, grad_buf, hess_buf
    else
        # Second-order: use naive sample average
        z = rand(rng, q, n_samples)
        for b in 1:n_samples
            zb = view(z, :, b)
            logπ, ∇logπ, ∇2logπ = LogDensityProblems.logdensity_gradient_and_hessian(
                prob, zb
            )

            rdiv!(∇logπ, n_samples)
            ∇logπ_div_nsamples = ∇logπ

            rdiv!(∇2logπ, n_samples)
            ∇2logπ_div_nsamples = ∇2logπ

            logπ_avg += logπ/n_samples
            grad_buf[:] .+= ∇logπ_div_nsamples
            hess_buf[:, :] .+= ∇2logπ_div_nsamples
        end
        hess_buf[:, :] .= tril(hess_buf * q.scale)
        return logπ_avg, grad_buf, hess_buf
    end
end

function AdvancedVI.step(
    rng::Random.AbstractRNG,
    alg::KLMinProxRepGradDescentGaussian,
    state,
    callback,
    objargs...;
    kwargs...,
)
    (; n_samples, stepsize, subsampling) = alg
    (; q, prob, iteration, sub_st, grad_buf, scale_buf) = state

    m = q.location
    C = q.scale
    η = convert(eltype(m), stepsize)
    iteration += 1

    # Maybe apply subsampling
    prob_sub, sub_st′, sub_inf = if isnothing(subsampling)
        prob, sub_st, NamedTuple()
    else
        batch, sub_st′, sub_inf = step(rng, subsampling, sub_st)
        prob_sub = subsample(prob, batch)
        prob_sub, sub_st′, sub_inf
    end

    logπ_avg, grad_buf, scale_buf = location_and_scale_grad!(
        rng, q, n_samples, grad_buf, scale_buf, prob_sub
    )

    # Gradient descent step
    m′ = m - η * (-grad_buf)
    C′ = C - η * LowerTriangular(-scale_buf)

    # Proximal step
    diag_idx = diagind(C′)
    scale_diag = C′[diag_idx]
    @. C′[diag_idx] = scale_diag + (sqrt(scale_diag^2 + 4*η) - scale_diag) / 2

    q′ = MvLocationScale(m′, C′, q.dist)

    state = KLMinProxRepGradDescentGaussianState(
        q′, prob, iteration, sub_st′, grad_buf, scale_buf
    )
    elbo = logπ_avg + entropy(q′)
    info = merge((elbo=elbo,), sub_inf)

    if !isnothing(callback)
        info′ = callback(; rng, iteration, q=q′, info)
        info = !isnothing(info′) ? merge(info′, info) : info
    end
    state, false, info
end

"""
    estimate_objective([rng,] alg, q, prob; n_samples)

Estimate the negative ELBO of the variational approximation `q` against the target log-density `prob`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `alg::KLMinProxRepGradDescentGaussian`: Variational inference algorithm.
- `q::MvLocationScale{<:Any,<:Normal,<:Any}`: Gaussian variational approximation.
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.

# Keyword Arguments
- `n_samples::Int`: Number of Monte Carlo samples for estimating the objective. (default: Same as the the number of samples used for estimating the gradient during optimization.)

# Returns
- `obj_est`: Estimate of the objective value.
"""
function AdvancedVI.estimate_objective(
    rng::Random.AbstractRNG,
    alg::KLMinProxRepGradDescentGaussian,
    q::MvLocationScale{S,<:Normal,L},
    prob;
    n_samples::Int=alg.n_samples,
) where {S,L}
    obj = RepGradELBO(n_samples; entropy=MonteCarloEntropy())
    if isnothing(alg.subsampling)
        return estimate_objective(rng, obj, q, prob)
    else
        sub = alg.subsampling
        sub_st = init(rng, sub)
        return mapreduce(+, 1:length(sub)) do _
            batch, sub_st, _ = step(rng, sub, sub_st)
            prob_sub = subsample(prob, batch)
            estimate_objective(rng, obj, q, prob_sub) / length(sub)
        end
    end
end
