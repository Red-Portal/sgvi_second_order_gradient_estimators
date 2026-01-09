
# The AdvancedVI.jl@0.6.1 implementation of KLMinWassFwdBwd has a subtle bug 
# This file overrides the function `AdvancedVI.step` that contains the error

using AdvancedVI
using AdvancedVI:
    gaussian_expectation_gradient_and_hessian!,
    step,
    subsample,
    KLMinWassFwdBwdState,
    KLMinWassFwdBwd
using LinearAlgebra
using Statistics

function AdvancedVI.step(
    rng::Random.AbstractRNG, alg::KLMinWassFwdBwd, state, callback, objargs...; kwargs...
)
    (; n_samples, stepsize, subsampling) = alg
    (; q, prob, sigma, iteration, sub_st, grad_buf, hess_buf) = state

    m = mean(q)
    Σ = sigma
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

    # Estimate the Wasserstein gradient
    logπ_avg, grad_buf, hess_buf = gaussian_expectation_gradient_and_hessian!(
        rng, q, n_samples, grad_buf, hess_buf, prob_sub
    )

    m′ = m - η * (-grad_buf)
    M = I - η * (-hess_buf')
    Σ_half = Hermitian(M*Σ*M')

    # Compute the JKO proximal operator
    Σ′ = (Σ_half + 2*η*I + sqrt(Hermitian(Σ_half*(Σ_half + 4*η*I))))/2
    q′ = MvLocationScale(m′, cholesky(Σ′).L, q.dist)

    state = KLMinWassFwdBwdState(q′, prob, Σ′, iteration, sub_st′, grad_buf, hess_buf)
    elbo = logπ_avg + entropy(q′)
    info = merge((elbo=elbo,), sub_inf)

    if !isnothing(callback)
        info′ = callback(; rng, iteration, q=q′, info)
        info = !isnothing(info′) ? merge(info′, info) : info
    end
    state, false, info
end
