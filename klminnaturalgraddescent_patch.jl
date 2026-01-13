
# The AdvancedVI.jl@0.6.1 implementation of KLMinWassFwdBwd has a subtle bug 
# This file overrides the function `AdvancedVI.step` that contains the error

using AdvancedVI
using AdvancedVI:
    gaussian_expectation_gradient_and_hessian!,
    step,
    subsample,
    KLMinNaturalGradDescentState,
    KLMinNaturalGradDescent
using LinearAlgebra
using Random
using Statistics

function AdvancedVI.step(
    rng::Random.AbstractRNG,
    alg::KLMinNaturalGradDescent,
    state,
    callback,
    objargs...;
    kwargs...,
)
    (; ensure_posdef, n_samples, stepsize, subsampling) = alg
    (; q, prob, prec, qcov, iteration, sub_st, grad_buf, hess_buf) = state

    m = mean(q)
    S = prec
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

    logπ_avg, grad_buf, hess_buf = gaussian_expectation_gradient_and_hessian!(
        rng, q, n_samples, grad_buf, hess_buf, prob_sub
    )

    S′ = if ensure_posdef
        # Udpate rule guaranteeing positive definiteness in the proof of Theorem 1.
        # Lin, W., Schmidt, M., & Khan, M. E.
        # Handling the positive-definite constraint in the Bayesian learning rule.
        # In ICML 2020.
        G_hat = S - -hess_buf
        Hermitian(S - η*G_hat + η^2/2*G_hat*qcov*G_hat)
    else
        Hermitian(((1 - η) * S + η * Symmetric(-hess_buf)))
    end
    m′ = m - η * (S′ \ (-grad_buf))

    prec_chol = cholesky(S′).L
    prec_chol_inv = inv(prec_chol)
    scale = prec_chol_inv'
    qcov = Hermitian(scale*scale')
    q′ = MvLocationScale(m′, scale, q.dist)

    state = KLMinNaturalGradDescentState(
        q′, prob, S′, qcov, iteration, sub_st′, grad_buf, hess_buf
    )
    elbo = logπ_avg + entropy(q′)
    info = merge((elbo=elbo,), sub_inf)

    if !isnothing(callback)
        info′ = callback(; rng, iteration, q=q′, info)
        info = !isnothing(info′) ? merge(info′, info) : info
    end
    state, false, info
end
