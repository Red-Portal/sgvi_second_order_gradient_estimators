
# The AdvancedVI.jl@0.6.1 implementation of `optimize` does not yet support early termination.
# This file overrides `optimize` to step whenever something goes wrong.

using AdvancedVI
using ProgressMeter
using Random

function AdvancedVI.optimize(
    rng::Random.AbstractRNG,
    algorithm::AdvancedVI.AbstractVariationalAlgorithm,
    max_iter::Int,
    prob,
    q_init,
    objargs...;
    show_progress::Bool=true,
    state::Union{<:Any,Nothing}=nothing,
    callback=nothing,
    progress::ProgressMeter.AbstractProgress=ProgressMeter.Progress(
        max_iter; desc="Optimizing", barlen=31, showspeed=true, enabled=show_progress
    ),
    kwargs...,
)
    info_total = NamedTuple[]
    state = if isnothing(state)
        AdvancedVI.init(rng, algorithm, q_init, prob)
    else
        state
    end

    for t in 1:max_iter
        info = (iteration=t,)

        state, terminate, info′ = AdvancedVI.step(
            rng, algorithm, state, callback, objargs...; kwargs...
        )
        info = merge(info′, info)

        if terminate || !isfinite(info′.elbo)
            throw(ErrorException("The ELBO is not finite"))
        end

        AdvancedVI.pm_next!(progress, info)
        push!(info_total, info)
    end
    out = AdvancedVI.output(algorithm, state)
    return out, map(identity, info_total), state
end
