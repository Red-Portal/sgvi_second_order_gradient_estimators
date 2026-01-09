
using LogDensityProblems

struct CapabilityAdjusted{P,C}
    prob::P
    cap::C
end

function LogDensityProblems.logdensity(prob::CapabilityAdjusted, θ)
    return LogDensityProblems.logdensity(prob.prob, θ)
end

function LogDensityProblems.logdensity(prob::CapabilityAdjusted, θ)
    return LogDensityProblems.logdensity(prob.prob, θ)
end

function LogDensityProblems.logdensity_and_gradient(prob::CapabilityAdjusted, θ)
    return LogDensityProblems.logdensity_and_gradient(prob.prob, θ)
end

function LogDensityProblems.logdensity_gradient_and_hessian(prob::CapabilityAdjusted, θ)
    return LogDensityProblems.logdensity_gradient_and_hessian(prob.prob, θ)
end

function LogDensityProblems.dimension(prob::CapabilityAdjusted)
    return LogDensityProblems.dimension(prob.prob)
end

function LogDensityProblems.capabilities(::Type{<:CapabilityAdjusted{P,C}}) where {P,C}
    return C()
end
