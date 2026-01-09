
using DataFrames
using DataFramesMeta
using HDF5
using JLD2
using Plots, StatsPlots
using Statistics

function statistics(df, group_key, statistic = :elbo)
    df = @chain groupby(df, group_key) begin
        @combine(
            $"$(statistic)_mean"   = mean($statistic),
            $"$(statistic)_median" = median($statistic),
	    $"$(statistic)_min"    = minimum($statistic),
	    $"$(statistic)_max"    = maximum($statistic),
	    $"$(statistic)_90"     = quantile($statistic, 0.9),
	    $"$(statistic)_10"     = quantile($statistic, 0.1),
	)
    end
end

function plot_envelope(df, iteration; show_plot=true)
    df = @chain df begin
        @subset(:iteration .== iteration)
	@select(:elbo, :logstepsize)
    end

    x = df[:,:logstepsize] |> Array{Float64}
    y = df[:,:elbo]    |> Array{Float64}

    df_stats = statistics(df, :logstepsize, :elbo)
    x   = 10.0.^(df_stats[:,:logstepsize])
    y   = df_stats[:,Symbol("elbo_median")]
    y_p = abs.(df_stats[:,Symbol("elbo_90")] - y)
    y_m = abs.(df_stats[:,Symbol("elbo_10")] - y)
    if show_plot
        display(Plots.plot!(x, y, xscale=:log10, ylims=(quantile(y, 0.5), Inf), ribbon=(y_m, y_p)))
    end
    x, y, y_p, y_m
end

function export_envelopes()
    problems     = [
        #"diamonds-diamonds",
        ("dogs-dogs", 2000),
        #"gp_pois_regr-gp_pois_regr",
        #"radon_mn-radon_hierarchical_intercept_centered",
    ]
    make_finite(x) = isfinite(x) ? x : -10e+10

    for (problem, iteration) in problems
        iteration_aligned = iteration + 1

        h5open("data/pro/$(problem).h5", "w") do h5
            df = JLD2.load("data/raw/$(problem).jld2", "data")
            df = @transform(df, :elbo = make_finite.(:elbo))

            for order in [1, 2], algorithm in ["WVI", "BBVI"]
                df_sub = @subset(
                    df,
                    :algorithm .== algorithm,
                    :order     .== order,
                    :problem   .== problem
                )
                display(df_sub)
        
                x, y, y_p, y_m = plot_envelope(df_sub, iteration_aligned; show_plot=false)

                write(h5, "x_$(algorithm)_$(order)", x)
                write(h5, "y_$(algorithm)_$(order)", hcat(y, y_p, y_m)' |> Array)
            end
        end
    end
end

function main()
    problems     = [
        #"diamonds-diamonds",
        "dogs-dogs",
        #"gp_pois_regr-gp_pois_regr",
        #"radon_mn-radon_hierarchical_intercept_centered",
    ]
    problem = only(problems)

    df = JLD2.load("data/raw/$(problem).jld2", "data")

    make_finite(x) = isfinite(x) ? x : -10e+10
    df   = @transform(df, :elbo = make_finite.(:elbo))

    iteration = 1000 - 99

    df_sub = @subset(
        df,
        :algorithm .== "WVI",
        :order     .== 1,
        :problem   .== problem
    )
    display(df_sub)
    plot_envelope(df_sub, iteration)
end
