
using Bootstrap
using DataFrames
using DataFramesMeta
using HDF5
using JLD2
using Plots, StatsPlots
using Statistics

function statistics(df, group_key, statistic = :elbo)
    @chain groupby(df, group_key) begin
        @combine(
            $"$(statistic)_mean"   = mean($statistic),
            $"$(statistic)_median" = median($statistic),
	    $"$(statistic)_min"    = minimum($statistic),
	    $"$(statistic)_max"    = maximum($statistic),
            $"$(statistic)_boot"   =
                confint(bootstrap(mean, $statistic, BalancedSampling(1024)), BCaConfInt(0.95))
	)
        @transform(
            $"$(statistic)_up" = first.(only.($"$(statistic)_boot")),
            $"$(statistic)_lo" = last.(only.($"$(statistic)_boot"))
        )
    end
end

function plot_curve(df, logstepsize; show_plot=true)
    df = @chain df begin
        @subset(:logstepsize .== logstepsize)
	@select(:elbo, :iteration)
    end
    x        = df[:,:iteration] |> Array{Int}
    df_stats = statistics(df, :iteration, :elbo)
    y        = df_stats[:,Symbol("elbo_mean")]
    y_p      = abs.(df_stats[:,Symbol("elbo_up")] - y)
    y_m      = abs.(df_stats[:,Symbol("elbo_lo")] - y)
    if show_plot
        display(Plots.plot!(x, y, xscale=:log10, ylims=(quantile(y, 0.5), Inf), ribbon=(y_m, y_p)))
    end
    x, y, y_p, y_m
end

function plot_envelope(df, iteration; show_plot=true)
    df = @chain df begin
        @subset(:iteration .== iteration)
	@select(:elbo, :logstepsize)
    end

    x        = df[:,:logstepsize] |> Array{Float64}
    df_stats = statistics(df, :logstepsize, :elbo)
    x        = 10.0.^(df_stats[:,:logstepsize])
    y        = df_stats[:,Symbol("elbo_mean")]
    y_p      = abs.(df_stats[:,Symbol("elbo_up")] - y)
    y_m      = abs.(df_stats[:,Symbol("elbo_lo")] - y)
    if show_plot
        display(Plots.plot!(x, y, xscale=:log10, ylims=(quantile(y, 0.5), Inf), ribbon=(y_m, y_p)))
    end
    x, y, y_p, y_m
end

function export_curves()
    problems     = [
        #"diamonds-diamonds",
        ("dogs-dogs", -4),
        # ("bones_data-bones_model", 2000),
        # ("rats_data-rats_model", 4000),
        # ("surgical_data-surgical_model", 4000),
        # ("GLMM_Poisson_data-GLMM_Poisson_model", 4000),
        # ("nes2000-nes", 4000),
        # ("pilots-pilots", 4000),
        # ("butterfly-multi_occupancy", 4000),
        # ("hudson_lynx_hare-lotka_volterra", 4000),
        # ("loss_curves-losscurve_sislob", 4000),
        # ("rstan_downloads-prophet", 4000),
        # ("gp_pois_regr-gp_pois_regr", 4000),
        # ("bball_drive_event_1-hmm_drive_1", 4000),
    ]

    for (problem, logstepsize) in problems
        h5open("data/pro/curves_$(problem).h5", "w") do h5
            df = JLD2.load("data/raw/$(problem).jld2", "data")

            for order in [1, 2], algorithm in ["WVI", "BBVI"]
                df_sub = @subset(
                    df,
                    :algorithm .== algorithm,
                    :order     .== order,
                    :problem   .== problem
                )
                display(df_sub)
        
                x, y, y_p, y_m = plot_curve(df_sub, logstepsize; show_plot=false)

                write(h5, "x_$(algorithm)_$(order)", x)
                write(h5, "y_$(algorithm)_$(order)", hcat(y, y_p, y_m)' |> Array)
            end
        end
    end
end

function export_envelopes()
    problems     = [
        #"diamonds-diamonds",
        ("dogs-dogs", 1000),
        ("bones_data-bones_model", 2000),
        ("rats_data-rats_model", 4000),
        ("surgical_data-surgical_model", 4000),
        ("GLMM_Poisson_data-GLMM_Poisson_model", 4000),
        ("nes2000-nes", 4000),
        ("pilots-pilots", 4000),
        ("butterfly-multi_occupancy", 4000),
        ("hudson_lynx_hare-lotka_volterra", 4000),
        ("loss_curves-losscurve_sislob", 4000),
        ("rstan_downloads-prophet", 4000),
        ("gp_pois_regr-gp_pois_regr", 4000),
        ("bball_drive_event_1-hmm_drive_1", 4000),
    ]
    make_finite(x) = isfinite(x) ? x : -10e+10

    for (problem, iteration) in problems
        iteration_aligned = iteration + 1

        h5open("data/pro/envelopes_$(problem).h5", "w") do h5
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
