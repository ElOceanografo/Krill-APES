using Distributed
addprocs(4, topology=:master_worker, exeflags="--project=$(Base.active_project())")
@everywhere begin
    using ProbabilisticEchoInversion
    using Turing
    using DimensionalData
end
using DataFrames, DataFramesMeta
using CSV
using StatsPlots, StatsPlots.PlotMeasures
using ColorSchemes
using LinearAlgebra

#=
Read in files and format as 3D DimArray (Layer x Interval x freq)
=#
intervals = CSV.read(joinpath(@__DIR__, "data", "v157-s202207-t008-f38-min_threshold-0 (intervals).csv"), 
    DataFrame)
intervals = @select(intervals, :Interval, :VL_start)

files = readdir(joinpath(@__DIR__, "data"))
files = filter(contains("(cells)"), files)
echo_df = map(files) do file
    freq = parse(Int, split(file, "-")[4][2:end])
    df = CSV.read(joinpath(@__DIR__, "data", file), DataFrame)
    df.freq .= freq
    df
end
echo_df = @chain vcat(echo_df...) begin
    @select(:Interval, :Layer, :freq, :Sv_mean)
    @subset(:Layer .<= 31)
    @orderby(:Interval, :Layer, :freq)
end

grid = allcombinations(DataFrame, 
    "Interval" => unique(echo_df.Interval),
    "Layer" => unique(echo_df.Layer),
    "freq" => unique(echo_df.freq))
echo_df = leftjoin(grid, echo_df, on=[:Interval, :Layer, :freq])
echo_df = @chain echo_df begin
    rightjoin(intervals, on=:Interval)
    @rename(:distance = :VL_start)
    @transform(:depth = (:Layer .- 1) .* 5)
end

echo = unstack_echogram(echo_df, :Interval, :depth, :freq, :Sv_mean)
freqs = sort(unique(echo_df.freq))

echograms = map(freqs) do f 
    p = heatmap(echo[F(At(f))], yflip=true, xlabel="Interval", ylabel="Depth (m)")
end
plot(echograms..., size=(1000, 600), margin=15px)
savefig(joinpath(@__DIR__, "plots", "echograms.png"))

#=
Define TS spectra and model
=#

# rows: 18, 38, 120, 200, cols: krill, munge, pollock
ΔTS38 = [-10.0  8.2  0.4;
           0.0  0.0  0.0;
          13.8 -2.1 -1.6;
          16.3 -2.2 -3.5]
TS38 = [-90 -50 -35]
TS = TS38 .+ ΔTS38
Σ = exp10.(TS ./ 10)
plot(freqs, TS, marker=:o, label=["Krill" "High-18 munge" "Pollock"],
    xlabel="Frequency (kHz)", ylabel="TS (dB re m⁻²)")

@everywhere begin
    @model function echo_model(data, params)
        ϵ ~ Exponential(0.5)
        Σ = params.Σ
        logn ~ arraydist(params.prior)
        n = exp10.(logn)
        Sv_pred = 10log10.(Σ * n)
        for i in eachindex(data.freqs)
            if ! ismissing(data.backscatter[i])
                data.backscatter[i] ~ Normal(Sv_pred[i], ϵ)
            end
        end
        return Sv_pred
    end
end

lognprior = [
    Normal(-1, 3), # Krill
    Normal(-1, 3), # Munge
    Normal(-3, 3)  # Pollock
]
params = (Σ=Σ, prior = lognprior)

# Takes about 4 minutes
solution = apes(echo, echo_model, MAPSolver(), params=params, distributed=true)

krill_post = map(passmissing(s -> s.optimizer.values[2]), solution)
munge_post = map(passmissing(s -> s.optimizer.values[3]), solution)
pollock_post = map(passmissing(s -> s.optimizer.values[4]), solution)

krill_var = map(passmissing(s -> max(eps(), diag(cov(s))[2])), solution)
munge_var = map(passmissing(s -> max(eps(), diag(cov(s))[3])), solution)
pollock_var = map(passmissing(s -> max(eps(), diag(cov(s))[4])), solution)

"""
Calculate coefficient of variation for lognormal distribution with parameters μ and σ².
"""
lognormalcv(μ, σ²) = std(LogNormal(μ, sqrt(σ²))) / mean(LogNormal(μ, sqrt(σ²)))
krill_cv = passmissing(lognormalcv).(krill_post, krill_var)
munge_cv = passmissing(lognormalcv).(munge_post, munge_var)
pollock_cv = passmissing(lognormalcv).(pollock_post, pollock_var)

clims = (-4, 1)
p_mean = plot(
    heatmap(krill_post, yflip=true, clims=clims, colorbar_title="log₁₀(Krill m⁻³)"),
    heatmap(munge_post, yflip=true, clims=clims, colorbar_title="log₁₀(Munge m⁻³)"),
    heatmap(pollock_post, yflip=true, clims=clims, colorbar_title="log₁₀(Pollock m⁻³)"),
    xlabel="Interval", ylabel="Depth (m)", layout=(3, 1), xlims=[7000, 9000]
)

p_cv = plot(
    heatmap(krill_cv, clims=(0, 5), yflip=true, c=:viridis, colorbar_title="Krill CV"),
    heatmap(munge_cv, clims=(0, 5), yflip=true, c=:viridis, colorbar_title="Munge CV"),
    heatmap(pollock_cv, clims=(0, 5), yflip=true, c=:viridis, colorbar_title="Pollock CV"),
    xlabel="Interval", ylabel="Depth (m)", layout=(3, 1), xlims=[7000, 9000]
)
plot(p_mean, p_cv, size=(1000, 600), leftmargin=10px)
savefig(joinpath(@__DIR__, "plots", "posteriors.png"))

function high_confidence(x, cv; thresh=1.0, fill=0)
    x1 = deepcopy(x)
    for (i, c) in enumerate(cv)
        if ! ismissing(c)
            x1[i] = c > thresh ? fill : x1[i]
        end
    end
    return x1
end
krill_good = high_confidence(krill_post, krill_cv, fill=-999)
pollock_good = high_confidence(pollock_post, pollock_cv, fill=-999)
plot(
    heatmap(krill_good, yflip=true, clim=(-4, 2)),
    heatmap(pollock_good, yflip=true, clim=(-4, 2)),
    layout=(2, 1), size=(1000, 600), margin=15px
)
