using Distributed
addprocs(4, topology=:master_worker, exeflags="--project=$(Base.active_project())")
@everywhere begin
    using ProbabilisticEchoInversion
    using DataFrames, DataFramesMeta
    using CSV
    using Turing
    using DimensionalData
    using StatsPlots, StatsPlots.PlotMeasures
end

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
          16.3 -2.2 -3.5] # -12.0 is made up
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

# Takes about 3 minutes
solution = apes(echo, echo_model, MAPSolver(), params=params, distributed=true)

krill_post = map(passmissing(s -> s.optimizer.values[2]), solution)
munge_post = map(passmissing(s -> s.optimizer.values[3]), solution)
pollock_post = map(passmissing(s -> s.optimizer.values[4]), solution)

clims = (-5, 2)
plot(
    heatmap(krill_post, yflip=true, clims=clims, colorbar_title="log₁₀(Krill m⁻³)"),
    heatmap(munge_post, yflip=true, clims=clims, colorbar_title="log₁₀(Munge m⁻³)"),
    heatmap(pollock_post, yflip=true, clims=clims, colorbar_title="log₁₀(Pollock m⁻³)"),
    xlabel="Interval", ylabel="Depth (m)", size=(1000, 600), margin=15px
)
savefig(joinpath(@__DIR__, "plots", "posteriors.png"))