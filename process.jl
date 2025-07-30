using DataFrames
using DataFramesMeta
using Pipe: @pipe
using JLD2
using FileIO
using ProgressMeter
using Parquet
using ArgParse
import Statistics: mean, median, quantile
import Base: zero
include("popinterface.jl")
#include("popsim.jl")
zero(::Type{Union{Nothing,Int}}) =  0
mean(x::Vector{Union{Nothing,Int}}) =  isempty(x) ? NaN : mean(convert(Vector{Int},x))
median(x::Vector{Union{Nothing,Int}}) =  isempty(x) ? NaN : median(convert(Vector{Int},x))
quantile(x::Vector{Union{Nothing,Int}}, p) = isempty(x) ? NaN : quantile(convert(Vector{Int},x),p)
mean_skipn(x) = mean(filter(!isnothing,x))
median_skipn(x) = median(filter(!isnothing,x))
qlow_skipn(x) = quantile(filter(!isnothing,x),0.025)
qup_skipn(x) = quantile(filter(!isnothing,x),0.975)
    
##

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--cores", "-c"
            help = "cores per run"
            arg_type = Int
            default = 1
        "--slurm"
            help = "use Slurm Manager flag"
            action = :store_true
        "--reps",  "-r"
            help = "number of replicates"
            arg_type = Int
            required = true
        "--d_size",  "-n"
            help = "deme size"
            arg_type = Int
            required = true
        "--ndemes",  "-d"
            help = "number of demes"
            arg_type = Int
            required = true
        "--Nm"
            help = "migration rate per deme"
            arg_type = Float64
        "--Nm_is_log"
            help = "use logarithmic scale for migration rate"
            action = :store_true
        "--Nmmin"
            help = "migration rate per deme minimum"
            arg_type = Float64
        "--Nmmax"
            help = "migration rate per deme maximum"
            arg_type = Float64
        "--Nm_grid_size"
            help = "migration rate grid size"
            arg_type = Int
        "--benefit",  "-B"
            help = "cooperation fitness benefit"
            arg_type = Float64
        "--bmin"
            help = "cooperation fitness benefit minimum"
            arg_type = Float64
        "--bmax"
            help = "cooperation fitness benefit maximum"
            arg_type = Float64
        "--B_grid_size"
            help = "cooperation fitness benefit grid size"
            arg_type = Int
        "--cost",  "-C"
            help = "cooperation fitness cost"
            arg_type = Float64
        "--cmin"
            help = "cooperation fitness cost minimum"
            arg_type = Float64
        "--cmax"
            help = "cooperation fitness cost maximum"
            arg_type = Float64
        "--C_grid_size"
            help = "cooperation fitness cost grid size"
            arg_type = Int
        "--output", "-o"
            help = "output path"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    slurm = parsed_args["slurm"]
    C = get(parsed_args, "cost", nothing)
    cmin = get(parsed_args, "cmin", nothing)
    cmax = get(parsed_args, "cmax", nothing)
    C_grid_size = get(parsed_args, "C_grid_size", nothing)
    B = get(parsed_args, "benefit", nothing)
    bmin = get(parsed_args, "bmin", nothing)
    bmax = get(parsed_args, "bmax", nothing)
    B_grid_size = get(parsed_args, "B_grid_size", nothing)
    Nm = get(parsed_args, "Nm", nothing)
    Nmmin = get(parsed_args, "Nmmin", nothing)
    Nmmax = get(parsed_args, "Nmmax", nothing)
    Nm_grid_size = get(parsed_args, "Nm_grid_size", nothing)

    # Mutually exclusive checks
    if !xor(!isnothing(C) , (!isnothing(cmin) || !isnothing(cmax) || !isnothing(C_grid_size)))
        error("Arguments --cost and (--cmin, --cmax, --C_grid_size) are mutually exclusive.")
    end
    if !xor(!isnothing(B) , (!isnothing(bmin) || !isnothing(bmax) || !isnothing(B_grid_size)))
        error("Arguments --benefit and (--bmin, --bmax, --B_grid_size) are mutually exclusive.")
    end
    if !xor(!isnothing(Nm) , (!isnothing(Nmmin) || !isnothing(Nmmax) || !isnothing(Nm_grid_size)))
        error("Arguments --Nm and (--Nmmin, --Nmmax, --Nm_grid_size) are mutually exclusive.")
    end

    # slurm related arguments
    if slurm
        @warn "Whatever value cores parameter has is being bypassed by the Slurm job manager."
    end
end

main()

function summarize_data(dataframe,poptype::Type=Island)
    if poptype <: Island
        df = combine(dataframe, 
                 :meanp  => mean => :pmean,
                 :meanp  => median => :pmedian,
                 :meanp  => (x->quantile(x,0.025)) => :plq,
                 :meanp  => (x->quantile(x,0.975)) => :puq,
                 :fst  => mean => :fst_mean,
                 :fst  => median => :fst_median,
                 :fst  => (x->quantile(x,0.025)) => :fstlq,
                 :fst  => (x->quantile(x,0.975)) => :fstuq)
    else
        error("$poptype is not valid type")
    end
    return df
end

function summarize_time_data(dataframe)
    df = combine(dataframe, 
                 :t_fix  => mean_skipn => :t_fix_mean,
                 :t_fix  => median_skipn => :t_fix_median,
                 :t_fix  => qlow_skipn => :tflq,
                 :t_fix  => qup_skipn => :tfuq,
                 :t_ext  => mean_skipn => :t_ext_mean,
                 :t_ext  => median_skipn => :t_ext_median,
                 :t_ext  => qlow_skipn => :telq,
                 :t_ext  => qup_skipn => :teuq)
    return df
end

##
function fill_df!(dataframe, poptype::Type=Island)
    avg_traj_times = sort(unique(dataframe.time)) # union set of time steps of all trajectories
    for r in levels(dataframe.rep)
        rdf = @subset(dataframe, :rep .== r)
        maxt = maximum(rdf.time)
        maxt_df = @subset(rdf, :time .== maxt)
        fill_time = findfirst(x->x==maxt,avg_traj_times)
        if avg_traj_times[fill_time] < avg_traj_times[end] 
            if poptype <: Island
                append!(dataframe,DataFrame(rep=r,time=avg_traj_times[fill_time+1:end],meanp=maxt_df.meanp[1],fst=maxt_df.fst[1]))
            else        
                error("$poptype is not valid type")
            end
        end
    end
end

##
function avg_traj(paramsets, results_path, file_id)
    summary_data = DataFrame()
    avg_trajectory_data = DataFrame()
    @showprogress 1 "progress..." 80 color=:color_normal for (i, v) in enumerate(paramsets)
        maxtime = maximum(v.data.time)
        nreps = v.nreps
        #=postburn = filter(:time => time -> time == maxtime, v.data, view=true)
        gd_postb = groupby(postburn, :time)=#
        fill_df!(v.data,hasfield(typeof(v), :structure) ? v.structure : Island)
        gd = groupby(v.data, :time)
        common_times = @subset(combine(gd, nrow => :count), :count .== nreps).time
        gd = filter(g->g.time[1] in common_times, gd)
        traj = summarize_data(gd,hasfield(typeof(v), :structure) ? v.structure : Island)
        sort!(traj, order(:time))
        dfsummary = @pipe last(traj,1) |> select(_, :pmean, :pmedian, :plq, :puq)
        pnames = [:deme_size, :ndemes, :B, :C, :m, :δ, :μ] 
        #=if v.evolve_func! == life_cycle_ext!
            push!(pnames, :sd)
        end=#
        len = nrow(traj)
        for (ki,k) in enumerate(pnames)
            insertcols!(traj, ki, k=>fill(getfield(v,k),len))
        end
        tempdf = [(k=> getfield(v,k)) for k in pnames] |> DataFrame
        tdfsummary = summarize_time_data(v.tdata)

        append!(avg_trajectory_data, traj)
        append!(summary_data, [tempdf dfsummary tdfsummary])
    end
    #FileIO.save(joinpath(results_path,"$(file_id)_summary_data.jld2"), "summary_data", summary_data; compress = true)
    #jldsave(joinpath(results_path,"$(file_id)_summary_data.jld2"), true; summary_data=summary_data)
    @save joinpath(results_path,"$(file_id)_summary_data.jld2") {compress=false} summary_data
    write_parquet(joinpath(results_path,"$(file_id)_summary_data.parquet"), summary_data)
    #FileIO.save(joinpath(results_path,"$(file_id)_avg_trajectory_data.jld2"), "avg_trajectory_data", avg_trajectory_data; compress = true)
    #jldsave(joinpath(results_path,"$(file_id)_avg_trajectory_data.jld2"), true; avg_trajectory_data=avg_trajectory_data)
    @save joinpath(results_path,"$(file_id)_avg_trajectory_data.jld2") {compress=false} avg_trajectory_data
    write_parquet(joinpath(results_path,"$(file_id)_avg_trajectory_data.parquet"), avg_trajectory_data)
    
    return summary_data, avg_trajectory_data
end

##

function make_file_id(model_tag::AbstractString, N::Int,nd::Int,fa_p0::Union{Float64,Int},nreps::Int,grid1::Union{Int,Nothing}=nothing,grid2::Union{Int,Nothing}=nothing,grid3::Union{Int,Nothing}=nothing; Nm::Union{Float64,Nothing}=nothing, B::Union{Float64,Nothing}=nothing, C::Union{Float64,Nothing}=nothing, sd::Float64=1.0,opt_suffix::AbstractString="", multi::Bool=false, Nm_is_log::Bool=false)
    NT = N*nd
    if round(Int,fa_p0*NT) == 1
        p0tag = "1_$NT"
    elseif round(Int, fa_p0*NT) == N
        p0tag = "1_$nd"
    elseif typeof(fa_p0) <: Int
        p0tag = "$(fa_p0)_$NT"
    else    
        p0tag = "$(round(fa_p0,sigdigits=3))"
    end
    #results_path = joinpath(base_folder,model_tag,"N$(N)_nd$nd")
    file_id = "sims$(model_tag)_"
    file_id *= "N$(N)_nd$(nd)_"
    if occursin("A27",model_tag)
        if multi
            file_id *= "multi_"
        else
            file_id *= "sd$(sd)_"
        end
    end
    if !isnothing(Nm)
        if Nm_is_log
            file_id *= "logNm$(round(Nm,digits=3))_"
        else
            file_id *= "Nm$(round(Nm,digits=3))_"
        end
    end
    if !isnothing(B)
        file_id *= "B$(round(B,digits=3))_"
    end
    if !isnothing(C)
        file_id *= "C$(round(C,digits=3))_"
    end
    n_x = sum( .! isnothing.([grid1, grid2, grid3])) - 1 # number of x (times) signs
    for grid in [grid1, grid2, grid3]
        if !isnothing(grid)
            file_id *= "$(grid)"
            if n_x > 0
                file_id *= "x"
                n_x -= 1
            end
        end
    end
    
    file_id *= "_p0$(p0tag)_$(nreps)reps"
    if opt_suffix != ""
        file_id *="_$(opt_suffix)"
    end
    return file_id
end
