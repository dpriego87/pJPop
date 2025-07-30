using ArgParse
using Distributed
using JLD2
using CodecBzip2
using StatsBase: midpoints
include("formulas.jl")
include("process.jl")

parsed_args = parse_commandline()
println("Parsed args:")
for (arg,val) in parsed_args
    println("  $arg  =>  $val")
end
   
if parsed_args["slurm"]
    using SlurmClusterManager
    addprocs(SlurmManager())
else
    using ClusterManagers
    nproc = parsed_args["cores"]
    addprocs(nproc)
end
    
@everywhere begin
    using Revise
    using LinearAlgebra
    using DataFrames
    includet("popsim.jl")
    BLAS.set_num_threads(1)
end

root_dir = parsed_args["output"]
if !isdir(root_dir)
    mkpath(root_dir)
end
#root_dir = "paral_metapop"
#root_dir = "/better_scratch/dpriego/paral_metapop"
#root_dir = "$(ENV["SCRATCH"])/sims/paral_metapop"
#root_dir = "/scratch/dapr234/paral_metapop"


##########
# FIXED PARAMETERS
fa = 2 # index of focal allele
p0 = 0.5 # initial frequency of focal allele
init_freqs = [1-p0, p0] # initial frequencies of alleles
nd_array = [50,100,150,200]
nreps_array = [3000]
nreps = parsed_args["reps"]
N = parsed_args["d_size"]
nd = parsed_args["ndemes"]
mutant_copy_number = 1
B = parsed_args["benefit"] # cooperation fitness benefit
B_grid_size = parsed_args["B_grid_size"] # number of cooperation benefits to test
bmin = parsed_args["bmin"] # cooperation fitness benefit minimum
bmax = parsed_args["bmax"] # cooperation fitness benefit maximum
C = parsed_args["cost"] # cooperation cost
C_grid_size = parsed_args["C_grid_size"] # number of cooperation costs to test
cmin = parsed_args["cmin"] # cooperation cost minimum
cmax = parsed_args["cmax"] # cooperation cost maximum
Nm = parsed_args["Nm"] # migration rate per deme
Nm_grid_size = parsed_args["Nm_grid_size"] # number of migration rates to test
Nmmin = parsed_args["Nmmin"] # migration rate per deme minimum
Nmmax = parsed_args["Nmmax"] # migration rate per deme maximum
δ = 1/(N*nd)
μ = 0.0
grid_size = 20


Neff_expr = quote 
    function (p)
        n = p[:deme_size]
        nd = p[:ndemes]
        m = p[:m]
        return Neff_b(n,m,nd)
    end
end

seff_expr = quote 
    function (p)
        n = p[:deme_size]
        nd = p[:ndemes]
        m = p[:m]
        B = p[:B]
        C = p[:C]
        δ = p[:δ]
        return δ*SIF_A9(B,C,n,m,nd)
    end
end

    
    save_data = (data, pop) -> begin
        p_mutant = mean_genotype(pop)[pop.params.focal_allele]
        #println("p($(pop.status.step))=$(p_mutant) has been saved.")
        push!(data, (pop.status.rep, pop.status.step, p_mutant, FST_multi(pop)))
    end 


save_tdata = (tdata, pop) -> begin
        #println("Absorption time for population $(pop.status.rep) was $(pop.status.step).")
        if pop.status.fa_status == :fixed
            status = "fixed"
        elseif pop.status.fa_status == :lost
            status = "lost"    
        else
            status = nothing    
    end
    push!(tdata, (pop.status.rep, pop.status.converged ? pop.status.step : nothing, status))
end 

grid_edges = vcat([0.01],@. round(0.011*exp(0.2726 *(1:14)),digits=3))

# Helping after reproduction before dispersal
#for (ndi,nd) in enumerate(nd_array) 
    
    base_params = Dict(:structure=>Island, :evolve_func! => life_cycle_base!, :payoff_func => payoff_base,
                  :data=>DataFrame(rep=Int[], time=Int[], meanp=Float64[], fst=Float64[]),
                  :tdata=>DataFrame(rep=Int[], t_abs=Union{Nothing,Int}[],abs_status=Union{Nothing,String}[]))
    
    push!(base_params,:ndemes=>nd, :deme_size=>N, 
                  :nreps=>nreps, :focal_allele => fa, :save_at_fixed_rate => false,
                  :rec_steps=>10, :μ => μ, :δ => δ, 
                  :save_data=>save_data, :save_tdata=>save_tdata)
    #base_params[:init_freqs] = [base_params[:ndemes]*base_params[:deme_size]-mutant_copy_number,mutant_copy_number]
    base_params[:init_freqs] = init_freqs
    sweep_params = Dict{Symbol,Any}()
    Nm_is_log = parsed_args["Nm_is_log"]
    if !isnothing(Nm)
        base_params[:m] = Nm / N
    else
        m_range = range(Nmmin, Nmmax, length=Nm_grid_size)
        if Nm_is_log
            m_range = 10 .^ m_range
        end
        m_range = m_range ./ N
        sweep_params[:m] = m_range
    end
    
    if !isnothing(B)
        base_params[:B] = B
    else
        B_range = range(bmin, bmax, length=B_grid_size)
        sweep_params[:B] = B_range
    end

    if !isnothing(C)
        base_params[:C] = C
    else
        C_range = range(cmin, cmax, length=C_grid_size)
        sweep_params[:C] = C_range
    end
    
    paramsets = build_param_sweep(base_params, sweep_params)
    model_tag = "A9"
    println("N=",N,", nd=",nd)
    file_id = make_file_id(model_tag, base_params[:deme_size], base_params[:ndemes], 
                    base_params[:init_freqs][fa],base_params[:nreps], Nm_grid_size, B_grid_size, C_grid_size, Nm=Nm, B=B, C=C, Nm_is_log=Nm_is_log)
    full_path = joinpath(root_dir,"$(file_id).jld2")
    println("\nLooking for existing simulation files $(full_path)... ")
    if isfile(full_path)
        println("Already existing: $(full_path)")
    else
        #=if !ispath(results_path)
            mkpath(results_path)
        end=#
        evolve_runs!(paramsets)
        println("\nSaving results at ")
        print(full_path)
        #@save joinpath(results_path,"$(file_id).jld2") {compress=true} paramsets
        jldopen(full_path, "w"; compress = Bzip2Compressor()) do f
            f["paramsets"] = paramsets;
        end
    end
#end
