using Distributed
using Revise
using ProgressMeter
includet("popinterface.jl")
includet("popfunctions.jl")

function evolve_worker!(run_info, pop::Population)#, results, free_workers)
    sim, rep, r = run_info
    pop.status.sim = sim
    pop.status.rep = rep
    # run replicate
    epoch!(pop)
    #=put!(results, (sim, rep, pop.params.data, pop.params.tdata))
    # tell main process worker is free again
    put!(free_workers, myid())
    # GC.gc(true)=#
    # send results to main process
    return (sim, rep, pop.params.data, pop.params.tdata)
end


function evolve_runs!(params::Union{Params,Array{Params,1}})
    params isa Params ? params = [params] : nothing # make sure params is Array
    accum_runs = accumulate(+, [p.nreps for p in params])
    nruns = accum_runs[end]
    println("Nruns=$nruns")
    accum_runs .-= accum_runs[1]
    
    # variables to track workers, worker/main communication, progress meter, and tasks done
    worker_pool = WorkerPool(workers())
    # channels to track free workers, results, worker/main communication, and progress meter
    # runs = RemoteChannel(()->Channel{Tuple{Int,Int,Int}}(nruns))
    #= free_workers = RemoteChannel(()->Channel{Int}(nworkers()))
    results = RemoteChannel(()->Channel{Any}(nruns))=#
    #workers_to_main = RemoteChannel(()->Channel{Any}(nworkers()))
    #main_to_workers = Dict([w=>RemoteChannel(()->Channel{Any}(1)) for w in workers()]...)
    prog_channel = RemoteChannel(()->Channel{Any}(typemax(Int32)))
    resolved_tasks = 0
    #= # load worker_ids to free_workers channel
    for w in workers()
        put!(free_workers, w)
    end=#
    @sync begin
        # setup and run distributed progress meter
        progress_meter = Progress(nruns, desc="run...", barlen=80, color=:color_normal)
        running_progress = true
        @async while running_progress !== false
            running_progress = take!(prog_channel)

            if running_progress !== false
                prog_step, sim, nd, N, Nm, δB, δC, rep, step, mean_p, var_p, absorbed, df_size = running_progress

                next!(progress_meter, step = prog_step, valuecolor=:color_normal,
                    showvalues = [(:sim, sim), (:ndemes, nd), (:deme_size, N), (:Nm, Nm), (:δB, δB), (:δC, δC), (:rep, rep), (:time, step), (:mean_p, mean_p), (:var_p, var_p),  (:absorbed, absorbed), (:time_points_saved, df_size)])
            end
        end

        # call workers to run simulations
        @async for (sim, p) in enumerate(params)
            for rep in 1:p.nreps
                r = accum_runs[sim] + rep
                # call worker on run
                w = take!(worker_pool)
                @async begin
                    try
                        # take free worker from pool
                        #println("worker pool size=",length(worker_pool))
                        
                        # run task on worker
                        #=task = remotecall(
                            evolve_worker!, w, (sim, rep, r), 
                            Population(Params(p),
                                Status(prog_channel=prog_channel)))
                        
                        #display(worker_pool.channel) 
                        # collect results
                    
                        (sim, rep, data, tdata) = fetch(task) =#
                        (sim, rep, data, tdata) = remotecall_fetch(evolve_worker!, w, (sim, rep, r), 
                                                            Population(Params(p),
                                                            Status(prog_channel=prog_channel)))
                        #println("worker=$w, run=$r, resolved_tasks=$(resolved_tasks+1), sim=$sim, rep=$rep, data=$(data[nrow(data),:meanp]), tdata=$(tdata[nrow(tdata),:t_fix]), nrow(params[$sim].data)=$(nrow(params[sim].data))")
                        append!(params[sim].data, data)
                        append!(params[sim].tdata, tdata)
                    # catch exception on worker
                    catch err
                        # print exception now
                        #println("ERROR: worker $w: ", err.captured.ex)
                        println("ERROR: worker $w: ", err)
                        #println(err)
                        #interrupt()
                        # rethrow except so we see it when workers are done
                        rethrow()
                    finally
                        # track number of resolved tasks
                        resolved_tasks += 1
                        #println("worker=$w, run=$r, resolved_tasks=$resolved_tasks, sim=$sim, rep=$rep, data=$(data[nrow(data),:meanp]), tdata=$(tdata[nrow(tdata),:t_fix]), nrow(params[$sim].data)=$(nrow(params[sim].data))")

                        # worker is free once task is done so put it back into worker pool
                        put!(worker_pool, w)
                    end
                end
            end
        end
        #=
        @async for (sim, p) in enumerate(params)
            for rep in 1:p.nreps
                r = accum_runs[sim] + rep
                #print("Replicate $rep in param set $sim will run on")
                # take free worker_id
                w = take!(free_workers)
                #println(" worker $w.")
                # call worker on run
                remotecall(evolve_worker!, w, (sim, rep, r), 
                        Population(Params(p),
                                    Status(prog_channel=prog_channel)),
                        results, free_workers)
            end
        end
        =#
        #println("Collecting results...")
        # track number of tasks resolved and finish loops when all tasks resolve
        @async begin
            while resolved_tasks < nruns
                sleep(0.25)
            end

            # end progress meter and convergence test loops
            put!(prog_channel, false)
        end     
        # collect results    
        #=for r in 1:nruns
            (sim, rep, data, tdata) = take!(results)
            append!(params[sim].data, data)
            append!(params[sim].tdata, tdata)
            #println("Post-append: sim $sim, rep $rep, nrow(data) = ",nrow(data)," nrow(params[$sim].data) = ",nrow(params[sim].data),", nrow(tdata) = ",nrow(tdata)," nrow(params[$sim].tdata) = ",nrow(params[sim].tdata))
        end
        
        # end progress meter
        put!(prog_channel, false)=#
    end
end

"""
base_params: (key,value) for parameters that don't vary during the sweep
sweep_params: (key,value) for parameters that vary during the sweep. 
key: Symbol with parameter name; e.g., :nsites
value: single value or array of values. 
each value can be scalar or function.
functions are specified as e.g., 1.0 or (x)->x[:nsites]*100 
where :nsites must be either in base_params or sweep_params
"""
function build_param_sweep(base_params::Dict{Symbol}, sweep_params::Dict{Symbol})
    base_keys  = keys(base_params)
    sweep_keys = keys(sweep_params)
    all_keys   = [ collect(base_keys); collect(sweep_keys) ]
    base_values  = collect(values(base_params))
    sweep_combos = Iterators.product(values(sweep_params)...)

    paramsets = Params[]
    for sweep_values in sweep_combos
        parset = Dict(zip(all_keys, [base_values..., sweep_values...]))
        
        # if param value is a function of other fixed params (specified using an Expr), 
        # evaluate function and save param value
        for (k,v) in parset
            if v isa Expr
                parset[k] = Base.invokelatest(eval(v), parset)
            end
        end

        push!(paramsets, Params(;parset...))
    end

    return paramsets
end
