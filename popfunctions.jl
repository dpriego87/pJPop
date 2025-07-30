using LinearAlgebra
using StatsBase
using Random
using Distributions


include("structures.jl")

function mean_genotype(p::Population, locus::Int=1)
    if p.nloci == 1
        mean_G = Vector{Float64}(undef,length(p.params.mut_list))
        for mut_i in eachindex(p.params.mut_list)
            mean_G[mut_i] = count(==(mut_i), p.Gt[:,1]) / p.size
        end
    else
        mean_G = Vector{Float64}(undef,length(p.params.mut_list[locus])) 
        for mut_i in eachindex(p.params.mut_list[locus])
            mean_G[mut_i] = count(==(mut_i), p.Gt[:,locus]) / p.size
        end
    end
    return mean_G
end

# Function to calculate the phenotype of all individuals in a structured population
function calc_pheno!(p::Population{Island})
    for i in 1:p.size
        p.Pt[i] = p.params.δ * (p.params.mut_list[p.Gt[i,p.params.focal_locus]] == 1.0 ? 1.0 : 0.0)
    end
end

# Function to calculate mean phenotype of each deme

function mean_pheno!(p::Population)
    for d in 1:p.ndemes
        p.mean_Pt[d] = mean(p.Pt[p.deme2ind[1,d]:p.deme2ind[p.deme_size,d]])
    end
end

# Payoff function in a helping after reproduction, before dispersal (A-9 in Lehmann & Rousset, 2010 )
function payoff_base(i::Int, d::Int, p::Population{Island})
    # focal phenotype
    zdot = p.Pt[i]
    # deme mean phenotype (without focal individual)
    z0R = p.mean_Pt[d]
    return  1 + p.params.B*z0R - p.params.C*zdot 
end

# Payoff function in a group competition, before dispersal (A-14 in Lehmann & Rousset, 2010 )
function payoff_gc(i::Int, d::Int, p::Population{Island})
    # focal phenotype
    zdot = p.Pt[i]
    # deme mean phenotype (without focal individual)
    z0 = (p.mean_Pt[d]*p.deme_size - zdot) / (p.deme_size - 1)
    return  1 + p.params.B*z0 - p.params.C*zdot 
end

# Payoff function in a helping after reproduction, before dispersal (A-9 in Lehmann & Rousset, 2010 )
function payoff_ext(i::Int, d::Int, p::Population{Island})
    # focal phenotype
    zdot = p.Pt[i]
    return  1 - p.params.C*zdot 
end

function calc_pheno_fitness!(p::Population{Island})
    calc_pheno!(p)
    mean_pheno!(p)
    N = p.deme_size
    p.mean_payoff .= 0.0
    @inbounds for i in 1:p.size
        d = p.ind2deme[i][2]
        p.payoff[i] = p.params.payoff_func(i,d,p) 
        p.mean_payoff[d] += p.payoff[i] / N
    end
    return hcat(p.Pt, p.payoff)
end

function group_competition!(p::Population{Island})
    nd = p.ndemes
    N = p.deme_size
    total_payoff = N * p.mean_payoff
    total_total_payoff = sum(total_payoff)
    if total_total_payoff == 0
        throw(ErrorException("total population payoff is zero"))
    end

    groupcomp = sampler(Categorical(total_payoff / total_total_payoff))
    par_demes = rand(groupcomp, nd)
    par_payoff = copy(p.payoff)
    @inbounds for (od, pd) in enumerate(par_demes)
        o_irange = p.deme2ind[1, od]:p.deme2ind[N, od]
        p_irange = p.deme2ind[1, pd]:p.deme2ind[N, pd]
        @. @views p.payoff[o_irange] = par_payoff[p_irange] / total_payoff[pd] * N
        @. @views p.par_indices[o_irange] = p_irange
    end
    return par_demes
end  

# Environnmental stochasticity, recolonization by migrant propagules
function deme_extinction!(p::Population{Island})
    nd = p.ndemes
    N = p.deme_size
    sd = p.params.sd
    B = p.params.B
    total_total_payoff = sum(N*p.mean_payoff)
    if total_total_payoff == 0
        throw(ErrorException("total population payoff is zero"))
    end
    par_demes = collect(1:nd)
    extinct_demes = rand(sampler(Binomial(1,1-sd)),nd)
    surv_demes = par_demes[extinct_demes .!= 1]
    if isempty(surv_demes)
        surv_demes = rand(par_demes,1)
    end
    deme_benef_surv = 1 .+ B*p.mean_Pt[surv_demes]
    recolonizers = sampler(Categorical(deme_benef_surv/sum(deme_benef_surv)))
    par_demes .= surv_demes[rand(recolonizers, nd)]
    par_payoff = copy(p.payoff)
    @inbounds for (od,pd) in enumerate(par_demes)
        o_irange = p.deme2ind[1,od]:p.deme2ind[N,od]
        p_irange = p.deme2ind[1,pd]:p.deme2ind[N,pd]
        @. p.payoff[o_irange] = par_payoff[p_irange]
        @. p.par_indices[o_irange] = p_irange
    end
    
    return par_demes
end

# Environnmental stochasticity, recolonization by migrant pool
function deme_extinction2!(p::Population{Island})
    nd = p.ndemes
    N = p.deme_size
    sd = p.params.sd
    B = p.params.B
    total_payoff = N*p.mean_payoff
    total_total_payoff = sum(total_payoff)
    if total_total_payoff == 0
        throw(ErrorException("total population payoff is zero"))
    end
    par_demes = collect(1:nd)
    deme_surv_prob = sd*(1 .+ B*p.mean_Pt)
    if !all(i -> 0 <= i <= 1, deme_surv_prob)
        println("Stopping simulation due to an inconsistency...")
        foreach(notp->println("deme_surv_prob[$notp]=",deme_surv_prob[notp]),findall(i-> !(0 <= i <= 1), deme_surv_prob))
        throw(ErrorException("deme survival probability has elements out the range 0 < p < 1 "))
    end
    surv_demes_tags = [rand(sampler(Binomial(1,ps))) for ps in deme_surv_prob]
    if sum(surv_demes_tags) == 0
        #throw(ErrorException("all demes got killed, simulation aborting..."))
        surv_demes_tags[rand(par_demes)] = 1
    end
    @inbounds for (od,deme_survived) in enumerate(surv_demes_tags)
        if deme_survived == 0
            o_irange = p.deme2ind[1,od]:p.deme2ind[N,od]
            @. @views p.payoff[o_irange] = 0.0
            p.mean_payoff[od] = 0.0
            par_demes[od] = 0
        end
    end
    
    return par_demes
end

function wf_migration!(p::Population{Island})
    nd = p.ndemes
    N = p.deme_size
    NT = p.size
    m = p.params.m
    parents = Vector{Int}(undef,NT)
    par_nd = count(x->x>0.0,p.mean_payoff)
    prob_migrant = par_nd > 1.0 ? m / (par_nd-1) : m
    # modify fitness, weighted by migration 
    # total dispersed
    @. p.mfitness = prob_migrant * p.payoff
    @inbounds for d in 1:nd
        # get index range of resident deme d
        irange = p.deme2ind[1,d]:p.deme2ind[N,d]
        #fitness = @view pop.fitness[irange,2]
        # dispersed = nd*m / (nd-1) .* fitness
        # add philopatric component 
        @. @views p.mfitness[irange] = (1-m) * p.payoff[irange]
        # get normalized fertility
        norm_fert = p.mfitness / sum(p.mfitness)
        if !isprobvec(norm_fert)
            @show norm_fert
            throw(ErrorException("norm_fert doesn't add up to 1 as in a probability vector")) 
        end
        # set categorical distribution using normalized fertilities ("sampler" uses "AliasTable")
        fertdist = sampler(Categorical(norm_fert))
        # survival and reproduction
        for d_i in irange
            # individual dies and is replaced by random new born
            parent = p.par_indices[rand(fertdist)]
            parents[d_i] = parent
            @. @views p.Gt[d_i,:] = map((g,l)->p.params.mut_func(g,l),p.Gtm1[parent,:],1:p.nloci) # parent replaces offspring spot
            #p.P[d_i,2] = p.P[parent,1] # parent replaces offspring spot
        end
        # get non-dispersed slice back to normal (replaced by dispersed probability times payoff)
        @. @views p.mfitness[irange] = prob_migrant *  p.payoff[irange]
    end
    return parents
end

function instant_rand_invasion!(p::Population)
    fl = p.params.focal_locus
    p.Gt[:,fl] .= rand(eachindex(p.params.mut_list))
    return 1:p.size
end

function life_cycle_base!(p::Population{Island})
    # update fitness values
    calc_pheno_fitness!(p)

    # save recent population state in "parent" vector
    p.Gtm1 .= p.Gt
    p.Ptm1 .= p.Pt
    p.mean_Ptm1 .= p.mean_Pt
      
    par_demes = collect(1:p.ndemes) # Pool of potential parent demes
    parents = wf_migration!(p) #wf_migration!(p)
    return par_demes, parents
end

function life_cycle_gc!(p::Population{Island})
    # update fitness values
    calc_pheno_fitness!(p)

    # save recent population state in "parent" vector
    p.Gtm1 .= p.Gt
    p.Ptm1 .= p.Pt
    p.mean_Ptm1 .= p.mean_Pt
      
    #set new fitness values according to payoffs after group competition
    par_demes = group_competition!(p) # Pool of potential parent demes
    parents = wf_migration!(p)
    return par_demes, parents
end

function life_cycle_ext!(p::Population{Island})
    # update fitness values
    calc_pheno_fitness!(p)

    # save recent population state in "parent" vector
    p.Gtm1 .= p.Gt
    p.Ptm1 .= p.Pt
    p.mean_Ptm1 .= p.mean_Pt
      
    #set new fitness values according to payoffs after deme extinction
    par_demes = deme_extinction!(p) # Pool of potential parent demes
    parents = wf_migration!(p)
    return par_demes, parents
end

function life_cycle_ext2!(p::Population{Island})
    # update fitness values
    calc_pheno_fitness!(p)

    # save recent population state in "parent" vector
    p.Gtm1 .= p.Gt
    p.Ptm1 .= p.Pt
    p.mean_Ptm1 .= p.mean_Pt
      
    #set new fitness values according to payoffs after deme extinction
    par_demes = deme_extinction2!(p) # Pool of potential parent demes
    parents = wf_migration!(p)
    return par_demes, parents
end

#=function delta_t_record(i::Int,NT::Int)
    return ceil(Int,NT/10*log(0.95+0.1*i))
end =#

# log10 intervals, i.e. 1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,...
function delta_t_record(i::Int)
    return i==0 ? 1 : round(Int,i + 10^floor(log10(i)))
end 

function epoch!(pop::Population)
    pars = pop.params
    stat = pop.status
    stat.step = 0
    fl = pars.focal_locus
    fa = pars.focal_allele
    stat.converged = false
    stat.prog_update = 0
    stat.fa_status = :none
    # let's empty data and tdata dataframes
    delete!(pars.data,1:nrow(pars.data))
    delete!(pars.tdata,1:nrow(pars.tdata))
    #@show pars
    # evolve for nsteps time steps during epoch
    #println("sim $(stat.sim), rep $(stat.rep), t=$(stat.step), nrow(pop.params.tdata)=",nrow(pop.params.tdata))
    next_time_to_rec = 0 # time step to be saved
    if pars.save_at_fixed_rate
        #for i in 1:pars.nsteps
        while !stat.converged
            # save data every rec_steps
            if stat.step % pars.rec_steps == 0
                pars.save_data(pars.data, pop)
                put!(stat.prog_channel, (0, stat.sim, pop.ndemes, pop.deme_size, pars.m*pop.deme_size, pars.δ*pars.B, pars.δ*pars.C, stat.rep, stat.step, sum(pop.Gt[:,fl] .== fa) / pop.size, std(pop.Gt[:,fl]), stat.converged, nrow(pars.data)))
            end
            stat.step += 1
            pars.evolve_func!(pop)
            stat.converged = std(pop.Gt[:,fl]) == 0.0
        end
        next_time_to_rec = stat.step + pars.rec_steps - 1
    else
        #iters_saved = 1
        while !stat.converged
            # save data after each interval dictated by the function delta_t_record
            if stat.step == next_time_to_rec # || stat.step == 1
                #println("sim $(stat.sim), rep $(stat.rep), t=$(stat.step), meanGt=", mean(pop.Gt[:,fl]))
                pars.save_data(pars.data, pop)
                #iters_saved += 1
                next_time_to_rec += delta_t_record(stat.step)
                put!(stat.prog_channel, (0, stat.sim, pop.ndemes, pop.deme_size, pop.deme_size*pars.m, pars.δ*pars.B, pars.δ*pars.C, stat.rep, stat.step, sum(pop.Gt[:,fl] .== fa) / pop.size, std(pop.Gt[:,fl]), stat.converged, nrow(pars.data)))
            end
            stat.step += 1
            try 
                pars.evolve_func!(pop) #@show stat.step, stat.converged, pars.evolve_func!(pop)
            catch lc_err
                println("LIFE CYCLE ERROR:",lc_err)
                rethrow()
            end
            stat.converged = std(pop.Gt[:,fl]) == 0.0
            #println("t=$(stat.step)")
        end
    end
    stat.fa_status = pop.Gt[1,fl] == fa ? :fixed : :lost
    pars.save_tdata(pars.tdata, pop) #  we save convergence result, i.e. fixation or extinction
    # Additional iterations until reaching the next time step to record given by delta_t_record 
    #if stat.step != next_time_to_rec
    stat.step = next_time_to_rec
    #end
    pars.save_data(pars.data, pop) 
    
    # progress update after replicate ends
    stat.prog_update += 1
    #put!(stat.prog_channel, true)
    # progress update for simulation completion (update progress %)
    put!(stat.prog_channel, (1, stat.sim, pop.ndemes, pop.deme_size, pars.m*pop.deme_size, pars.δ*pars.B, pars.δ*pars.C, stat.rep, stat.step, sum(pop.Gt[:,fl] .== fa) / pop.size, std(pop.Gt[:,fl]), stat.converged, nrow(pars.data)))
    #println("sim $(stat.sim), rep $(stat.rep), t=$(stat.step), meanGt=", mean(pop.Gt[1,fl]))
end


# Calculate FST for a multiallelic locus in a haploid population
# having an n-island-model structure
function FST_multi(p::Population,locus::Int=1)
    N = p.deme_size
    nd = p.ndemes
    NT = p.size

    # calculate allele frequency distribution in each deme
    allele_fd = Dict{Int,Array{Int,1}}()
    for d in 1:nd
        for i in 1:N
            allele = p.Gt[p.deme2ind[i,d],locus]
            if allele in keys(allele_fd)
                allele_fd[allele][d] += 1
            else
                allele_fd[allele] = zeros(Int, nd)
                allele_fd[allele][d] = 1
            end
        end
    end

    # calculate mean allele frequencies for each deme
    nalleles = length(keys(allele_fd))
    pm = zeros(nd,nalleles)
    for (a, allele) in enumerate(keys(allele_fd))
        pm[:,a] .= allele_fd[allele] / N
    end
    pmm = mean(pm, dims=1)

    # FST is calculated by homozygosity (G) within (w) and between (b) populations
    # using sampling w/o replacement
    Gw = mean([ sum([ pm[d,a]*(N*pm[d,a] - 1) / (N-1) for a in 1:nalleles ]) for d in 1:nd ])
    Gb = mean([ sum([ pm[d,a]*(nd*pmm[a] - pm[d,a]) / (nd-1) for a in 1:nalleles ]) for d in 1:nd ])

    return ( Gw - Gb ) == 0.0 ? 0.0 : ( Gw - Gb ) / ( 1 - Gb)
end

