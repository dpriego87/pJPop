using Parameters
import Random: AbstractRNG
using Distributed


# round to nearest integer randomly
function rround(x)
    return floor(Int,x) + (rand() < x - floor(Int,x))
end

# Population parameters data type
@with_kw struct Params
    structure::Type = Island
    ndemes::Int = 100
    deme_size::Int = 10
    nloci::Int = 1
    focal_locus::Int = 1
    focal_allele::Int = 2
    init_freqs::Union{Vector{Float64},Vector{Vector{Float64}},Vector{Int},Vector{Vector{Int}}} = [0.5,0.5]
    B::Float64 = 1.0
    C::Float64 = 0.1
    δ::Float64 = 0.01
    m::Float64 = 0.01
    μ::Float64 = 0.01
    sd::Float64 = 0.9 
    #Ne::Float64 = ndemes*deme_size
    #se::Float64 = 0.0
    evolve_func!::Function
    payoff_func::Function
    mut_list::Union{Vector{Float64},Vector{Vector{Float64}}} = [0.0,1.0]     # List of allele repertoire (possible mutations)
    mut_mod::Int = 1
    mut_func::Function = (x::UInt,locus::Int) -> x
    nsteps::Int = 1
    nepochs::Int = 1
    nreps::Int = 1
    rec_steps::Int = 1
    save_at_fixed_rate::Bool = true
    save_data::Function = (data,pop)->()
    save_tdata::Function = (tdata,pop)->()
    data = []
    tdata = []

    # inner constructor just makes sure data field is a COPY so that copied Params do not
    # link to identical data structures
    function Params(structure::Type,ndemes::Int, deme_size::Int, nloci::Int, focal_locus::Int, focal_allele::Int, init_freqs, B::Float64, C::Float64, 
        δ::Float64, m::Float64, μ::Float64, sd::Float64, evolve_func!::Function, payoff_func::Function, mut_list, mut_mod::Int,
            mut_func::Function, nsteps::Int, nepochs::Int, nreps::Int, rec_steps::Int, save_at_fixed_rate::Bool, save_data::Function, save_tdata::Function, data, tdata)
        if typeof(init_freqs) <: Vector{Vector{Float64}} || typeof(init_freqs) <: Vector{Vector{Int}}
            nloci = length(init_freqs)
            if nloci == 1
                init_freqs = init_freqs[1]
            end
        else
            nloci = 1
        end
        if nloci == 1
        end
        err_coordinate1 = 1<= focal_locus <= nloci
        err_coordinate1 || error("First part of coordinate of focal allele does not match number of loci specified by init_freqs")
        if nloci == 1
            nvals = length(init_freqs)
        else
            nvals = length(init_freqs[focal_locus])
        end
        err_coordinate2 = 1 <= focal_allele <= nvals
        err_coordinate2 || error("Coordinate of focal allele does not match number of alleles specified by init_freqs")
        if nloci == 1
            if typeof(init_freqs) <: Vector{Float64} 
                isprobvec(init_freqs) || throw(DomainError(init_freqs,"initial allelic frequencies vector is not a probability vector"))
            else
                sum(init_freqs) == ndemes * deme_size || throw(DomainError(init_freqs,"initial allelic counts vector must sum to the census size"))
            end
            mut_list = collect(range(0,1,length=nvals))
            
        else
            mut_list = Vector{Vector{Float64}}(undef,nloci)
            for l in eachindex(mut_list)
                if typeof(init_freqs[l]) <: Vector{Float64} 
                    isprobvec(init_freqs[l]) || throw(DomainError(init_freqs[l],"entry $l of initial allelic frequencies vector is not a probability vector"))
                else
                    sum(init_freqs[l]) == ndemes * deme_size || throw(DomainError(init_freqs[l],"initial allelic counts vector must sum to the census size"))
                end
                nvals = length(init_freqs[l])
                mut_list[l] .= collect(range(0,1,length=nvals))
            end
        end
        # Initialize individuals
        if μ > 0.0
            if mut_mod == 1
                if nloci == 1
                    mut_func = (x::UInt,locus::Int=1) -> rand() < μ ? rand(eachindex(mut_list)[1:end .!= x]) : x
                else
                    mut_func = (x::UInt,locus::Int) -> rand() < μ ? rand(eachindex(mut_list[locus])[1:end .!= x]) : x
                end        
            elseif mut_mod == 2
                if nloci == 1
                     mut_func = function(x::UInt,locus::Int)
                        if rand()  < μ
                            mut = rand() 
                            sampled_mut = rand([mut_list; mut])
                            if !(sampled_mut in mut_list)
                                push!(mut_list, mut)
                            end
                            return findfirst(isequal(sampled_mut),mut_list)
                        else
                            return x
                        end
                    end
                else
                    mut_func = function(x::UInt,locus::Int)
                        if rand()  < μ
                            mut = rand() 
                            sampled_mut = rand([mut_list[locus]; mut])
                            if !(sampled_mut in mut_list[locus])
                                push!(mut_list[locus], mut)
                            end
                            return findfirst(isequal(sampled_mut),mut_list[locus])
                        else
                            return x
                        end
                    end
                end
            else
                throw(DomainError(mut_mod,"mutation mode argument must be 1 or 2"))
            end
        end
        return new(structure,ndemes, deme_size, nloci, focal_locus, focal_allele, init_freqs, B, C, δ, m, μ, sd, evolve_func!, payoff_func, mut_list, mut_mod,
            mut_func, nsteps, nepochs, nreps, rec_steps, save_at_fixed_rate, save_data, save_tdata, copy(data), copy(tdata))
    end
end

# struct to hold time step, epoch #, replicate #, sim #, and progress for a population
@with_kw mutable struct Status
    step::Int   = 0
    epoch::Int  = 0
    rep::Int    = 0
    sim::Int    = 0
    prog_update::Int = 0
    fa_status::Symbol = :none
    converged::Bool = false
    prog_channel::RemoteChannel{Channel{Any}} = RemoteChannel(()->Channel{Any}(0))
end

# individual data type
struct Individual
    loci::Vector{Float64}
end

# population data type
struct Population{S<:StrType}
    size::Int
    ndemes::Int
    deme_size::Int
    nloci::Int
    deme2ind::LinearIndices{2}
    ind2deme::Array{CartesianIndex{2},2}
    #individuals::Vector{Individual}
    Gt::Matrix{UInt}              # Genotype indices in the current generation (G(t))
    Gtm1::Matrix{UInt}            # Genotype indices in the previous generation (G(t-1))
    Pt::Vector{Float64}           # Phenotype values in the current generation
    Ptm1::Vector{Float64}         # Phenotype values in the previous generation
    mean_Pt::Vector{Float64}      # Mean phenotype, deme-wise in the current generation
    mean_Ptm1::Vector{Float64}    # Mean phenotype, deme-wise in the previous generation
    payoff::Vector{Float64}       # Payoffs
    mean_payoff::Vector{Float64}  # Mean payoff, deme-wise
    mfitness::Vector{Float64}     # Auxilliary array to calculate migrant and resident Fitness values 
    par_indices::Vector{Int}      # Vector to track the indices of parents
    
    params::Params
    status::Status

    function Population(params::Params, status=Status())
        S = params.structure
        ndemes = params.ndemes
        deme_size = params.deme_size
        pop_size = ndemes * deme_size
        #individuals = Vector{Individual}(undef, pop_size)
        nloci = params.nloci
        # Store individuals in Vector and use site2ind and ind2site to map
        # from (site,individual) to vector index and vice versa
        deme2ind = LinearIndices((deme_size, ndemes))
        ind2deme = CartesianIndices((deme_size, ndemes))
        Gt = Matrix{UInt}(undef,pop_size,nloci)
        Gtm1 = Matrix{UInt}(undef,pop_size,nloci)
        Pt = Vector{Float64}(undef,pop_size)
        Ptm1 = Vector{Float64}(undef,pop_size)
        mean_Pt = Vector{Float64}(undef,ndemes)
        mean_Ptm1 = Vector{Float64}(undef,ndemes)
        payoff = Vector{Float64}(undef,pop_size)
        mean_payoff = Vector{Float64}(undef,ndemes)
        mfitness = Vector{Float64}(undef,pop_size)
        par_indices = collect(1:pop_size)
        
        init = ones(UInt,pop_size)     # The first index stores the resident, wild-type allele by default 
        if nloci == 1
            a = 0
            x = 0
            for i in 1:pop_size
                if i > x 
                    a += 1
                    if typeof(params.init_freqs) <: Vector{Float64}
                        x += rround(params.init_freqs[a]*pop_size) # Amount of individuals bearing the allele a
                        if length(params.mut_list) == a
                            x = pop_size
                        end
                    else
                        x += params.init_freqs[a] # Amount of individuals bearing the allele a    
                    end
                end
                init[i] = a
            end
            Gt[:,1] .= shuffle(init)
        else
            for (l,p0) in enumerate(params.init_freqs)
                a = 0
                x = 0
                for i in 1:pop_size
                    if i > x 
                        a += 1
                        if typeof(p0[l]) <: Vector{Float64}
                            x += rround(p0[l][a]*pop_size) # Amount of individuals bearing the allele a
                            if length(params.mut_list[l]) == a
                                x = pop_size
                            end
                        else
                            x += p0[l][a]
                        end
                    end
                    init[i] = a
                end
                Gt[:,l] .= shuffle(init)
            end
            
        end
        return new{S}(pop_size::Int, ndemes::Int, deme_size::Int, nloci::Int, deme2ind, ind2deme,
                   Gt, Gtm1, Pt, Ptm1, mean_Pt, mean_Ptm1, payoff,
                   mean_payoff, mfitness, par_indices, params, status)
    end
end
