using QuadGK 
using SpecialFunctions
################################################
# Expressions for structred populations in Wright's island model

# Fixation index (relatedness coefficient), approximation when Nm<<1
fst_approx(N::Int,m::Float64, pl::Int=1) = 1/(1+2*pl*N*m)

# Full expression, infinite island, gametic migration
fst_inf(N::Real, m::Float64, pl::Int=1) = (1-m)^2 / (pl*N - (pl*N-1)*(1-m)^2) 
# Fixation index in finite island model, gametic migration
fst_b(N::Real, m::Float64, nd::Int; pl::Int=1) = (1-m*nd/(nd-1))^2 / (pl*N - (pl*N-1)*(1-m*nd/(nd-1))^2) 
fst_b(N::Real, m::Float64, nd::Int; pl::Int=1, u::Float64=0.0) = (1-u)^2*(1-m*nd/(nd-1))^2 / (pl*N - (1-u)^2*(pl*N-1)*(1-m*nd/(nd-1))^2) 

# Fixation index in finite island model, diploid reproduction, zygotic migration, self-fertilization rate 1/n

function fst_b_zyg(N::Real, m::Float64, nd::Int) 
    ω = (1 - m*nd/(nd-1)) ^ 2 
    return ω / (2*N - 1 - (2*N-2)*ω + (1 - ω)/nd)
end

function fst_b_zyg(N::Real, m::Float64, nd::Int, u::Float64=0.0) 
    ω = (1 - m*nd/(nd-1)) ^ 2 
    γ = (1-u)^2
    return γ*ω / (2*N - γ*(1 + (2*N-2)*ω - (1 - ω)/nd))
end

# Fixation index for group competition in a finite island model, gametic migration
fst_g(N::Real, m::Float64, nd::Int; pl::Int=1) = (1-m*nd/(nd-1))^2 / (pl*N * (nd/(nd-1)) - (pl*N-1)*(1-m*nd/(nd-1))^2)
fst_g(N::Real, m::Float64, nd::Int; pl::Int=1, u::Float64=0.0) = (1-u)^2*(1-m*nd/(nd-1))^2 / (pl*N * (nd/(nd-1)) - (1-u)^2*(pl*N-1)*(1-m*nd/(nd-1))^2)
# Fixation index in finite island model with group competition, diploid reproduction, zygotic migration, self-fertilization rate 1/n
function fst_g_zyg(N::Real, m::Float64, nd::Int) 
    ω = (1 - m*nd/(nd-1)) ^ 2 
    return  ω / (2*N*nd/(nd-1) - 1 - (2*N-2)*ω + (1 - ω)/nd)
end

function fst_g_zyg(N::Real, m::Float64, nd::Int, u::Float64=0.0) 
    ω = (1 - m*nd/(nd-1)) ^ 2 
    γ = (1-u)^2
    return γ*ω / (2*N*nd/(nd-1) - γ*(1 + (2*N-2)*ω - (1 - ω)/nd))
end

# Fixation index for group competition with envinronmental stochasticity
#=fst_sd(N::Real, m::Float64, nd::Int, sd::Float64, u::Float64=0.0) =  ( (1-u)^2*(1 - m*nd/(nd - 1))^2)/(n/((1 + sd/(nd - 1))*(1 - 1/(
    nd*sd))) - (1-u)^2*(n - 1)*(1 - m*nd/(nd - 1))^2) #sd*(1-u)^2*(1-m*nd/(nd-1))^2 / (n - sd*(1-u)^2*(n-1)*(1-m*nd/(nd-1))^2)
=#
fst_sd(N::Real, m::Float64, nd::Int, sd::Float64, u::Float64=0.0) = ((1 - (m*nd*sd)/(-1 + nd*sd))^2*(1 - u)^2)/((n*(nd-1))/(-1 + sd*nd) - (-1 + N)*(1 - (m*nd*sd)/(-1 + nd*sd))^2*(1 - u)^2)

fst_sd_ninf(N::Real, m::Float64, sd::Float64, u::Float64=0.0) = (sd*(1-u)^2*(1 - m)^2)/(
 N - sd*(-1 + N)*(1-u)^2*(1 - m)^2)

 ########### EFFECTIVE POPULATION SIZE ###############
# Baseline scheme (Wright island, gametic migration), 1st order Taylor series approx. in the limit nd->infty
Neff_b(N::Real, m::Float64, nd::Int, pl::Int=1) = nd*N / (1 - fst_inf(N,m,pl)) 
# , full expression
function Neff_b_full(N::Real, m::Float64, nd::Int, pl::Int=1) 
    M = m*nd/(nd-1)
    β = M*(2 - M)
    α = (nd + β*(1 + nd*(pl*N - 1)))/2
    return (N*nd)/(α - sqrt(α^2 - pl*N*nd*β))
end
# Baseline scheme (Wright island, zygotic migration), full expression
function Neff_b_zyg_full(N::Real, m::Float64, nd::Int) 
    M = m*nd/(nd-1)
    β = M*nd*(nd*((2 - M)*nd - 4) + 2)
    α = ((nd - 1)^4)/2 + β*(1 + (N - 1)*nd)
    return (N*(nd - 1)^4)/(α - sqrt(α^2 - 2*N*(nd - 1)^4*β))
end

# Island model with group competition, gametic migration, 1st order Taylor series approx. in the limit nd->infty
Neff_g(N::Real, m::Float64, nd::Int, pl::Int=1) = nd*N / (1 - fst_inf(N,m,pl)*(1 - pl*N / (1-m)^2 ) )
# Island model with group competition, gametic migration, full expression)
function Neff_g_full(N::Real, m::Float64, nd::Int, pl::Int=1) 
    β = m*(2 - m*nd/(nd-1))
    α = (nd + pl*N + β*(1 + nd*(pl*N - 1)))/2
    return (N*nd)/(α - sqrt(α^2 - pl*N*nd*(β+1)))
end
# Island model with group competition, zygotic migration, full expression)
function Neff_g_zyg_full(N::Real, m::Float64, nd::Int) 
    M = m*nd/(nd-1)
    β = M*(2 - M*( nd/(nd - 1))^2)
    α = nd/2 + N + nd*(N*nd/(nd - 1) - 1)*β
    return (N*nd)/(α - sqrt(α^2 - 2*N*nd*(nd/(nd - 1)*β + 1)))
end

# Effective population size for finite island model with envinronmental stochasticity, migrant pool dispersal
function Neff_gsd(n::Real,m::Float64,nd::Int, sd::Float64=1.0)
    R = fst_sd_ninf(n,m,sd)
    return (sd*nd*n)/(1 - R*(1 - n*(1 - sd)))
end

# Effective selection coefficient, infinite island
seff(s::Float64,N::Int,m::Float64,fst::Function) = (1-fst(N,m))*s #(1+1/(2*N*m))*s 
# Effective selection coefficient, finite island 
seff(s::Float64,N::Int,m::Float64,nd::Int,fst::Function) = (1-fst(N,m,nd))*s #(1+1/(2*N*m))*s 

########### INCLUSIVE FITNESS EFFECT EXPRESSIONS #################
######## Gametic migration cases
SIF_A9(B::Real, C::Real,N::Real,m::Float64,nd::Int, pl::Int=1) = (-C + B/(N))*(1-fst_b(N, m, nd, pl=pl)) # Inclusive fitness effect, infinite-island model, helping after rep, before dispersal
SIF_A15(B::Real, C::Real, N::Real, m::Float64, nd::Int, pl::Int=1) = -C + B * fst_g(N, m, nd, pl=pl) # Inclusive fitness effect, finite-island model, group reproduction
function SIF_A27(B::Real,C::Real,n::Real,m::Float64,nd::Int,sd::Float64,u::Float64=0.0)
    R = fst_sd(n,m,nd,sd,u)
    RR = 1/n + (n - 1)/n*R
    return -C*(1 - R) + B*RR # ( (1/(n*(1 - (1 - m*nd/(nd-1))^2*sd))) )*B-C 
end

######## Zygotic migration cases
function SIF_A9_zyg(B::Real, C::Real, N::Real, m::Float64, nd::Int; BGS::Float64=1.0, BGS_mig::Float64=1.0) # Inclusive fitness effect, infinite-island model, helping after rep, before dispersal
    fst = fst_b_zyg(BGS * N, m, nd)
    #nu_hard_s = (BGS*N*nd/Neff_b_zyg_full(BGS*N, m, nd)) 
    fis = 1/(2*BGS*N - 1)
    nu_hard_s = (1-fst)*(1+fis) #= 
    nu_hard_s = (1 + fst) / 2 
    nu_hard_s = 1 - 2*fst / (1 + fst)=# 
    return (-C + B / (2*BGS*N)) * nu_hard_s
end

function SIF_A15_zyg(B::Real, C::Real, N::Real, m::Float64, nd::Real, BGS::Float64=1.0) # Inclusive fitness effect, finite-island model, group reproduction
    fst = fst_g_zyg(BGS * N, m, nd)
    #f0N = BGS * N
    #r = 1/(f0N-(f0N-1)*(1-m)^2)
    r = 2*fst/(1 + fst) 
    return -C + B * r
end
  
# analytical fixation probability, diffusion approximation, arbitrary dominance 0<h<1
function fixprob(x, n, s; h::Float64=0.5)
    if s==0
        return x # per L'Hopitale rule
    elseif -1<=s<=1
        α = 2*n*s
        if h==0.5
            return (1 - exp(- α * x)) / (1 - exp(- α ))
        elseif 0<=h && h<=1
        # Based on Eqs. 4.17 and 5.46 (Ewens, 2004)
            ζ = sqrt(Complex(α)) / sqrt(Complex(-1 + 2*h))
            return real((erfi(h*ζ) - erfi((x + h*(1 - 2*x))*ζ)) / (erfi(h*ζ) - erfi((1 - h)*ζ)))
        else
            throw(DomainError("argument h must be equal or greater than 0 and less or equal to 1"))
        end
    else
        throw(DomainError("argument s must be equal or greater than -1 and less or equal to 1"))
    end
end

################################################
# Diffusion approximation for the mean time until fixation for a biallelic locus, 
# in a haploid population

# Initial frquency of allele A1 is 1-p and that of A2 is p.

# Conditional fixation of a neutral allele, i.e. only drift, no mutation 
# Based on equation 14, Kimura and Ohta, 1969
function tbar1(p::Float64,N::Real; pl::Int=1) 
    return -2*pl*N*(1-p)*log(1-p) / p
end

# Conditional fixation of an advantageous allele, i.e. drift and selection, no mutation, semidominance h=1/2
# Based on equation 12, Kimura and Ohta, 1969  
function tbar1(p::Float64,N::Real,s::Float64; pl::Int=1)
    if s==0.0
        return tbar1(p,N;pl=pl)
    elseif 0.0 <= p <= 1.0
    #    u(p) = (1 - exp(-α*p)) / (1 - exp(-α)) 

    # \psi(x) = \frac{2 \int_{0}^{1} G(x) dx}{V(x)G(x)}$
    # with
    # $G(x) = e^{-\int \frac{2 M(x)}{V(x)} dx}$
    # $V(x) = \frac{x(1-x)}{N_e}$
        #psi(x) = (exp(α*x)*(1 - exp(-α))) / (s*(1 - x)*x)
        #t_int1(ξ) = psi(ξ)*u(ξ)*(1-u(ξ))
        #t_int2(ξ) = psi(ξ)*u(ξ)^2
        α = N*s
        t_int1(ξ) = 2*pl*(coth(α*p) - coth(α)) * sinh(α*ξ)^2 / (s*ξ*(1-ξ))
        t_int2(ξ) = 2*pl*csch(α) * sinh(α*(1-ξ)) * sinh(α*ξ) / (s*ξ*(1-ξ))
        #t_int1(ξ) = (2*exp(-α*ξ)*(1 - exp(-α*(1 - p)))*(exp(α*ξ) - 1)^2)/(α*ξ*(1 - ξ)*(1 - exp(-α))*(exp(α*p) - 1))
        #t_int2(ξ) = (2*(exp(α*ξ) - 1)*(exp(α*(1 - ξ)) - 1))/(α*ξ*(1 - ξ)*(exp(α) - 1))
        tbar1_1, err_tbar1_1 = quadgk(t_int1,0,p)
        tbar1_2, err_tbar1_2 = quadgk(t_int2,p,1)
        return (tbar1_1 + tbar1_2) # tbar1_1 + (1 - u(p))/u(p) * tbar1_2 
    else
        throw(DomainError("argument p must be equal or greater than 0 and less or equal to 1"))
    end
end

# Mean absorption time with neither selection nor mutation
function t_bar_star(p::Float64,N::Real; pl::Int=1)
    return -2*pl*N*(p*log(p)+(1-p)*log(1-p)) # Eq. 5.19 Mathematical Population Genetics, 2004, Ewens
end

# Conditional fixation of an advantageous allele, i.e. drift and selection, no mutation, arbitrary dominance 0<h<1
# Based on equations 4.48-4.50 Mathematical Population Genetics, 2004, Ewens
function tbar1(p::Float64,N::Real,s::Float64; pl::Int=2, h::Float64=0.5)
    if s==0.0
        return tbar1(p,N;pl=pl)
    elseif 0.0 <= p <= 1.0
        α = 2*N*s
        t_int1(ξ) = (2*pl*(coth(α*p/2) - coth(α/2)) * sinh(α*ξ/2)^2) / (s*ξ*(1-ξ))
        t_int2(ξ) = 2*pl*csch(α/2) * sinh(α*(1-ξ)/2) * sinh(α*ξ/2) / (s*ξ*(1-ξ))
        #t_int1(ξ) = (2*exp(-α*ξ)*(1 - exp(-α*(1 - p)))*(exp(α*ξ) - 1)^2)/(α*ξ*(1 - ξ)*(1 - exp(-α))*(exp(α*p) - 1))
        #t_int2(ξ) = (2*(exp(α*ξ) - 1)*(exp(α*(1 - ξ)) - 1))/(α*ξ*(1 - ξ)*(exp(α) - 1))
        tbar1_1, err_tbar1_1 = quadgk(t_int1,0,p)
        tbar1_2, err_tbar1_2 = quadgk(t_int2,p,1)
        return (tbar1_1 + tbar1_2) # tbar1_1 + (1 - u(p))/u(p) * tbar1_2 
    else
        throw(DomainError("argument p must be equal or greater than 0 and less or equal to 1"))
    end
end

#=
# Mean absorption time with selection, no mutation
function tbar(p::Float64,N::Float64,s::Float64)  # Eq. 5.48 Mathematical Population Genetics, 2004, Ewens
    if s==0.0
        return t_bar_star(p,N)
    else
        α = 2*N*s
        P1(p) = (1 - exp(-α*p)) / (1 - exp(-α))  # Eq. 5.47, Probability of fixation

        t_int1(x) = 2*(1-P1(p)) / (α*x*(1-x)) * (exp(α*x) - 1)
        t_int2(x) = 2*P1(p) / (α*x*(1-x)) * (1 - exp(-α*(1-x)))

        tbar1_1, err_tbar1_1 = quadgk(t_int1,0,p)
        tbar1_2, err_tbar1_2 = quadgk(t_int2,p,1)
        return N*(tbar1_1 + tbar1_2)
    end
end
=#

######################################################
######## HETEROZYGOSITY  IN BASELINE SCHEME ##########

function Het(Ne::Real,μ::Float64,pl::Int=1)
    θ = 2*pl*Ne*μ
    return θ
end
######### WITHIN individual heterozygosity given by 1-Q0 ############
# Zygotic migration, full expression
function Het_Q0_zyg_full(N::Real, nd::Int, m::Float64, μ::Float64)
    γ = (1 - μ)^2
    ω = (1 - m * nd / (nd - 1))^2
    return 1 - γ * (1 - γ + γ*(1 - ω) / nd) / (γ * (1 - ω) * (2 - γ) / nd +
                                                   (1 - γ) * (γ * (2 * ω - 1) + 2 * N * (1 - γ * ω)))
end


######### WITHIN deme heterozygosity given by 1-Q1 ############
# Gametic migration, infinite-deme approximation, 
function Het_Q1(Ne::Real,μ::Float64,pl::Int=1)
    θ = 2*pl*Ne*μ
    return θ
end

# Gametic migration, full expression
function Het_Q1_full(N::Real,nd::Int,m::Float64, μ::Float64, pl::Int=1)
    γ = (1-μ)^2
    Γ = γ*(1 - m*nd/(nd-1))^2
    return (1 - γ)*(1 - Γ)/(γ/(pl*N)*(m^2/(nd - 1) + (1 - m)^2 - Γ) + (1 - γ)*(1 - Γ))
end

# Zygotic migration, full expression
function Het_Q1_zyg_full(N::Real,nd::Int,m::Float64, μ::Float64)
    γ = (1-μ)^2
    ω = (1 - m*nd/(nd-1)) ^ 2
    return 1 - γ * (ω * (1-γ) + (1-ω)/nd) / ( γ * (1-ω) * (2-γ)/nd +
                                                (1-γ)*(γ*(2*ω-1) + 2*N * (1-γ*ω))  ) 
end
######### BETWEEN deme heterozygosity given by 1-Q2 ############
# Gametic migration
function Het_Q2(N::Real,nd::Int,m::Float64,μ::Float64,pl::Int=1)
    θ = Het_Q1(N*nd,μ,pl)-((2*(1 - (1 - m)*nd)^2)/(m*(2 + (-2 + m)*nd)))*μ 
    return θ
end
# Gametic migration, full expression
function Het_Q2_full(N::Real,nd::Int,m::Float64, μ::Float64, pl::Int=1)
    γ = (1-μ)^2
    Γ = γ*(1 - m*nd/(nd-1))^2
    return (1 - γ)*(1 - Γ*(1 - 1/(pl*N)))/(γ/(pl*N)*(m^2/(nd - 1) + (1 - m)^2 - Γ) + (1 - γ)*(1 - Γ))
end
# Zygotic migration, full expression
function Het_Q2_zyg_full(N::Real,nd::Int,m::Float64, μ::Float64)
    γ = (1-μ)^2
    ω = (1 - m*nd/(nd-1)) ^ 2
    return 1 - γ * (1-ω)/nd / ( γ * (1-ω) * (2-γ)/nd +
                                                (1-γ)*(γ*(2*ω-1) + 2*N * (1-γ*ω))  ) 
end

###############################################################
######### HETEROZYGOSITY IN GROUP COMPETITION SCHEME ##########

function Het_g(N::Real,nd::Int,m::Float64,μ::Float64,pl::Int=1)
    nd_1 = nd/(nd-1)
    θ = 2*pl*N*nd*μ*(nd_1 - (1 - m*nd_1)^2)/( 1 + nd_1-(1 - m*nd_1)^2)
    return θ
end
######### WITHIN individual heterozygosity given by 1-Q0 ############
# Zygotic migration, full approximation
function Het_g_Q0_zyg_full(N::Real, nd::Int, m::Float64, μ::Float64)
    γ = (1 - μ)^2
    NDm1 = nd - 1
    ω = (1 - m * nd / NDm1)^2
    return 1 - ((1 + NDm1)^2 - NDm1 * γ * (NDm1 + ω)) / (1 +
                                                         NDm1 * (2 - γ * ω + NDm1 * (1 - γ) * (2 * ω - 1)) +
                                                         2 * N * nd * (1 - γ) / γ * (1 + NDm1 * (1 - γ * ω))
    )
end
######### WITHIN deme heterozygosity given by 1-Q1 ############
# Gametic migration, infinite-deme approximation,
function Het_g_Q1(N::Real,nd::Int,m::Float64,μ::Float64,pl::Int=1)
    nd_1 = nd/(nd-1)
    θ = 2*pl*N*nd*μ*(nd_1 - (1 - m*nd_1)^2)/( 1 + nd_1 - (1 - m*nd_1)^2)
    return θ
end
# Gametic migration, full expression
function Het_g_Q1_full(N::Real,nd::Int,m::Float64,μ::Float64,pl::Int=1)
    γ = (1-μ)^2
    nd_1 = nd/(nd-1)
    Γ = γ*(1 - m*nd_1)^2
    return ((1 - γ)*( nd_1 - Γ))/(γ/(pl*N)*( (1 + m^2)/(nd - 1) + (1 - m)^2 - Γ) + (1 - γ)*(nd_1 - Γ))
end
# Zygotic migration, full approximation
function Het_g_Q1_zyg_full(N::Real,nd::Int,m::Float64,μ::Float64)
    γ = (1-μ)^2
    NDm1 = nd-1
    ω = (1 - m*nd/NDm1) ^ 2
    return 1 - ( 1 + NDm1*(2 - ( γ*nd - NDm1)*ω))/(1 + 
        NDm1*(2 - γ*ω + NDm1*(1 - γ)*(2*ω - 1)) + 
        2*N*nd*(1 - γ)/γ*(1 + NDm1*(1 - γ*ω))
      )
end

########## BETWEEN deme heterozygosity given by 1-Q2 ############
# Gametic migration, infinite-deme approximation
function Het_g_Q2(N::Real,nd::Int,m::Float64,μ::Float64,pl::Int=1)
    nd_1 = nd/(nd-1)
    θ = Het_g_Q1(N,nd,m,μ,pl) + (2*nd*μ*(1 - m*nd_1)^2)/(1 + nd_1 - (1 - m*nd_1)^2)
    return θ
end
# Gametic migration, full expression
function Het_g_Q2_full(N::Real,nd::Int,m::Float64,μ::Float64,pl::Int=1)
    γ = (1-μ)^2
    nd_1 = nd/(nd-1)
    Γ = γ*(1 - m*nd_1)^2
    return ((1 - γ)*( nd_1 - Γ*(1 - 1/(pl*N))))/(γ/(pl*N)*( (1 + m^2)/(nd - 1) + (1 - m)^2 - Γ) + (1 - γ)*(nd_1 - Γ))
end
# Zygotic migration, full approximation
function Het_g_Q2_zyg_full(N::Real,nd::Int,m::Float64,μ::Float64)
    γ = (1-μ)^2
    NDm1 = nd-1
    ω = (1 - m*nd/NDm1) ^ 2
    return 1 - (1 + NDm1*(2 - ω))/(1 + 
    NDm1*(2 - γ*ω + NDm1*(1 - γ)*(2*ω - 1)) + 
    2*N*nd*(1 - γ)/γ*(1 + NDm1*(1 - γ*ω))
      )
end
#########################################################3######

function f0_BGS(Ud,sh,R)
    if sh == 0.0
        return 1.0
    else
        return exp(-Ud/(2*sh + R)) 
    end
end

function f0_BGS_mig(Ud,sh,R,m)
    if sh == 0.0
        return 1.0
    else
        return exp(-Ud*sh/((sh + m)*(2*sh + R + 2*m))) 
    end
end

function f0_BGS_Good(Ud,sh,R,N;pl=2)
    if sh == 0.0
        return 1.0
    else
        λ = Ud/(2*sh + R)
        zint(z) = z*log(1/(1-z))*exp(λ*z^2)
        zint_sol, _ = quadgk(zint,0,1)
        return exp(-λ) + 2*λ*exp(-λ) / (pl*N*sh) * zint_sol
    end
end

function f0_BGS_mig_Good(Ud,sh,R,m,N;pl=2)
    if sh == 0.0
        return 1.0
    else
        λ = Ud*sh / ((sh + m)*(2*sh + R + 2*m))
        zint(z) = z*log(1/(1-z))*exp(λ*z^2)
        zint_sol, _ = quadgk(zint,0,1)
        return exp(-λ) + 2λ*exp(-λ) / (pl*N*sh) * zint_sol
    end
end
# binomial distribution conf intervals
function errorBarBinomialConf(fvec::Vector{Float64}, n::Int, alpha::Float64=0.05)
    # fvec is a vector of frequencies, n is the sample size, alpha is the significance level
    len = length(fvec)
    errBars = zeros((len,2))
    levels = [alpha/2,1-alpha/2] # lower and upper bounds
    for i=1:len
        errBars[i,:] .= map(level->quantile(Binomial(n,fvec[i]), level) / n, levels) .- fvec[i]
    end
    return errBars
end

function errorBarConfInt(nobs::Int,sample_std::Float64,alpha::Float64=0.05,sigma_isknown::Bool=false)
    levels = [alpha/2,1-alpha/2] # lower and upper bounds
    # sigma_isknown sets whether the population standard deviation is known or not
    # if sigma_isknown is true, use normal distribution, otherwise use t-distribution 
    if sigma_isknown || (!sigma_isknown && nobs >= 30) # normal distribution for large sample sizes
        return map(level->quantile(Normal(0,1),level)*sample_std/sqrt(nobs),levels)
    else # t-distribution for small sample sizes
        return map(level->quantile(TDist(nobs-1),level)*sample_std/sqrt(nobs),levels)
    end
end