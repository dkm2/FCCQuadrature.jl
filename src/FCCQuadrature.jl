"""
    FCCQuadrature

Compute finite Fourier integrals using Filon-Clenshaw-Curtis (FCC) quadrature.

Provide adaptive quadrature methods for oscillatory integrals of the form
`∫ f(x) g(x) exp(im ω x) dx`.

See also the preprint: [Filon-Clenshaw-Curtis Quadrature with Automatic Tone Removal](https://dkmj.org/academic/numfour.pdf)
"""
module FCCQuadrature
export fccquad, fccquad_cc, fccquad_cs, fccquad_sc, fccquad_ss

include("Fct.jl")
include("Cheb.jl")
include("chirps.jl")
include("chebweights.jl")
include("Duals.jl")
include("Jets.jl")

using LinearAlgebra, Scratch

const AV = AbstractVector
const AA = AbstractArray

#=
Filon-Clenshaw-Curtis quadrature for f(x) on [-1,1].
Assuming chebfun is the coefficient vector of
a Chebyshev expansion of degree at least N of some function f(x),
returns f(x)*exp(im*w*x)*dx integrated over [-1,1]
using degree N, and discrepancy between that and using reduced_degree(N).
WARNING: N and baseN must be even. An @assert enforces.
=#
function fccquadUnit!(chebfun::AV,freq::Real,N::Integer,
                      weightmethod::Symbol,weights::AV,weights_workspace::AA,
                      baseN=reduced_degree(N))
    getweights!(weights,weights_workspace,length(weights)-1,freq,weightmethod)
    fcc_core(chebfun,weights,N,baseN)
end    
reduced_degree(N)=div(3N,4)
function fcc_core(chebfun::AV,weights::Vector,N::Integer,baseN::Integer)
    @assert iseven(N) && iseven(baseN)
    a=chebfun
    i=weights
    base=(sum(a[n]*i[n] for n in baseN+1:-2:1) +
          sum(a[n]*i[n] for n in baseN:-2:2)*im)
    diff=(sum(a[n]*i[n] for n in N+1:-2:baseN+3) +
          sum(a[n]*i[n] for n in N:-2:baseN+2)*im)
    full = base + diff
    full,diff
end
#=
Alternative, more flexible error estimation method
that compares the results of using two user-provided Chebyshev expansions 
instead of one Chebyshev expansion and its truncation.
WARNING: N and baseN must be even. An @assert enforces.
=#
function fcc_alt(chebfun::AV,basefun::AV,
                 weights::Vector,N::Integer,baseN::Integer)
    @assert iseven(N) && iseven(baseN)
    a,b,i=chebfun,basefun,weights
    full=(sum(a[n]*i[n] for n in N+1:-2:1) +
          sum(a[n]*i[n] for n in N:-2:2)*im)
    base=(sum(b[n]*i[n] for n in baseN+1:-2:1) +
          sum(b[n]*i[n] for n in baseN:-2:2)*im)
    diff = full - base
    full,diff
end

#=
Filon-Clenshaw-Curtis quadrature for f(x) with arbitrary finite interval domain.
Assuming chebfun is the coefficient vector of
a Chebyshev expansion of degree at least N of some function g(x)=f(c+rx),
returns the integral over [-1,1] of f(c+rx)*exp(im*(w*(c+rx)+v*rx))*d(c+rx)
using degree N, and discrepancy between that and using degree baseN.
Above, w=freq, c=center, r=radius, and v=doppler.
WARNING: N and baseN must be even. An @assert enforces.
=#
function fccquadSampled!(chebfun::AV,freq::Real,N::Integer,
                         center::Real,radius::Real,doppler::Real,
                         weightmethod::Symbol,weights::AV,weights_workspace::AA,
                         baseN::Integer=reduced_degree(N))
    w = freq*radius + doppler
    interval_transform(fccquadUnit!(chebfun,w,N,weightmethod,weights,weights_workspace,baseN)...,
                       freq,center,radius)
end
function interval_transform(full,diff,freq,center,radius)
    si,co=sincos(freq*center)
    mult = radius*complex(co,si)
    full*mult, diff*mult
end

#Integrates f1(x)*f2(x)*exp(im*w*x)*dx over [xmin,xmax] for each w in freqs
#using degree N and discrepancy between that and using reduced_degree(N).
#WARNING: log2N must be at least 3. An @assert enforces.
function fccquadBatch(f1::Function,f2::Function,freqs::AA,log2N::Integer;
                      weightmethod::Symbol=:thomas,T::Type=Complex{Float64},
                      xmin::Real=-one(real(T)),xmax::Real=one(real(T)))
    @assert 3 <= log2N <= 62
    xminT,xmaxT = real(T)(xmin),real(T)(xmax)
    center = real(T)(0.5)*(xmaxT + xminT)
    radius = real(T)(0.5)*(xmaxT - xminT)
    output,workspaces = fccquad_alloc(freqs,log2N,weightmethod,T)
    fccquadBatch!(output,workspaces,center,radius,f1,f2,freqs,log2N,weightmethod)
    output,1+(1<<log2N)
end
function fccquadBatch!(output::AA,workspaces,center::Real,radius::Real,
                       f1::Function,f2::Function,freqs::AA,log2N::Integer,
                       weightmethod::Symbol,baseN=reduced_degree(1<<log2N))
    N=1<<log2N
    samples,fct_workspaces = workspaces[1]
    a=Fct.chebcoeffs!(Fct.shiftedchebsample!(center,radius,f1,f2,samples,N),fct_workspaces...)
    M=length(freqs)
    weights,weights_workspace = workspaces[2]
    for m in 1:M
        output[:,m] = collect(fccquadSampled!(a,freqs[m],N,center,radius,zero(real(eltype(output))),
                                              weightmethod,weights,weights_workspace,baseN))
    end
    output
end
function fccquad_alloc(freqs::AA,log2N::Integer,weightmethod::Symbol,T::Type=Complex{Float64})
    output=Matrix{T}(undef,2,length(freqs))
    N=1<<log2N
    output,fccquad_workspaces(N,weightmethod,T)
end
function fccquad_workspaces(N::Integer,weightmethod::Symbol,T::Type=Complex{Float64})
    samples = Vector{T}(undef,1+N)
    (samples,Fct.fct_alloc(samples,T)),weights_alloc(N,weightmethod,real(T))[1:2]
end

#=
Degree-adaptive FCC quadrature implementation.
Integrates f1(c+rx)*f2(c+rx)*exp(im*(w*(c+rx)+v*rx))*d(c+rx) over [-1,1] 
for each w in freqs, where c=center, r=radius, and v=doppler.

Expects output,workspaces = fccquad_alloc(freqs,maxlog2degree,weightmethod,T)
OR workspaces==nothing and size(output)==(2,length(freqs)).

WARNING: log2N is assumed to be at least 3.
=#
function adaptdegree!(f1::Function,f2::Function,freqs::AA,
                      center::Real,radius::Real,doppler::Real,
                      weightmethod::Symbol,output::AA,workspaces,
                      minlog2degree::Integer,maxlog2degree::Integer,
                      reltol::Real,abstol::Real,vectornorm::Function)
    N = 1 << minlog2degree
    T = eltype(output)
    if workspaces == nothing
        samples = Fct.shiftedchebsample(center,radius,f1,f2,N;T=T)
    else
        samples,fct_workspaces = workspaces[1]
        ifft_in,ifft_out,ifft_work = fct_workspaces
        weights_workspaces = workspaces[2]
        Fct.shiftedchebsample!(center,radius,f1,f2,samples,N)
    end
    
    R = real(T)
    base = zero(R)
    success = false
    maxN = 1 << maxlog2degree
    while true
        if workspaces == nothing
            ifft_out=Fct.chebcoeffs(samples)
        else
            Fct.chebcoeffs!(samples,ifft_in,ifft_out,ifft_work,N)
        end
        for m in 1:length(freqs)
            if workspaces == nothing
                weights_workspaces = weights_alloc(N,weightmethod,R)[1:2]
            end
            output[:,m] = collect(fccquadSampled!(ifft_out,freqs[m],N,center,radius,doppler,
                                                  weightmethod,weights_workspaces...))
        end
        base=vectornorm(view(output,1,:))
        delta=vectornorm(view(output,2,:))
        success = delta <= base * reltol || delta <= abstol
        if success || N >= maxN
            break
        end
        if workspaces == nothing
            samples = Fct.shifteddoublesample(center,radius,f1,f2,samples)
        else
            Fct.shifteddoublesample!(center,radius,f1,f2,samples,N)
        end
        N<<=1
    end
    success,N+1,base
end

#=
Degree-adaptive FCC quadrature with tone removal.
Expects output,workspaces = fccquad_alloc(freqs,maxlog2N,weightmethod,T).
Integrates prefactor(x)*oscillator(x)*exp(im*w*x)*dx over [c-r,c+r] 
for each w in freqs, where c=center and r=radius.
WARNING: arg(oscillator(x)) is assumed to be smooth at x=c.
WARNING: log2N is assumed to be at least 3.
=#
function tonequad!(output::AA,workspaces,center::Real,radius::Real,
                   prefactor::Function,oscillator::Function,freqs::AA,
                   minlog2N::Integer,maxlog2N::Integer,weightmethod::Symbol,
                   T::Type,reltol::Real,abstol::Real,vectornorm::Function)
    cfreq = Duals.phase_velocity(oscillator(Duals.Dual(center,radius)))
    if !isfinite(cfreq)
        cfreq = zero(cfreq)
    end
    
    samples,fct_workspaces = workspaces[1]
    ifft_in,ifft_out,ifft_work = fct_workspaces
    weights,weights_work = workspaces[2]

    N = 1 << minlog2N
    maxN = 1 << maxlog2N
    Fct.tonechebsample!(cfreq,center,radius,prefactor,oscillator,samples,N)
    success = false
    base = 0
    while true
        Fct.chebcoeffs!(samples,ifft_in,ifft_out,ifft_work,N)
        for m in 1:length(freqs)
            output[:,m] = collect(fccquadSampled!(ifft_out,freqs[m],N,center,radius,cfreq,
                                                  weightmethod,weights,weights_work))
        end
        base=vectornorm(view(output,1,:))
        delta=vectornorm(view(output,2,:))
        success = delta <= base * reltol || delta <= abstol
        if success || N >= maxN
            break
        end
        Fct.tonedoublesample!(cfreq,center,radius,prefactor,oscillator,samples,N)
        N<<=1
    end
    success,N+1,base
end

#=
Degree-adaptive FCC quadrature with chirp removal.
Integrates prefactor(x)*oscillator(x)*exp(im*w*x)*dx over [c-r,c+r] 
for each w in freqs, where c=center and r=radius.
WARNING: arg(oscillator(x)) is assumed to be smooth at x=c.
WARNING: log2N is assumed to be at least 3.
=#
function chirpquad!(failfast::Bool,output::AA,center::Real,radius::Real,
                    prefactor::Function,oscillator::Function,freqs::AA,
                    minlog2N::Integer,maxlog2N::Integer,weightmethod::Symbol,
                    T::Type,reltol::Real,abstol::Real,vectornorm::Function,
                    maxchirp::Integer)
    N = 1 << minlog2N
    maxN = 1 << maxlog2N
    R = real(T)

    jet = oscillator(Jets.Jet(center,radius))
    cfreq::Real = Jets.phase_velocity(jet)
    if !isfinite(cfreq)
        cfreq = zero(cfreq)
    end
    chirp::Real = Jets.phase_acceleration(jet)
    if !isfinite(chirp)
        chirp = zero(chirp)
    end
    if Chirps.rates[maxchirp] < abs(chirp)
        failfast && return false,1,zero(R)
        workspaces=fccquad_workspaces(maxN,weightmethod,T)
        return tonequad!(output,workspaces,center,radius,
                         prefactor,oscillator,freqs,
                         minlog2N,maxlog2N,weightmethod,
                         T,reltol,abstol,vectornorm)
    end
    
    samples = Fct.chirpchebsample(cfreq,chirp,center,radius,prefactor,oscillator,N;T=T)
    success = false
    base = zero(R)
    while true
        a=Cheb.ChebSeries(Fct.chebcoeffs(samples))

        index=findfirst(x->abs(chirp)<=x,view(Chirps.rates,1:maxchirp))::Integer
        faster=Chirps.rates[index]
        b0 = Cheb.EvenChebSeries(Chirps.getchirp(index))
        b=Cheb.dilate(b0,sqrt(abs(chirp)/faster))
        if chirp < zero(chirp)
            b = conj(b)
        end
        ab,ab2 = a*b, Cheb.truncate(a,reduced_degree(N))*b
        deg = Cheb.degree(Cheb.autotruncate(ab))
        deg2 = Cheb.degree(Cheb.autotruncate(ab2))
        @assert deg >=2 && deg2 >= 2
        deg1 = max(1, deg >> 2) << 2 #truncate to multiple of 4
        deg2 = max(1, deg2 >> 2) << 2 #truncate to multiple of 4
        deg3 = max(deg,deg2)
        
        weights,workspace,deg4=weights_alloc(deg3,weightmethod,R)
        for m in 1:length(freqs)
            w = freqs[m]*radius + cfreq
            getweights!(weights,workspace,deg4,w,weightmethod)
            step1 = fcc_alt(ab.coeff,ab2.coeff,weights,deg1,deg2)
            step2 = interval_transform(step1...,freqs[m],center,radius)
            output[:,m] = collect(step2)
        end
        
        base=vectornorm(view(output,1,:))
        delta=vectornorm(view(output,2,:))
        success = delta <= base * reltol || delta <= abstol
        if success || N >= maxN
            break
        end
        samples = Fct.chirpdoublesample(cfreq,chirp,center,radius,prefactor,oscillator,samples)
        N<<=1
    end
    success,N+1,base
end

#constants for fccquad()

"""
    interval_methods

Tuple of interval-adaptive FCC methods: `(:plain, :tone, :chirp)`.
"""
const interval_methods = (:plain, :tone, :chirp)

"""
    all_methods

Tuple of all available FCC methods: `(:plain, :tone, :chirp, :degree, :nonadaptive)`.
"""
const all_methods = (interval_methods..., :degree, :nonadaptive)

"""
    supported_types

Dictionary mapping method symbols to supported complex types.

# Example
```julia
julia> FCCQuadrature.supported_types[:tone]
(ComplexF32, ComplexF64, Complex{BigFloat})
```
"""
const supported_types = Dict(
    (m=>(Complex{T} for T in (Float32,Float64,BigFloat)) for m in (:plain, :tone))...,
    (m=>(Complex{T} for T in (Float64,BigFloat)) for m in (:degree, :nonadaptive))...,
    :chirp=>(Complex{Float64},))

"""
    default_reltol

Dictionary of default relative tolerances for each real type:
- `Float32`: `1e-6`
- `Float64`: `1e-8`
- `BigFloat`: `1e-39`
"""
const default_reltol = Dict(Float32=>1e-6,Float64=>1e-8,BigFloat=>1e-39)
const weightmethod=:thomas

"""
    fccquad(prefactor, oscillator, freqs; kwargs...)

Integrate `prefactor(x) * oscillator(x) * exp(im * w * x)` over a finite interval
for all `w` in `freqs`, using Filon-Clenshaw-Curtis (FCC) quadrature.

# Arguments
- `prefactor::Function`: prefactor function, should not be highly oscillatory.
- `oscillator::Function`: oscillator function, should be nonzero on [xmin, xmax].
- `freqs::AbstractArray{<:Real}`: array of angular frequencies.

# Keyword Arguments
- `xmin=-1.0`, `xmax=1.0`: domain of integration [xmin, xmax].
- `T=Complex{Float64}`: working complex scalar datatype. The functions `prefactor` and
  `oscillator` should return type `T` or `real(T)` when given input of type `real(T)`.
  See Supported Types below for available types per method.
- `reltol=default_reltol[real(T)]`: relative error tolerance.
- `abstol=0`: absolute error tolerance.
- `method=:tone`: FCC variant (see below).
- `vectornorm=LinearAlgebra.norm`: norm for discrepancy vectors. The adaptive methods use
  stopping rules based on vectors of discrepancies involving all frequencies in `freqs`.

# Methods
- `:nonadaptive`: FCC quadrature with the base-2 logarithm of the interpolation degree
  specified by `nonadaptivelog2degree` (default: 10).
- `:degree`: degree-adaptive FCC quadrature with the base-2 logarithm of the interpolation
  degree starting at `minlog2degree` (default: 3) and adaptively increasing but not beyond
  `globalmaxlog2degree` (default: 20).
- `:plain`, `:tone`, `:chirp`: hybrid interval-adaptive degree-adaptive FCC quadrature
  variants with the base-2 logarithm of the per-subinterval interpolation degree starting
  at `minlog2degree` (default: 3) and adaptively increasing but not beyond
  `localmaxlog2degree` (default: 6). If a subinterval's accuracy goals are not met even
  with the maximum degree, the subinterval is divided into `branching`-many equal
  subintervals (default: 4).

Methods `:tone` and `:chirp` use automatic differentiation to factor out a per-subinterval
tone or chirp (respectively) from `oscillator` before the polynomial interpolation step.
To use these methods, `oscillator` must be generic enough to accept input of type
[`FCCQuadrature.Jets.Jet`](@ref FCCQuadrature.Jets.Jet)`{real(T)}`, which is a subtype of `Number`.

# Returns
A tuple `(results, function_evaluation_count)` where `results` is a 2×L complex matrix:
- `results[1,i]`: estimated integral for i-th frequency
- `results[2,i]`: discrepancy (error estimate) for i-th frequency

# Supported Types
- `:nonadaptive`, `:degree`: `Complex{Float64}`, `Complex{BigFloat}`
- `:plain`, `:tone`: `Complex{Float32}`, `Complex{Float64}`, `Complex{BigFloat}`
- `:chirp`: `Complex{Float64}`

See [`FCCQuadrature.supported_types`](@ref supported_types) for programmatic access.

See also [`fccquad_cc`](@ref), [`fccquad_cs`](@ref), [`fccquad_sc`](@ref), [`fccquad_ss`](@ref).
"""
function fccquad(prefactor::Function,oscillator::Function,freqs::AA{<:Real};
                 T::Type=Complex{Float64},xmin::Real=-one(real(T)),xmax::Real=one(real(T)),
                 reltol::Real=default_reltol[real(T)],abstol::Real=zero(real(T)),
                 method::Symbol=:tone,vectornorm=LinearAlgebra.norm,
                 branching::Integer=4,maxdepth::Integer=10,maxchirp::Integer=6,
                 minlog2degree::Integer=3,
                 localmaxlog2degree::Integer=6,
                 nonadaptivelog2degree::Integer=10,
                 globalmaxlog2degree::Integer=20)
    @assert xmin < xmax && isfinite(xmin) && isfinite(xmax)
    @assert 3 <= minlog2degree <= localmaxlog2degree <= 62
    @assert 3 <= minlog2degree <= globalmaxlog2degree <= 62
    @assert 3 <= nonadaptivelog2degree <= 62
    @assert method in all_methods
    @assert T in supported_types[method]
    @assert 2 <= branching
    @assert 1 <= maxchirp <= length(Chirps.rates)

    
    if method == :nonadaptive
        return fccquadBatch(prefactor,oscillator,freqs,nonadaptivelog2degree;
                            T=T,xmin=xmin,xmax=xmax,weightmethod=weightmethod)
    end
    
    xminT,xmaxT = real(T)(xmin),real(T)(xmax)
    center = real(T)(0.5)*(xmaxT + xminT)
    radius = real(T)(0.5)*(xmaxT - xminT)
    if method == :degree
        output=Matrix{T}(undef,2,length(freqs))
        success,evals,base=adaptdegree!(prefactor,oscillator,freqs,
                                        center,radius,zero(real(T)),
                                        weightmethod,output,nothing,
                                        minlog2degree,globalmaxlog2degree,
                                        reltol,abstol,vectornorm)
        return output,evals
    end

    output=zeros(T,2,length(freqs)) #initialize accumulator to zero
    if method == :chirp
        subintegrals,workspaces = Matrix{T}(undef,2,length(freqs)),nothing
    else # :plain, :tone
        subintegrals,workspaces = fccquad_alloc(freqs,localmaxlog2degree,weightmethod,T)
    end
    interval_adaptive!(output,subintegrals,workspaces,center,radius,
                       prefactor,oscillator,freqs,branching,maxdepth,0,
                       minlog2degree,localmaxlog2degree,reltol,abstol,
                       method,weightmethod,T,vectornorm,maxchirp)
end
#adds results in place to output
function interval_adaptive!(output::AA,subintegrals::AA,workspaces,center::Real,radius::Real,
                            prefactor::Function,oscillator::Function,freqs::AA,
                            branching::Integer,maxdepth::Integer,depth::Integer,
                            minlog2N::Integer,maxlog2N::Integer,reltol::Real,abstol::Real,
                            method::Symbol,weightmethod::Symbol,T::Type,vectornorm::Function,
                            maxchirp::Integer)
    @assert method in interval_methods
    if method == :chirp #degree-adaptive FCC quadrature with linear chirp removal
        success,evals,base=chirpquad!(depth<maxdepth,subintegrals,center,radius,
                                      prefactor,oscillator,freqs,minlog2N,maxlog2N,
                                      weightmethod,T,reltol,abstol,vectornorm,maxchirp)
    elseif method == :tone #degree-adaptive FCC quadrature with tone removal
        success,evals,base=tonequad!(subintegrals,workspaces,center,radius,
                                     prefactor,oscillator,freqs,
                                     minlog2N,maxlog2N,weightmethod,
                                     T,reltol,abstol,vectornorm)
    elseif method == :plain #degree-adaptive FCC quadrature
        success,evals,base=adaptdegree!(prefactor,oscillator,freqs,center,radius,zero(real(T)),
                                        weightmethod,subintegrals,workspaces,
                                        minlog2N,maxlog2N,reltol,abstol,vectornorm)
    end
    if success || depth >= maxdepth
        output .+= subintegrals
    else
        shrink = inv(real(T)(branching))
        r = shrink*radius
        abstol = shrink*max(abstol, base*reltol)
        for t in 1-branching:2:branching-1
            evals += interval_adaptive!(output,subintegrals,workspaces,r*t+center,r,
                                        prefactor,oscillator,freqs,branching,maxdepth,depth+1,
                                        minlog2N,maxlog2N,reltol,abstol,
                                        method,weightmethod,T,vectornorm,maxchirp)[2]
        end
    end
    output,evals
end


"""
    fccquad_cc(amplitude, angle, freqs; kwargs...)

Integrate `amplitude(x) * cos(angle(x)) * cos(w * x)` over a finite interval
for all `w` in `freqs`, using Filon-Clenshaw-Curtis quadrature.

# Arguments
- `amplitude::Function`: amplitude function, should be real-valued and not highly oscillatory.
- `angle::Function`: angle function, should be real-valued. Its derivative should not be highly oscillatory.
- `freqs::AbstractArray{<:Real}`: array of angular frequencies.

# Keyword Arguments
- `xmin=-1.0`, `xmax=1.0`: domain of integration [xmin, xmax].
- `T=Float64`: working real scalar datatype. The functions `amplitude` and `angle` should
  return type `T` when given input of type `T`.

See [`fccquad`](@ref) for documentation of all other keyword arguments.

# Returns
A tuple `(results, function_evaluation_count)` where `results` is a 2×L real matrix:
- `results[1,i]`: estimated integral for i-th frequency
- `results[2,i]`: discrepancy (error estimate) for i-th frequency

See also [`fccquad_cs`](@ref), [`fccquad_sc`](@ref), [`fccquad_ss`](@ref), [`fccquad`](@ref).
"""
function fccquad_cc(amplitude::Function,angle::Function,freqs::AA{<:Real};
                    kwargs...)
    fccquad_trig(coscos!,amplitude,angle,freqs;kwargs...)
end

"""
    fccquad_cs(amplitude, angle, freqs; kwargs...)

Like [`fccquad_cc`](@ref), but integrate `amplitude(x)*cos(angle(x))*sin(w*x)*dx`.
"""
function fccquad_cs(amplitude::Function,angle::Function,freqs::AA{<:Real};
                    kwargs...)
    fccquad_trig(cossin!,amplitude,angle,freqs;kwargs...)
end

"""
    fccquad_sc(amplitude, angle, freqs; kwargs...)

Like [`fccquad_cc`](@ref), but integrate `amplitude(x)*sin(angle(x))*cos(w*x)*dx`.
"""
function fccquad_sc(amplitude::Function,angle::Function,freqs::AA{<:Real};
                    kwargs...)
    fccquad_trig(sincos!,amplitude,angle,freqs;kwargs...)
end

"""
    fccquad_ss(amplitude, angle, freqs; kwargs...)

Like [`fccquad_cc`](@ref), but integrate `amplitude(x)*sin(angle(x))*sin(w*x)*dx`.
"""
function fccquad_ss(amplitude::Function,angle::Function,freqs::AA{<:Real};
                    kwargs...)
    fccquad_trig(sinsin!,amplitude,angle,freqs;kwargs...)
end

function fccquad_trig(transform!,amplitude::Function,angle::Function,freqs::AA;
                      T::Type=Float64,vectornorm=LinearAlgebra.norm,kwargs...)
    symmetricfreqs = Vector{eltype(freqs)}(undef,2length(freqs))
    for i in 1:length(freqs)
        w = freqs[i]
        symmetricfreqs[2i-1],symmetricfreqs[2i] = w,-w
    end
    workspace = Vector{T}(undef,length(freqs))
    pseudonorm(v) = trignorm!(v, workspace, transform!, vectornorm)
    
    oscillator(z) = exp(im*angle(z))
    cis,evals = fccquad(amplitude,oscillator,symmetricfreqs;T=Complex{T},vectornorm=pseudonorm,kwargs...)
    output = Matrix{T}(undef,2,length(freqs))
    for i in 1:2
        transform!(view(cis,i,:),view(output,i,:))
    end
    output,evals
end

function trignorm!(v, workspace, transform!, vectornorm)
    transform!(v, workspace)
    vectornorm(workspace)
end

function coscos!(exps, trigs)
    for i in 1:length(trigs)
        #2cos(a)cos(b)==cos(a+b)+cos(a-b)
        trigs[i] = 0.5(real(exps[2i-1])+real(exps[2i]))
    end
end

function cossin!(exps, trigs)
    for i in 1:length(trigs)
        #2cos(a)sin(b)==sin(a+b)-sin(a-b)
        trigs[i] = 0.5(imag(exps[2i-1])-imag(exps[2i]))
    end
end

function sincos!(exps, trigs)
    for i in 1:length(trigs)
        #2sin(a)cos(b)==sin(a+b)+sin(a-b)
        trigs[i] = 0.5(imag(exps[2i-1])+imag(exps[2i]))
    end
end

function sinsin!(exps, trigs)
    for i in 1:length(trigs)
        #2sin(a)sin(b)==cos(a-b)-cos(a+b)
        trigs[i] = 0.5(real(exps[2i])-real(exps[2i-1]))
    end
end

function __init__()
    cache_dir = @get_scratch!("precomputed_chebyshev_weights")
    cache_file = joinpath(cache_dir,"chirps.h5")
    isfile(cache_file) || Chirps.store(cache_file)
    Chirps.load(cache_file)
end

end #module
