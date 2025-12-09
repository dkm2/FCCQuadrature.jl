using FCCQuadrature: Fct, Cheb, getweights, fccquad_cc, fccquad_cs, fccquad_sc, fccquad_ss, all_methods, supported_types, interval_methods
using QuadGK, Test

#=
Integral of cos(13cos(2x))*52sin(x)cos(x)dx from 0 to 17pi/4
  == integral of cos(u)*du from 0 to 13 == sin(13).
Integral of sin(13cos(2x))*52sin(x)cos(x)dx from 0 to 17pi/4
  == integral of sin(u)*du from 0 to 13 == 1-cos(13).
=#
@testset "analytic integrals" begin
    println("Integral of (cos or sin)(13cos(2x))*52sin(x)cos(x)dx from 0 to 17pi/4:")
    angle(x)=13cos(2x)
    ampcos(x)=52cos(x)
    ampsin(x)=52sin(x)
    println("\tmethod \tdata type \tintegrand evaluation counts")
    for mthd in all_methods
        for T in supported_types[mthd]
            print("\t$mthd \t$T \t")
            R=real(T)
            freqs=[one(R)]
            kwargs=(:xmin=>zero(R),:xmax=>17*R(pi)/4,:method=>mthd,:T=>R)
            
            exact=sin(R(13))
            result=fccquad_cc(ampsin,angle,freqs;kwargs...)
            print("$(result[2]) \t")
            @test isapprox(result[1][1,1],exact)
            
            result=fccquad_cs(ampcos,angle,freqs;kwargs...)
            print("$(result[2]) \t")
            @test isapprox(result[1][1,1],exact)
            
            exact=one(R)-cos(R(13))
            result=fccquad_sc(ampsin,angle,freqs;kwargs...)
            print("$(result[2]) \t")
            @test isapprox(result[1][1,1],exact)

            result=fccquad_ss(ampcos,angle,freqs;kwargs...)
            println("$(result[2])")
            @test isapprox(result[1][1,1],exact)
        end
    end
end

#slow type I DCT (for testing purposes):
#(dct(x_0,...,x_N))_n = sum_{0<=m<=N}x_m*cos(pi*m*n/N)
#with 1/2 weight given to first and last summands.
#N can be any positive integer
function dct(x::AbstractVector,T::Type=Complex{Float64})::Vector{T} #type I DCT
    n=length(x)-1
    y=Vector{T}(undef,n+1)
    p = real(T)(pi)
    for i in 0:n
        y[i+1]=0.5x[1]
        @simd for j in 1:n-1
            @inbounds y[i+1]+=x[j+1]*cos(p*i*j/n)
        end
        y[i+1]+=x[n+1]*ifelse(i%2==0,0.5,-0.5)
    end
    y
end

@testset "Comparing fast (I)DCT to simpler implementation" begin
    x0=rand(Complex{Float64},129)
    y1=Fct.fct(x0)
    y2=dct(x0)
    @test isapprox(y1,y2)
    x1=Fct.idct(y1)
    @test isapprox(x0,x1)
end

@testset "Chebyshev series evaluation, dilation, and multiplication" begin
    f=Dict(:cube=>x->x^3, :sin=>sin, :cos=>cos, :exp=>exp)
    c=Dict(:cube=>Cheb.OddChebSeries(x->x^3),
           :sin=>Cheb.OddChebSeries(sin),
           :cos=>Cheb.EvenChebSeries(cos),
           :exp=>Cheb.ChebSeries(exp))
    x0=-0.5
    kappa=0.9
    for k in keys(f)
        @test isapprox(Cheb.chebinterp(c[k],x0),f[k](x0))
        @test isapprox(Cheb.chebinterp(Cheb.dilate(c[k],kappa),x0),f[k](kappa*x0))
        for j in keys(f)
            @test isapprox(Cheb.chebinterp(c[j]*c[k],x0),f[j](x0)*f[k](x0))
        end
    end
end

#T_n(x)
function chebpoly(n::Integer,x)
    @assert n>=0
    n<=1 && return iszero(n) ? one(x) : x
    m = div(n,2)
    iseven(n) && return 2chebpoly(m,x)^2 - one(x) #T_{2m}(x)
    a,b = chebpair(m,x) #T_m(x), T_{m+1}(x)
    2a*b - x #T_{2m+1}(x)
end

#T_n(x),T_{n+1}(x)
function chebpair(n::Integer,x)
    @assert n>=0
    n==0 && return (one(x),x)
    m = div(n,2)
    e,o = chebpair(m,x) #T_m, T_{m+1}
    if iseven(n) #n == 2m
        2e^2 - one(x), 2e*o - x #T_{2m}, T_{2m+1}
    else # n == 2m+1
        2o*e - x, 2o^2 - one(x) #T_{2m+1}, T_{2m+2}        
    end
end

@testset "Chebyshev weights" begin
    function checkweights(N,w,args...)
        a = getweights(N,w,args...)
        atol2 = 2N*eps()^2
        gk_atol=eps()
        gk_rtol=10^4 * eps()
        for n in 0:2:N
        gk,delta = collect(quadgk(x->chebpoly(n,x)*cos(w*x),-1,1,rtol=gk_rtol,atol=gk_atol))
            pass = abs2(a[n+1]-gk) <= max(atol2,abs2(delta))
            pass || println("N=$N,n=$n,omega=$w,weight=$(a[n+1]),gk=$gk,gk_delta=$delta")
            @test pass
        end
        for n in 1:2:N
            gk,delta = collect(quadgk(x->chebpoly(n,x)*sin(w*x),-1,1,rtol=gk_rtol,atol=gk_atol))
            pass = abs2(a[n+1]-gk) <= max(atol2,abs2(delta))
            pass || println("N=$N,n=$n,omega=$w,weight=$(a[n+1]),gk=$gk,gk_delta=$delta")
            @test pass
        end
    end
    println("Testing Chebyshev weights (0<=n<=100):")
    for freq in [0,-1e-9,1e-6,-1e-3,1e-2,-1e-1,1,2,3,4,5,-99.5,200]
        println("\ttesting angular frequency $freq...")
        checkweights(100,freq)
    end
end
