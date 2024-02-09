#=
Fast Type-I Discrete Cosine Transform and Chebyshev polynomial interpolation
=#
module Fct

#fast type I DCT:
#(fct(x_0,...,x_N))_n == sum_{0<=m<=N}x_m*cos(pi*m*n/N)
#with 1/2 weight given to first and last summands.
#WARNING: assumes N is power of 2.
function fct(x::AbstractVector,T::Type=Complex{Float64})::AbstractVector{T}
    fct!(x,fct_alloc(x,T)...)
end
function fct_alloc(x::AbstractVector,T::Type=Complex{Float64})
    N = length(x)-1
    twoN = 2N
    Vector{T}(undef,twoN), Vector{T}(undef,twoN), Vector{T}(undef,twoN)
end
function fct!(dct_in::AbstractVector, #input
              ifft_in::AbstractVector, #workspace: ifft input
              ifft_out::AbstractVector, #output is view(ifft_out,1:N+1)
              ifft_work::AbstractVector, #workspace: ifft workspace
              N=length(dct_in)-1)
    ifft_in[1] = dct_in[1]
    for i in 2:N
        ifft_in[i] = dct_in[i]
    end
    ifft_in[N+1] = dct_in[N+1]
    for i in 1:N-1
        ifft_in[N+1+i] = dct_in[N+1-i]
    end
    ifft!(ifft_in,ifft_out,ifft_work,2N)
    dct_out = view(ifft_out,1:N+1)
    dct_out .*= 0.5
end

#inverse of fct and dct
function idct(x::AbstractVector,T::Type=Complex{Float64})::AbstractVector{T}
    idct!(x,fct_alloc(x,T)...)
end
function idct!(idct_in::AbstractVector, #input
               ifft_in::AbstractVector, #workspace: ifft input
               ifft_out::AbstractVector, #output is view(ifft_out,1:N+1)
               ifft_work::AbstractVector, #workspace: ifft workspace
               n=length(idct_in)-1)
    idct_out = fct!(idct_in,ifft_in,ifft_out,ifft_work,n)
    T = eltype(ifft_work)
    r = real(T)(2)/n
    idct_out .*= r
end

ifft_cache=Dict{Pair{Int64,Type},Vector}()

#fast inverse DFT without 1/N factor
#WARNING: assumes N is power of 2.
#(ifft(x_0,...,x_{N-1}))_n = sum_{0<=m<N}x_m*exp(2*pi*m*n/N)
function ifft(x::AbstractVector,T::Type=Complex{Float64})
    N=length(x)
    y=Vector{T}(undef,N) #output
    z=Vector{T}(undef,N) #work space
    ifft!(x,y,z)
end
function ifft!(input::AbstractVector,output::AbstractVector,workspace::AbstractVector,
               N=length(input))
    T=eltype(workspace)
    key = N=>T
    if haskey(ifft_cache,key)
        exps = ifft_cache[key]
    else
        ifft_cache[key] = exps = exp_populate(N,T)
    end
    ifft_helper!(input,output,workspace,exps,0,1,N)
end

#x is input; y is output; z is work space
#assumes length(x)==length(y)==length(exps)<=length(z)
#input/output indices: 1+offset,1+offset+stride,...,1+offset+(len-1)*stride
function ifft_helper!(x::AbstractVector,
                      y::AbstractVector,
                      z::AbstractVector,
                      exps::AbstractVector,
                      offset::Integer,
                      stride::Integer,
                      len::Integer)
    if len==1
        y[1+offset]=x[1+offset]
        return y
    end
    doublestride=stride<<1
    halflen=len>>1
    ifft_helper!(x,y,z,exps,offset,doublestride,halflen)
    ifft_helper!(x,y,z,exps,offset+stride,doublestride,halflen)
    i=1
    j=1+offset
    k=1
    while k<=halflen
        z[k]=y[j]+y[j+stride]*exps[i]
        i+=stride
        j+=doublestride
        k+=1
    end
    j=1+offset
    while k<=len
        z[k]=y[j]+y[j+stride]*exps[i]
        i+=stride
        j+=doublestride
        k+=1
    end
    i=1+offset
    k=1
    while k<=len
        y[i]=z[k]
        i+=stride
        k+=1
    end
    y
end

#output: exps[1+m]==exp(2im*pi*m/N) for 0<=m<N
#N can be any positive integer.
function exp_populate(N::Integer,T::Type=Complex{Float64})
    exps=Vector{T}(undef,N)
    w = 2im*T(pi)/N
    for m in 0:N-1
        exps[m+1] = exp(w*m)
    end
    exps
end

#=
Chebyshev polynomial interpolation of f on [-1,1] intended for Clenshaw-Curtis quadrature.
On input y_0,...,y_N, outputs c_0,...,c_N such that y_n == sum_m c_m cos(m*n*pi/N).
y_m should be f(cos(m*pi/N)).
WARNING: N must be a power of 2.
=#
function chebcoeffs(samples::AbstractVector,T::Type=promote_type(Complex{Bool},eltype(samples)))
    chebcoeffs!(samples,fct_alloc(samples,T)...)
end
function chebcoeffs!(samples::AbstractVector,
                     ifft_in::AbstractVector,
                     ifft_out::AbstractVector,#output is view(ifft_out,1:length(samples))
                     ifft_work::AbstractVector,
                     n=length(samples)-1)
    coeffs = idct!(samples,ifft_in,ifft_out,ifft_work,n)
    coeffs[1] *= 0.5
    coeffs[end] *= 0.5
    coeffs
end

function chebsample(f,n::Integer,base=0,stride=1;T::Type=Complex{Float64})
    samples = Vector{T}(undef,length(base:stride:n))
    chebsample!(f,samples,n,base,stride)
end
function chebsample!(f,samples,n::Integer=length(samples)-1,base=0,stride=1)
    p = real(eltype(samples))(pi)
    i=1
    for j in base:stride:n
        samples[i] = f(cos(p*j/n))
        i += 1
    end
    samples
end

function shiftedchebsample(center,radius,f1,f2,n::Integer,base=0,stride=1;T::Type=Complex{Float64})
    samples = Vector{T}(undef,length(base:stride:n))
    shiftedchebsample!(center,radius,f1,f2,samples,n,base,stride)
end
function shiftedchebsample!(center,radius,f1,f2,samples,n::Integer=length(samples)-1,base=0,stride=1)
    p = real(eltype(samples))(pi)
    i=1
    for j in base:stride:n
        x = center + radius*cos(p*j/n)
        samples[i] = f1(x)*f2(x)
        i += 1
    end
    samples
end

function tonechebsample(cfreq,center,radius,f1,f2,n::Integer,base=0,stride=1;T::Type=Complex{Float64})
    samples = Vector{T}(undef,length(base:stride:n))
    tonechebsample!(cfreq,center,radius,f1,f2,samples,n,base,stride)
end
function tonechebsample!(cfreq,center,radius,f1,f2,samples,n::Integer=length(samples)-1,base=0,stride=1)
    p = real(eltype(samples))(pi)
    i=1
    for j in base:stride:n
        x = cos(p*j/n)
        y = center + radius*x
        z = x * -cfreq
        s,c = sincos(z)
        samples[i] = f1(y)*f2(y)*complex(c,s)
        i += 1
    end
    samples
end

function chirpchebsample(cfreq,chirp,center,radius,f1,f2,n::Integer,base=0,stride=1;T::Type=Complex{Float64})
    samples = Vector{T}(undef,length(base:stride:n))
    chirpchebsample!(cfreq,chirp,center,radius,f1,f2,samples,n,base,stride)
end
function chirpchebsample!(cfreq,chirp,center,radius,f1,f2,samples,n::Integer=length(samples)-1,base=0,stride=1)
    p = real(eltype(samples))(pi)
    i=1
    for j in base:stride:n
        x = cos(p*j/n)
        y = center + radius*x
        z = -x*(cfreq + x*chirp)
        s,c = sincos(z)
        samples[i] = f1(y)*f2(y)*complex(c,s)
        i += 1
    end
    samples
end

#double Chebyshev degree
function doublesample(f,oldsamples)
    N = length(oldsamples)-1
    T = eltype(oldsamples)
    newsamples = chebsample(f,2N,1,2;T=T)
    samples = Vector{T}(undef,2N+1)
    samples[1] = oldsamples[1]
    for n in 1:N
        samples[2n] = newsamples[n]
        samples[2n+1] = oldsamples[n+1]
    end
    samples
end

function shifteddoublesample(center,radius,f1,f2,oldsamples)
    N = length(oldsamples)-1
    T = eltype(oldsamples)
    newsamples = shiftedchebsample(center,radius,f1,f2,2N,1,2;T=T)
    samples = Vector{T}(undef,2N+1)
    samples[1] = oldsamples[1]
    for n in 1:N
        samples[2n] = newsamples[n]
        samples[2n+1] = oldsamples[n+1]
    end
    samples
end

function chirpdoublesample(cfreq,chirp,center,radius,f1,f2,oldsamples)
    N = length(oldsamples)-1
    T = eltype(oldsamples)
    newsamples = chirpchebsample(cfreq,chirp,center,radius,f1,f2,2N,1,2;T=T)
    samples = Vector{T}(undef,2N+1)
    samples[1] = oldsamples[1]
    for n in 1:N
        samples[2n] = newsamples[n]
        samples[2n+1] = oldsamples[n+1]
    end
    samples
end

function doublesample!(f,samples,N)
    for n in N:-1:1
        samples[2n+1] = samples[n+1]
    end
    chebsample!(f,view(samples,2:2:2N),2N,1,2)
    samples
end

function shifteddoublesample!(center,radius,f1,f2,samples,N)
    for n in N:-1:1
        samples[2n+1] = samples[n+1]
    end
    shiftedchebsample!(center,radius,f1,f2,view(samples,2:2:2N),2N,1,2)
    samples
end

function tonedoublesample!(cfreq,center,radius,f1,f2,samples,N)
    for n in N:-1:1
        samples[2n+1] = samples[n+1]
    end
    tonechebsample!(cfreq,center,radius,f1,f2,view(samples,2:2:2N),2N,1,2)
    samples
end

end #module
