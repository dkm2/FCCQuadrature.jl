#=
Lightweight implementation of forward-mode second-degree autodiff.
=#
module Jets

import SpecialFunctions: erf, erfc, erfcinv, dawson
import Base: convert, promote_rule, +, -, *, /, inv, cos, sin, tan, cot, sec, csc, exp, cosh, sinh, tanh, coth, sech, csch, log, sqrt, one, zero, isnan, isinf, conj, real, imag

const erfinf=2/sqrt(pi)
const ierfinf=sqrt(pi)/2

"""
    Jet{T<:Number} <: Number

Second-degree Taylor polynomial for forward-mode automatic differentiation.

Represents `a + b*ε + c*ε²` where `ε³ = 0`.

# Fields
- `a::T`: value at evaluation point
- `b::T`: first derivative coefficient
- `c::T`: second derivative coefficient divided by 2

# Example
```julia
julia> f(x) = sin(x)
julia> j = f(FCCQuad.Jets.Jet(0.0, 1.0))  # evaluate at x=0 with dx=1
julia> j.a  # f(0) = 0
julia> j.b  # f'(0) = 1
```
"""
struct Jet{T<:Number} <: Number
  a::T
  b::T
  c::T
end

Standard=Union{Real,Complex{<:Real}}

Jet(x::T) where T<:Number = Jet{T}(x, zero(T), zero(T))

function Jet(x::T1,y::T2) where {T1<:Number,T2<:Number}
    T = promote_type(T1,T2)
    Jet{T}(promote(x,y)..., zero(T))
end

function Jet(x::T1,y::T2,z::T3) where {T1<:Number,T2<:Number,T3<:Number}
    T = promote_type(T1,T2,T3)
    Jet{T}(promote(x,y,z)...)
end

convert(::Type{Jet},x::Standard)=Jet(x)
convert(::Type{Jet{T}},x::Standard) where T<:Number = Jet(convert(T,x)) 
convert(T::Type{<:Standard},x::Jet)=convert(T,x.a)

promote_rule(::Type{Jet{T1}}, ::Type{T2}) where {T1<:Number,T2<:Standard} = Jet{promote_type(T1,T2)}
promote_rule(::Type{Jet{T1}}, ::Type{Jet{T2}}) where {T1<:Number,T2<:Number} = Jet{promote_type(T1,T2)}

function (+)(x::Jet,y::Jet)
  Jet(x.a+y.a,x.b+y.b,x.c+y.c)
end

function (-)(x::Jet,y::Jet)
  Jet(x.a-y.a,x.b-y.b,x.c-y.c)
end

function (-)(y::Jet)
  Jet(-y.a,-y.b,-y.c)
end

function (+)(x::Jet,y::Standard)
  Jet(x.a+y,x.b,x.c)
end

function (-)(x::Jet,y::Standard)
  Jet(x.a-y,x.b,x.c)
end

function (+)(x::Standard,y::Jet)
  Jet(x+y.a,y.b,y.c)
end

function (-)(x::Standard,y::Jet)
  Jet(x-y.a,-y.b,-y.c)
end

function (*)(x::Standard,y::Jet)
  Jet(x*y.a,x*y.b,x*y.c)
end

function (*)(y::Jet,x::Standard)
  Jet(y.a*x,y.b*x,y.c*x)
end

function (/)(x::Jet,y::Standard)
  x*inv(y)
end

function (*)(x::Jet,y::Jet)
  Jet(x.a*y.a,x.a*y.b+x.b*y.a,x.a*y.c+x.b*y.b+x.c*y.a)
end

#x=z/y only if x*y=z
function (/)(z::Jet,y::Jet)
  r=inv(y.a)
  xa=r*z.a
  xb=r*(z.b-xa*y.b)
  xc=r*(z.c-xa*y.c-xb*y.b)
  Jet(xa,xb,xc)
end

function inv(y::Jet)
  xa=inv(y.a)
  xb=-xa*xa*y.b
  xc=-xa*(xa*y.c + xb*y.b)
  Jet(xa,xb,xc)  
end

function (/)(z::Standard,y::Jet)
  r=inv(y.a)
  xa=r*z
  xb=-r*xa*y.b
  xc=-r*(xa*y.c + xb*y.b)
  Jet(xa,xb,xc)  
end

#f(a+e)=f(a)+ef'(a)+e^2f''(a)/2
function taylorcoeffs(func, pt::Standard)
  func(Jet(pt,one(pt)))
end

function taylorapprox(taylor::Jet,x::Standard)
  taylor.a+x*(taylor.b+x*taylor.c)
end

#computes f(x::Jet) from x and
#f,df,ddf=f(x.a),f'(x.a),f''(x.a)
function derivschain(x::Jet,
                     f::Standard,
                     df::Standard,
                     ddf::Standard)
  #f(a+be+ce^2)=f(a)+bef'(a)+e^2(cf'(a)+b^2f''(a)/2)
  Jet(f, x.b*df, x.c*df + x.b*x.b*ddf/2)
end

cos(x::Jet,si::Standard,co::Standard) = Jet(co, -x.b*si, -x.c*si - x.b*x.b*co/2)
sin(x::Jet,si::Standard,co::Standard) = Jet(si, x.b*co, x.c*co - x.b*x.b*si/2)
tan(x::Jet,si::Standard,co::Standard) = sin(x,si,co) / cos(x,si,co)
cot(x::Jet,si::Standard,co::Standard) = cos(x,si,co) / sin(x,si,co)
sec(x::Jet,si::Standard,co::Standard) = inv(cos(x,si,co))
csc(x::Jet,si::Standard,co::Standard) = inv(sin(x,si,co))

cos(x::Jet)=cos(x,sincos(x.a)...)
sin(x::Jet)=sin(x,sincos(x.a)...)
tan(x::Jet)=tan(x,sincos(x.a)...)
cot(x::Jet)=cot(x,sincos(x.a)...)
sec(x::Jet)=sec(x,sincos(x.a)...)
csc(x::Jet)=csc(x,sincos(x.a)...)

function exp(x::Jet)
  ex=exp(x.a)
  Jet(ex, ex*x.b, ex*(x.c + x.b*x.b/2))
end

function log(x::Jet)
    f=log(x.a)
    df=inv(x.a)
    ddf=-df^2
    derivschain(x,f,df,ddf)
end

function sqrt(x::Jet)
  f=sqrt(x.a)
  r=inv(2x.a)
  df=f*r
  ddf=-df*r
  derivschain(x,f,df,ddf)
end

function dawson(x::Jet)
  f=dawson(x.a)
  df=1-2*x.a*f
  ddf=2*(f*(2*x.a*x.a-1)-x.a)
  derivschain(x,f,df,ddf)
end

function erf(x::Jet)
  f=erf(x.a)
  df=erfinf*exp(-x.a*x.a)
  ddf=-2*x.a*df
  derivschain(x,f,df,ddf)
end

function erfc(x::Jet)
  f=erfc(x.a)
  df=-erfinf*exp(-x.a*x.a)
  ddf=-2*x.a*df
  derivschain(x,f,df,ddf)
end

function erfcinv(x::Jet)
  #erfcinv only defined on (0,2)
  f=erfcinv(real(x.a))
  df=-ierfinf*exp(f*f)
  ddf=2*f*df*df
  derivschain(x,f,df,ddf)
end

function cosh(x::Jet)
    p=exp(x)
    m=inv(p)
    (p+m)/2
end
function sinh(x::Jet)
    p=exp(x)
    m=inv(p)
    (p-m)/2
end
function tanh(x::Jet)
    p=exp(x)
    m=inv(p)
    (p-m)/(p+m)
end
function coth(x::Jet)
    p=exp(x)
    m=inv(p)
    (p+m)/(p-m)
end
function sech(x::Jet)
    p=exp(x)
    m=inv(p)
    2/(p+m)
end
function csch(x::Jet)
    p=exp(x)
    m=inv(p)
    2/(p-m)
end

function real(x::Jet)
  Jet(real(x.a),real(x.b),real(x.c))
end

function imag(x::Jet)
  Jet(imag(x.a),imag(x.b),imag(x.c))
end

function conj(x::Jet)
  Jet(conj(x.a),conj(x.b),conj(x.c))
end

one(::Jet{T}) where T<:Number = Jet(one(T))
zero(::Jet{T}) where T<:Number = Jet(zero(T))

isnan(x::Jet)=isnan(x.a) || isnan(x.b) || isnan(x.c)
isinf(x::Jet)=isinf(x.a) || isinf(x.b) || isinf(x.c)


#input: 2nd-order Taylor polynomial of function f at some point
#output: quadratic and linear coefficients of Taylor polynomial of arg(f) at that point
phasepoly(z::Jet)=complex(0.,phase_acceleration(z)),complex(0.,phase_velocity(z))

phase_velocity(z::Jet)=imag(z.b/z.a)
phase_acceleration(z::Jet)=imag((z.c*z.a-0.5*z.b*z.b)/(z.a*z.a))

end #module
