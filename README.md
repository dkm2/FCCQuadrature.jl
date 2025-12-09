# FCCQuadrature.jl

[![CI](https://github.com/dkm2/FCCQuad.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/dkm2/FCCQuad.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://dkm2.github.io/FCCQuad.jl/dev/)

`FCCQuadrature.jl` is a Julia implementation of variants of Filon-Clenshaw-Curtis (FCC) quadrature.

## Complex finite Fourier integrals

`fccquad` computes complex finite Fourier integrals of the form

$$\int_a^b f(x)g(x)e^{i\omega x}dx$$

where $g$ is nonzero and
$\omega$ ranges over a finite set.
It is intended for $f$,
$|g|$, and
$d(\arg(g(x)))/dx$ not highly oscillatory.
(But $g$ may be highly oscillatory.)

## Real finite Fourier integrals

`fccquad_cc` computes real finite Fourier integrals of the form

$$\int_a^b f(x)\cos(g(x))\cos(\omega x)dx$$

where $\omega$ ranges over a finite set.
It is intended for $f$ and
$g'$ not highly oscillatory.
(But $\cos(g)$ may be highly oscillatory.)

`fccquad_cs`, `fccquad_sc`, and `fccquad_ss`
replace one or both of the cosines in the above integrand with sines.

## `method` keyword argument

Specify a FCC quadrature variant with the `method` keyword argument.
(Default: `:tone`.)

* Method `:nonadaptive` is FCC quadrature with the base-2 logarithm of
  the interpolation degree specified by `nonadaptivelog2degree` (default: 10).

* Method `:degree` is a degree-adaptive FCC quadrature with the base-2 logarithm of
  the interpolation degree starting at `minlog2degree` (default: 3)
  and adaptively increasing but not beyond `globalmaxlog2degree` (default: 20).

* Methods `:plain`, `:tone`, and `:chirp` are 
  hybrid interval-adaptive degree-adaptive FCC quadrature variants
  with the base-2 logarithm of the per-subinterval interpolation degree
  starting at `minlog2degree` (default: 3) and adaptively increasing
  but not beyond `localmaxlog2degree` (default: 6).
  If a subinterval's accuracy goals are not met even with the maximum degree,
  then the subinterval is divided into `branching`-many equal subintervals
  (default: 4).

* Methods `:tone` and `:chirp` use automatic differentiation
  to factor out a per-subinterval tone or chirp (respectively)
  before the polynomial interpolation step. To use these methods,
  the function $g$ must have a generic implementation that accepts
  an input whose type is an arbitrary `Number` subtype.

Method | Supported Working Precisions
--- | ---
`:nonadaptive`, `:degree` | `Float64`, `BigFloat(precision=256)`
`:plain`, `:tone` | `Float32`, `Float64`, `BigFloat(precision=256)`
`:chirp` | `Float64`

## Algorithms

All methods involve evaluating a function $h\colon[-1,1]\to\mathbb{C}$
at $N+1$ Chebyshev nodes
$\cos(\pi k/N)$ for
$0\leq k\leq N$
and computing the coefficients of the Chebyshev interpolant
$\sum_{n\leq N} a_n T_n(x)$ in time
$O(N\log N)$ using an FFT.
Given $\omega$, the fundamental integrals
$\int_{-1}^1T_n(x)e^{i\omega x}dx$
are evaluated for all $n\leq N$ using method "RR" of (Evans and Webster, 1999),
which has time complexity $O(N)$ uniformly with respect to
$\omega$.

The `:tone` and `:chirp` methods compute Taylor coefficients
of $\arg(h)$ at
$0$ and then factor out a corresponding
tone $e^{i\nu x}$ or chirp
$e^{i\nu x+i\mu x^2}$
from $h$ before sampling at the Chebyshev nodes.
To compensate, the $e^{i\nu x}$ factor is absorbed into
$e^{i\omega x}$.

The `:chirp` method compensates for the $e^{i\mu x^2}$ factor
using dilation and multiplication of Chebyshev series that come
from interpolating $h$ and from precomputed interpolations
of a fixed set of chirps. (The `:chirp` method will divide
the domain of integration into subintervals and recurse if
$|\mu|$ is too large for this fixed set.)
Because of its complexity, the `:chirp` method is only recommended
for integrands that are extremely expensive to sample.

For more information, see the preprint:

[Filon-Clenshaw-Curtis Quadrature with Automatic Tone Removal](https://dkmj.org/academic/numfour.pdf)

