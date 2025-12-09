# FCCQuadrature.jl

Filon-Clenshaw-Curtis (FCC) quadrature for oscillatory integrals.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/dkm2/FCCQuad.jl")
```

## Quick Start

```julia
using FCCQuadrature

# Complex finite Fourier integral
f(x) = exp(-x^2)
g(x) = 1.0
freqs = [1.0, 2.0, 3.0]
result, evals = fccquad(f, g, freqs; xmin=0.0, xmax=10.0)
```

## API Reference

See [API Reference](@ref api) for full documentation.
