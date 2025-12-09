using Documenter
using FCCQuadrature

makedocs(
    sitename = "FCCQuadrature.jl",
    authors = "David Milovich (@dkm2) and contributors",
    modules = [FCCQuadrature, FCCQuadrature.Jets],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/dkm2/FCCQuad.jl.git",
    devbranch = "master",
)
