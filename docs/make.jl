using Documenter
using FCCQuad

makedocs(
    sitename = "FCCQuad.jl",
    authors = "David Milovich (@dkm2) and contributors",
    modules = [FCCQuad, FCCQuad.Jets],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/dkm2/FCCQuad.jl.git",
    devbranch = "master",
)
