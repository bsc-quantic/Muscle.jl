using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

push!(LOAD_PATH, "$(@__DIR__)/..")

using Muscle
using Documenter

DocMeta.setdocmeta!(Muscle, :DocTestSetup, :(using Muscle); recursive=true)

makedocs(;
    modules=[Muscle],
    authors="Sergio Sánchez Ramírez <sergio.sanchez.ramirez@bsc.es> and contributors",
    repo="https://github.com/bsc-quantic/Muscle.jl/blob/{commit}{path}#{line}",
    sitename="Muscle.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://bsc-quantic.github.io/Muscle.jl",
        edit_link="master",
        assets=String[]
    ),
    pages=[
        "Home" => "index.md",
    ]
)

deploydocs(;
    repo="github.com/bsc-quantic/Muscle.jl",
    devbranch="master"
)
