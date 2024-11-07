#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
    Pkg.instantiate()
end

using Documenter
using Documenter.Remotes
using Finch

DocMeta.setdocmeta!(Finch, :DocTestSetup, :(using Finch; using SparseArrays); recursive=true)

makedocs(;
    modules=[Finch],
    authors="Willow Ahrens",
    repo=Remotes.GitHub("finch-tensor", "Finch.jl"),
    sitename="Finch.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://finch-tensor.github.io/Finch.jl",
        assets=["assets/favicon.ico"],
        size_threshold = 1_000_000,
    ),
    pages=[
        "Home" => "index.md",
        #"Getting Started" => "getting_started.md",
        #"Practical Tutorials and Use Cases" => "tutorials_use_cases/tutorials_use_cases.md",
        "Comprehensive Guides" => [
            "Calling Finch" => "guides/calling_finch.md",
            "Tensor Formats" => "guides/tensor_formats.md",
            "Sparse and Structured Utilities" => "guides/sparse_utils.md",
            "The Finch Language" => "guides/finch_language.md",
            "Dimensionalization" => "guides/dimensionalization.md",
            #"Tensor Lifecycles" => "guides/tensor_lifecycles.md",
            "Index Sugar" => "guides/index_sugar.md",
            "Mask Sugar" => "guides/mask_sugar.md",
            "Iteration Protocols" => "guides/iteration_protocols.md",
            "User-Defined Functions" => "guides/user-defined_functions.md",
            "High-Level Array API" => "guides/array_api.md",
            "Parallelization" => "guides/parallelization.md",
            "FileIO" => "guides/fileio.md",
            "Interoperability" => "guides/interoperability.md",
            "Optimization Tips" => "guides/optimization_tips.md",
            "Benchmarking Tips" => "guides/benchmarking_tips.md",
            #"Debugging Tips" => "guides/debugging_tips.md",
        ],
        "Technical Reference" => [
            "Documentation Listing" => "reference/listing.md",
            "Advanced Internal Details" => [
                "Virtualization" => "reference/internals/virtualization.md",
                "Tensor Interface" => "reference/internals/tensor_interface.md",
                "Compiler Interfaces" => "reference/internals/compiler_interface.md",
                "Finch Notation" => "reference/internals/finch_notation.md",
                "Finch Logic" => "reference/internals/finch_logic.md",
        #        "Looplets and Coiteration" => "reference/internals/looplets_coiteration.md",
            ],
        ],
        "Community and Contributions" => "CONTRIBUTING.md",
        "Appendices and Additional Resources" => [
            #"Glossary" => "appendices/glossary.md",
            #"FAQs" => "appendices/faqs.md",
            "Directory Structure" => "appendices/directory_structure.md",
            #"Changelog" => "appendices/changelog.md",
            #"Publications and Articles" => "appendices/publications_articles.md",
        ],
    ],
    warnonly=[:missing_docs],
)

deploydocs(;
    repo="github.com/finch-tensor/Finch.jl",
    devbranch="main",
)
