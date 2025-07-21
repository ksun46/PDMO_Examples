using Documenter, TikzPictures

# Load all required packages first
using LinearAlgebra, SparseArrays, Printf, Random, Logging, Dates
using JuMP, Arpack, DataFrames, DataStructures, JSON, MathOptInterface
using HiGHS, Ipopt, FileIO, FilePathsBase, Metis, CodecZlib
using PDMO

# Add the parent directory to the load path so Julia can find the PDMO package
push!(LOAD_PATH, "../")

# Try to load the PDMO package
try
    using PDMO
    @info "Successfully loaded PDMO package for automatic docstring extraction"
catch e
    @error "Failed to load PDMO package: $e"
    @info "Documentation will build with empty @docs blocks"
end

# Build documentation with automatic docstring extraction
makedocs(
    sitename = "PDMO.jl Documentation",
    authors = "PDMO.jl contributors",
    repo = "https://github.com/alibaba-damo-academy/primal-dual-methods-for-optimization.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://alibaba-damo-academy.github.io/primal-dual-methods-for-optimization.jl/",
        assets = ["assets/tikz-support.css"],
        edit_link = "main"
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "S1_getting_started.md",
        "Algorithms" => [
            "ADMM" => "S2_algorithms/ADMM.md",
            "AdaPDM" => "S2_algorithms/AdaPDM.md"
        ],
        "Examples" => [
            "Least L1 Norm" => "S3_examples/LeastL1Norm.md",
            "Fused Lasso" => "S3_examples/FusedLasso.md",
            "Dual Lasso" => "S3_examples/DualLasso.md",
            "Dual SVM" => "S3_examples/DualSVM.md",
            "Distributed OPF" => "S3_examples/DistributedOPF.md"
        ],
        "API Reference" => [
            "Main Algorithm Interface" => "S4_api/main.md"
            , "Functions" => "S4_api/functions.md"
            , "Mappings" => "S4_api/mappings.md"
            , "Formulations" => "S4_api/formulations.md"
            , "ADMM Components" => "S4_api/admm.md"
            , "AdaPDM Components" => "S4_api/pdm.md"
            # , "Utilities" => "S4_api/utilities.md"
        ]
    ]
)

# Deploy documentation to GitHub Pages
deploydocs(
    repo = "github.com/alibaba-damo-academy/primal-dual-methods-for-optimization.jl.git",
    push_preview = true,
    devbranch = "main"
) 