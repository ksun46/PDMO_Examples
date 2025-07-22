using Documenter
using PDMO  # Now safe to load with conditional dependencies

# Build documentation with automatic docstring extraction
makedocs(
    sitename = "PDMO.jl Documentation",
    authors = "PDMO.jl contributors",
    repo = Documenter.Remotes.GitHub("alibaba-damo-academy", "PDMO.jl"),
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://alibaba-damo-academy.github.io/PDMO.jl/",
        edit_link = "main",
        repolink = "https://github.com/alibaba-damo-academy/PDMO.jl"
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "S1_getting_started.md",
        "Algorithms" => [
            "AdaPDM" => "S2_algorithms/AdaPDM.md",
            "ADMM" => "S2_algorithms/ADMM.md"
        ],
        "Examples" => [
            "Distributed OPF" => "S3_examples/DistributedOPF.md",
            "Dual Lasso" => "S3_examples/DualLasso.md",
            "Dual SVM" => "S3_examples/DualSVM.md",
            "Fused Lasso" => "S3_examples/FusedLasso.md",
            "Least L1 Norm" => "S3_examples/LeastL1Norm.md"
        ],
        "API Reference" => [
            "Main" => "S4_api/main.md",
            "Formulations" => "S4_api/formulations.md",
            "Functions" => "S4_api/functions.md",
            "Mappings" => "S4_api/mappings.md", 
            "ADMM" => "S4_api/admm.md",
            "AdaPDM" => "S4_api/pdm.md"
        ]
    ]
)

# Deploy documentation to GitHub Pages
deploydocs(
    repo = "github.com/alibaba-damo-academy/PDMO.jl.git",
    push_preview = true,
    devbranch = "main"
) 