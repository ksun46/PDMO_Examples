import Pkg
Pkg.activate(".")

Pkg.instantiate()
Pkg.status()

# Change to full path to the HSL library  
Pkg.develop(path="../HSL_jll_placeholder")

# Change to full path to the PDMO repository
Pkg.develop(path = "../../PDMO_Examples")

include("src/include.jl")


