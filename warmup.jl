import Pkg
Pkg.activate(".")

"""
PDMO.jl Setup
=============

This script sets up PDMO.jl with all required dependencies.

FOR USERS WITH HSL ACCESS:
1. Obtain HSL library from https://www.hsl.rl.ac.uk/
2. Set up HSL_jll following the structure below
3. Uncomment and modify the HSL_PATH below BEFORE running this script
4. Run: julia warmup.jl

HSL_jll Directory Structure:
    HSL_jll/
    └── override/
        └── lib/
            └── {SYSTEM_ARCHITECTURE}/
                └── libhsl.{so|dylib}

Example paths:
- Linux: "/home/username/HSL_jll"  
- macOS: "/Users/username/HSL_jll"

FOR USERS WITHOUT HSL:
Simply run: julia warmup.jl (no modifications needed)
"""

# =============================================================================
# HSL SETUP SECTION - Uncomment and modify path if you have HSL
# =============================================================================

# Default: Use placeholder (no HSL)
HSL_PATH = "HSL_jll_placeholder"

# If you have HSL: Uncomment and update the path below
# HSL_PATH = "/PATH/TO/HSL_jll"

# =============================================================================

println("Setting up HSL_jll...")
Pkg.develop(path = HSL_PATH)
println("✅ HSL_jll development package added from: $HSL_PATH")

# Now instantiate and check status
Pkg.instantiate()
Pkg.status()

# Basic setup
println("Setting up PDMO.jl...")
using PDMO
println("PDMO.jl setup complete!")

# Report HSL status
if PDMO.HSL_FOUND
    println("✅ HSL detected and properly configured")
    println("🚀 Ipopt will use HSL linear solvers for enhanced performance")
else
    println("ℹ️  HSL not detected - using Ipopt's default linear solver")
    println("📝 This is fine for most use cases")
    println("")
    println("To add HSL support:")
    println("1. Obtain HSL from https://www.hsl.rl.ac.uk/")
    println("2. Uncomment and modify the HSL_PATH in this script")
    println("3. Run this script again")
end

println("\n" * "="^70)
println("📝 WORKFLOW:")
println("• HSL_jll is now a regular dependency")
println("• Default: Uses HSL_jll_placeholder (HSL disabled)")
println("• Users with HSL: Uncomment HSL_PATH and set your path")
println("• No manual Project.toml modifications needed!")
println("="^70)