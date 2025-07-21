"""
    MultiblockProblemScaling.jl

Logarithmic scaling utilities for multiblock optimization problems.

This module provides functionality to:
- Analyze constraints with mixed mapping types (Matrix/Identity/Extraction)
- Scale block variables and constraint rows using logarithmic scaling
- Handle partial scaling where only some variables/rows can be scaled

# Scaling Rules
For constraint A1*x1 + A2*x2 + ... + An*xn = b:
- If ALL Ai are LinearMappingMatrix: Can scale variables AND rows
- If ANY Ai is LinearMappingIdentity/LinearMappingExtraction: 
  * Can scale xj only for j where Aj is LinearMappingMatrix
  * CANNOT scale rows (keep row scaling = Identity)
- Block variables use single scalar per block (geometric mean of column scalars within constraint, geometric mean across constraints)
- Row scaling applied independently per constraint
- Globally scalable blocks use the same scaling factor across all constraints
"""

const log2_val = log(2.0)
const inv_log2_val = 1.0 / log2_val

"""
    ScalingStrategy

Enum for different scaling strategies with varying levels of aggressiveness.

# Options
- `CONSERVATIVE`: Powers of 2 with well-conditioned checks, minimum aggregation (most conservative)
- `MODERATE`: Exact scaling factors with well-conditioned checks, geometric mean aggregation (balanced)
- `AGGRESSIVE`: Exact scaling factors without well-conditioned checks, maximum aggregation (most aggressive)

# Notes
- Row scalars are computed consistently across all strategies using powers of 2 for numerical stability
- Only column scalar aggregation varies between strategies (min/geomean/max)
- Aggregation strategies only affect how column scalars are combined into block scalars
"""
@enum ScalingStrategy begin
    CONSERVATIVE = 1
    MODERATE = 2
    AGGRESSIVE = 3
end

"""
    ScalingOptions

Configuration options for scaling strategies.

# Fields
- `strategy::ScalingStrategy`: The scaling strategy to use
- `maxScalarBound::Float64`: Maximum allowed scaling factor (default: 1000.0)
- `minScalarBound::Float64`: Minimum allowed scaling factor (default: 1/1000.0)
- `wellConditionedMin::Float64`: Minimum value for well-conditioned check (default: 0.25)
- `wellConditionedMax::Float64`: Maximum value for well-conditioned check (default: 4.0)
- `skipWellConditionedCheck::Bool`: Whether to skip well-conditioned checks (default: false)
- `usePowersOfTwo::Bool`: Whether to round to powers of 2 (default: true)
- `minScalingThreshold::Float64`: Minimum scaling factor to apply (default: 1e-12)
"""
mutable struct ScalingOptions
    strategy::ScalingStrategy
    maxScalarBound::Float64
    minScalarBound::Float64
    wellConditionedMin::Float64
    wellConditionedMax::Float64
    skipWellConditionedCheck::Bool
    usePowersOfTwo::Bool
    minScalingThreshold::Float64
    
    function ScalingOptions(strategy::ScalingStrategy = CONSERVATIVE)
        opts = new()
        opts.strategy = strategy
        
        # Set defaults based on strategy
        if strategy == CONSERVATIVE
            opts.maxScalarBound = 1000.0
            opts.minScalarBound = 1.0/1000.0
            opts.wellConditionedMin = 0.25
            opts.wellConditionedMax = 4.0
            opts.skipWellConditionedCheck = false
            opts.usePowersOfTwo = true
            opts.minScalingThreshold = 1e-12
        elseif strategy == MODERATE
            opts.maxScalarBound = 1000.0
            opts.minScalarBound = 1.0/1000.0
            opts.wellConditionedMin = 0.1
            opts.wellConditionedMax = 10.0
            opts.skipWellConditionedCheck = false
            opts.usePowersOfTwo = true
            opts.minScalingThreshold = 1e-10
        elseif strategy == AGGRESSIVE
            opts.maxScalarBound = 1e6
            opts.minScalarBound = 1.0/1e6
            opts.wellConditionedMin = 0.001
            opts.wellConditionedMax = 1000.0
            opts.skipWellConditionedCheck = true
            opts.usePowersOfTwo = false
            opts.minScalingThreshold = 1e-6
        end
        
        return opts
    end
end

"""
    ConstraintScalingAnalysis

Detailed analysis of a single constraint's scaling possibilities.

# Fields
- `constraintID::BlockID`: ID of the constraint
- `canScaleRows::Bool`: Whether rows can be scaled (true only if all mappings are Matrix)
- `nonScalableBlocks::Vector{BlockID}`: Block variables that cannot be scaled
- `effectiveMatrix::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Nothing}`: Combined constraint matrix
- `matrixBlocks::Vector{BlockID}`: Blocks with LinearMappingMatrix mappings (in order)
"""
mutable struct ConstraintScalingAnalysis
    constraintID::BlockID
    canScaleRows::Bool
    nonScalableBlocks::Vector{BlockID}
    matrixBlocks::Vector{BlockID}
    effectiveMatrix::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int64}, Nothing}
    
    function ConstraintScalingAnalysis(constraintID::BlockID)
        new(constraintID, 
        false, 
        BlockID[],  
        BlockID[], 
        nothing)
    end
end

function ConstraintScalingAnalysis(constraint::BlockConstraint)
    analysis = ConstraintScalingAnalysis(constraint.id)
    
    # Analyze each mapping in the constraint
    for (blockID, mapping) in constraint.mappings
        if isa(mapping, LinearMappingMatrix)
            push!(analysis.matrixBlocks, blockID)
        else 
            push!(analysis.nonScalableBlocks, blockID)
        end
    end
    
    # Can only scale rows if ALL mappings are Matrix
    analysis.canScaleRows = isempty(analysis.nonScalableBlocks)
    
    # Construct effective constraint matrix for Matrix mappings only
    if isempty(analysis.matrixBlocks) == false
        matrices = [] 
        for blockID in analysis.matrixBlocks  
            push!(matrices, constraint.mappings[blockID].A)
        end
        analysis.effectiveMatrix = sparse(hcat(matrices...))
    end
    
    return analysis
end

"""
    MultiblockScalingInfo

Container for scaling information for multiblock problems.

# Fields
- `blockScalings::Dict{BlockID, Float64}`: Scaling factors for each block variable
- `constraintRowScalings::Dict{BlockID, Vector{Float64}}`: Row scaling factors for each constraint
- `constraintAnalyses::Dict{BlockID, ConstraintScalingAnalysis}`: Analysis per constraint
- `globallyScalableBlocks::Set{BlockID}`: Blocks that can be scaled across all constraints
- `scalableConstraints::Set{BlockID}`: Constraints that support row scaling
- `maxIterations::Int`: Maximum iterations for logarithmic scaling
- `tolerance::Float64`: Convergence tolerance
- `scalingOptions::ScalingOptions`: Configuration options for scaling strategy
- `originalFunctions::Dict{BlockID, Tuple{AbstractFunction, AbstractFunction}}`: Original block functions for unscaling
- `originalMappings::Dict{BlockID, Dict{BlockID, AbstractMapping}}`: Original constraint mappings for unscaling
- `originalRHS::Dict{BlockID, NumericVariable}`: Original RHS values for unscaling
- `originalBlockValues::Dict{BlockID, NumericVariable}`: Original block values for unscaling
- `isScaled::Bool`: Whether scaling has been applied
"""
mutable struct MultiblockScalingInfo
    blockScalings::Dict{BlockID, Float64}
    constraintRowScalings::Dict{BlockID, Vector{Float64}}
    constraintAnalyses::Dict{BlockID, ConstraintScalingAnalysis}
    globallyScalableBlocks::Set{BlockID}
    scalableConstraints::Set{BlockID}

    # Scaling parameters
    maxIterations::Int
    tolerance::Float64
    scalingOptions::ScalingOptions
    
    # Original data for unscaling
    originalFunctions::Dict{BlockID, Tuple{AbstractFunction, AbstractFunction}}
    originalMappings::Dict{BlockID, Dict{BlockID, AbstractMapping}}
    originalRHS::Dict{BlockID, NumericVariable}
    originalBlockValues::Dict{BlockID, NumericVariable}
    
    isScaled::Bool
    
    function MultiblockScalingInfo(maxIter::Int=10, tol::Float64=1e-8, options::ScalingOptions=ScalingOptions())
        info = new()
        info.blockScalings = Dict{BlockID, Float64}()
        info.constraintRowScalings = Dict{BlockID, Vector{Float64}}()
        info.constraintAnalyses = Dict{BlockID, ConstraintScalingAnalysis}()
        info.globallyScalableBlocks = Set{BlockID}()
        info.scalableConstraints = Set{BlockID}()
        info.maxIterations = maxIter
        info.tolerance = tol
        info.scalingOptions = options
        info.originalFunctions = Dict{BlockID, Tuple{AbstractFunction, AbstractFunction}}()
        info.originalMappings = Dict{BlockID, Dict{BlockID, AbstractMapping}}()
        info.originalRHS = Dict{BlockID, NumericVariable}()
        info.originalBlockValues = Dict{BlockID, NumericVariable}()
        info.isScaled = false
        return info
    end
end



"""
    MultiblockScalingInfo(mbp::MultiblockProblem; maxIter::Int=10, tol::Float64=1e-8, options::ScalingOptions=ScalingOptions()) -> MultiblockScalingInfo

Constructor that analyzes a multiblock problem for scaling opportunities and creates scaling information.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to analyze
- `maxIter::Int=10`: Maximum iterations for logarithmic scaling algorithm
- `tol::Float64=1e-8`: Convergence tolerance for scaling iterations
- `options::ScalingOptions=ScalingOptions()`: Scaling strategy configuration

# Returns
- `MultiblockScalingInfo`: Object containing scaling analysis results and configuration

# Analysis Process
1. **Constraint Analysis**: Examines each constraint's mapping types to identify scalable blocks
2. **Global Scalability**: Determines blocks that are scalable across ALL constraints
3. **Row Scaling**: Identifies constraints that support row/constraint scaling
4. **Scaling Configuration**: Sets up parameters for logarithmic scaling algorithm

# Notes
- Only blocks that are scalable in every constraint are considered globally scalable
- The analysis prepares the problem for logarithmic scaling with bounded scaling factors
- Scalable constraints are tracked separately for row scaling operations
"""
function MultiblockScalingInfo(mbp::MultiblockProblem; maxIter::Int=20, tol::Float64=1e-8, options::ScalingOptions=ScalingOptions())
    info = MultiblockScalingInfo(maxIter, tol, options)
   
    info.globallyScalableBlocks = Set{BlockID}(block.id for block in mbp.blocks)
    
    for constraint in mbp.constraints
        analysis = ConstraintScalingAnalysis(constraint)
        info.constraintAnalyses[constraint.id] = analysis
        
        # A block is globally scalable only if it's scalable in ALL constraints
        # Use setdiff! for bulk removal instead of individual deletions - O(n) vs O(n²)
        setdiff!(info.globallyScalableBlocks, analysis.nonScalableBlocks)

        if analysis.canScaleRows 
            push!(info.scalableConstraints, constraint.id)
        end 
    end
    return info
end

"""
    applyScaling!(mbp::MultiblockProblem, info::MultiblockScalingInfo)

Apply logarithmic scaling to the multiblock problem.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem
- `info::MultiblockScalingInfo`: Scaling information (modified in-place)

# Implementation
Applies logarithmic scaling algorithm (iterative row/column equilibration using log space).
"""
function applyScaling!(mbp::MultiblockProblem, info::MultiblockScalingInfo)
    
    # Initialize scaling factors
    for block in mbp.blocks
        info.blockScalings[block.id] = 1.0
    end
    
    for constraint in mbp.constraints
        rhs_size = isa(constraint.rhs, Number) ? 1 : length(constraint.rhs)
        info.constraintRowScalings[constraint.id] = ones(Float64, rhs_size)
    end
    
    try
        applyLogScaling!(mbp, info)
    catch e
        @error "applyScaling!: Error applying logarithmic scaling: $(e)"
        rethrow(e)
    end
end

"""
    applyLogScaling!(mbp::MultiblockProblem, info::MultiblockScalingInfo)

Apply logarithmic scaling algorithm to optimize condition numbers.

# Implementation
This implements the logarithmic scaling algorithm that:
1. Checks if matrix elements are already well-conditioned (between 0.25 and 4)
2. Uses logarithmic computations for numerical stability
3. Iteratively computes row and column scaling factors
4. Applies bounded scaling factors using powers of 2

This is a matrix-free implementation that works constraint by constraint.
"""
function applyLogScaling!(mbp::MultiblockProblem, info::MultiblockScalingInfo)
    # @info "Applying logarithmic scaling (max $(info.maxIterations) iterations)..."
    
    # Pre-compute constants based on scaling options
    opts = info.scalingOptions
    max_scalar = opts.maxScalarBound
    min_scalar = opts.minScalarBound
    bound = log(max_scalar) * inv_log2_val
    tolerance_threshold = opts.minScalingThreshold
    well_conditioned_min = opts.wellConditionedMin
    well_conditioned_max = opts.wellConditionedMax
    
    # Step 1: Compute geometric mean scaling factor for each (block, constraint) pair
    block_constraint_scalars = Dict{Tuple{BlockID, BlockID}, Float64}()
    constraint_row_scalars = Dict{BlockID, Vector{Float64}}()
    
    # Process each constraint to compute scaling factors
    for constraint in mbp.constraints
        analysis = info.constraintAnalyses[constraint.id]
        
        if analysis.effectiveMatrix === nothing || isempty(analysis.matrixBlocks)
            continue
        end
        
        A = analysis.effectiveMatrix  # Keep as sparse matrix
        m, n = size(A)
        
        # Get sparse matrix structure: row indices, column indices, values
        rows, cols, vals = findnz(A)
        nnz_count = length(vals)
        
        # Optimization: Combined filtering and min/max computation in single pass
        # Pre-allocate with estimated size to avoid repeated allocations
        valid_indices = Vector{Int}()
        sizehint!(valid_indices, nnz_count)  # Worst case: all elements are valid
        
        min_element = Inf
        max_element = 0.0
        
        for k in 1:nnz_count
            abs_val = abs(vals[k])
            if abs_val > tolerance_threshold
                push!(valid_indices, k)
                min_element = min(min_element, abs_val)
                max_element = max(max_element, abs_val)
            end
        end
        
        # Handle edge case: no valid elements
        if isempty(valid_indices)
            @info "Constraint $(constraint.id) has no valid elements (all below threshold), skipping scaling"
            continue
        end
        
        # Check if matrix is well-conditioned (valid min/max values) - only if not disabled
        if !opts.skipWellConditionedCheck && min_element < Inf && max_element > 0.0 && 
           min_element >= well_conditioned_min && max_element <= well_conditioned_max
            # @info "Constraint $(constraint.id) is already well-conditioned (min: $min_element, max: $max_element), skipping scaling"
            continue
        end
        
        # Optimization: Pre-allocate arrays and use more efficient data structures
        col_counts = zeros(Int, n)
        row_counts = zeros(Int, m)
        
        # Count non-zeros per column and row (optimization: single pass)
        for k in valid_indices
            i, j = rows[k], cols[k]
            col_counts[j] += 1
            row_counts[i] += 1
        end
        
        # Check for empty rows/columns (defensive programming)
        empty_cols = count(==(0), col_counts)
        empty_rows = count(==(0), row_counts)
        if empty_cols > 0 || empty_rows > 0
            @warn "Constraint $(constraint.id) has $empty_cols empty columns and $empty_rows empty rows"
        end
        
        # Initialize scaling vectors (optimization: reuse vectors)
        c = zeros(Float64, n)   # Column scaling factors (log space)
        cp = zeros(Float64, n)  # Column sums of log elements
        r = zeros(Float64, m)   # Row scaling factors (log space)  
        rp = zeros(Float64, m)  # Row sums of log elements
        
        # Optimization: Vectorized initial sums computation
        for k in valid_indices
            i, j = rows[k], cols[k]
            log_val = log(abs(vals[k])) * inv_log2_val
            cp[j] += log_val
            rp[i] += log_val
        end
        
        # Optimization: Pre-allocate temporary arrays for eta calculations
        col_eta = zeros(Float64, n)
        row_eta = zeros(Float64, m)
        
        # Iterative scaling with convergence check
        c_prev = zeros(Float64, n)
        r_prev = zeros(Float64, m)
        converged = false
        
        for iter in 1:info.maxIterations
            # @info "  Constraint $(constraint.id), iteration $iter"
            
            # Store previous values for convergence check (before update)
            copyto!(c_prev, c)
            copyto!(r_prev, r)
            
            # Optimization: Reset eta arrays instead of reallocating
            fill!(col_eta, 0.0)
            fill!(row_eta, 0.0)
            
            # Optimization: Combined computation of eta values
            for k in valid_indices
                i, j = rows[k], cols[k]
                col_eta[j] += r[i]
                row_eta[i] += c[j]
            end
            
            # Update column scaling factors (optimization: vectorized bounds)
            for j in 1:n
                if col_counts[j] > 0
                    eta = cp[j] + col_eta[j]
                    c[j] = clamp(-eta / col_counts[j], -bound, bound)
                end
            end
            
            # Update row scaling factors (optimization: vectorized bounds)
            for i in 1:m
                if row_counts[i] > 0
                    eta = rp[i] + row_eta[i]
                    r[i] = clamp(-eta / row_counts[i], -bound, bound)
                end
            end
            
            # Optimization: Early convergence check without temporary arrays
            if iter > 1
                c_diff = 0.0
                r_diff = 0.0
                
                # Manual max computation to avoid array allocation
                for j in 1:n
                    c_diff = max(c_diff, abs(c[j] - c_prev[j]))
                end
                for i in 1:m
                    r_diff = max(r_diff, abs(r[i] - r_prev[i]))
                end
                
                if c_diff < info.tolerance && r_diff < info.tolerance
                    # @info "  Converged after $iter iterations"
                    converged = true
                    break
                end
            end
        end
    
        
        # Convert scaling factors based on options
        # Row scalars use consistent approach regardless of strategy (powers of 2 for stability)
        row_scalars = [2.0^floor(r[i] + 0.5) for i in 1:m]
        
        # Column scalars follow the strategy-specific approach
        if opts.usePowersOfTwo
            # Round to nearest power of 2
            column_scalars = [2.0^floor(c[j] + 0.5) for j in 1:n]
        else
            # Use exact scaling factors with bounds
            column_scalars = [clamp(2.0^c[j], min_scalar, max_scalar) for j in 1:n]
        end
        
        # Store column scalars and compute geometric mean for each block in this constraint
        col_idx = 1
        for blockID in analysis.matrixBlocks
            mapping = constraint.mappings[blockID]
            block_cols = size(mapping.A, 2)
            
            # Extract column scalars for this specific block
            block_scalars = @view column_scalars[col_idx:(col_idx + block_cols - 1)]
            
            # Compute aggregation based on strategy
            if !isempty(block_scalars)
                if opts.strategy == CONSERVATIVE
                    # Use minimum for conservative scaling (most conservative)
                    min_scalar = Inf
                    for scalar in block_scalars
                        if scalar > 0.0  # Guard against non-positive values
                            min_scalar = min(min_scalar, scalar)
                        end
                    end
                    
                    if min_scalar < Inf
                        block_constraint_scalars[(blockID, constraint.id)] = min_scalar
                        # @info "  Block $(blockID) in constraint $(constraint.id): minimum = $(min_scalar)"
                    end
                elseif opts.strategy == MODERATE
                    # Use geometric mean for moderate scaling (balanced)
                # Geometric mean: exp(mean(log(x))) = (x1 * x2 * ... * xn)^(1/n)
                log_sum = 0.0
                valid_scalars = 0
                
                for scalar in block_scalars
                    if scalar > 0.0  # Guard against non-positive values
                        log_sum += log(scalar)
                        valid_scalars += 1
                    end
                end
                
                if valid_scalars > 0
                    geom_mean = exp(log_sum / valid_scalars)
                    block_constraint_scalars[(blockID, constraint.id)] = geom_mean
                    # @info "  Block $(blockID) in constraint $(constraint.id): geometric mean = $(geom_mean)"
                    end
                elseif opts.strategy == AGGRESSIVE
                    # Use maximum for aggressive scaling (most aggressive)
                    max_scalar = 0.0
                    for scalar in block_scalars
                        if scalar > 0.0  # Guard against non-positive values
                            max_scalar = max(max_scalar, scalar)
                        end
                    end
                    
                    if max_scalar > 0.0
                        block_constraint_scalars[(blockID, constraint.id)] = max_scalar
                        # @info "  Block $(blockID) in constraint $(constraint.id): maximum = $(max_scalar)"
                    end
                end
            end
            
            col_idx += block_cols
        end
        
        # Store row scalars for this constraint
        constraint_row_scalars[constraint.id] = row_scalars
        
        # @info "  Constraint $(constraint.id) scaling factors computed"
    end
    
    # Step 2: Compute global block scaling factors
    # if isempty(info.globallyScalableBlocks)
    #     @info "No globally scalable blocks found, skipping global scaling"
    # else
    #     @info "Computing global scaling factors for $(length(info.globallyScalableBlocks)) blocks"
    # end
    
    # Collect geometric means from each constraint and compute geometric mean
    for blockID in info.globallyScalableBlocks
        # Optimization: Use pre-allocated vector and avoid repeated allocations
        constraint_means = Vector{Float64}()
        sizehint!(constraint_means, length(mbp.constraints))  # Hint expected size
        
        # Collect geometric means from all constraints where this block appears
        for constraint in mbp.constraints
            key = (blockID, constraint.id)
            if haskey(block_constraint_scalars, key)
                push!(constraint_means, block_constraint_scalars[key])
                # @info "  Block $(blockID): collected geometric mean $(block_constraint_scalars[key]) from constraint $(constraint.id)"
            end
        end
        
        # Compute aggregation of constraint means and convert to appropriate scaling factor
        if !isempty(constraint_means)
            if opts.strategy == CONSERVATIVE
                # Use minimum for conservative scaling (most conservative)
                min_mean = minimum(constraint_means)
                
                if opts.usePowersOfTwo
                    # Convert to power of 2 (round to nearest)
                    log_min_mean = log(min_mean) * inv_log2_val
                    block_scalar = 2.0^floor(log_min_mean + 0.5)
                else
                    # Use exact scaling factors with bounds
                    block_scalar = clamp(min_mean, min_scalar, max_scalar)
                end
                
                info.blockScalings[blockID] = block_scalar
                # @info "Block $(blockID) global scaling factor: $(block_scalar) (minimum of $(length(constraint_means)) constraint means)"
            elseif opts.strategy == MODERATE
                # Use geometric mean for moderate scaling (balanced)
            # Optimization: Fast geometric mean computation without intermediate allocation
            log_sum = 0.0
            valid_means = 0
            
            for mean_val in constraint_means
                if mean_val > 0.0  # Guard against non-positive values
                    log_sum += log(mean_val)
                    valid_means += 1
                end
            end
            
            if valid_means > 0
                geom_mean = exp(log_sum / valid_means)
                
                    if opts.usePowersOfTwo
                # Convert to power of 2 (round to nearest)
                log_geom_mean = log(geom_mean) * inv_log2_val
                block_scalar = 2.0^floor(log_geom_mean + 0.5)
                    else
                        # Use exact scaling factors with bounds
                        block_scalar = clamp(geom_mean, min_scalar, max_scalar)
                    end
                    
                info.blockScalings[blockID] = block_scalar
                    # @info "Block $(blockID) global scaling factor: $(block_scalar) (from geometric mean $(geom_mean) of $(valid_means) constraint means)"
                end
            elseif opts.strategy == AGGRESSIVE
                # Use maximum for aggressive scaling (most aggressive)
                max_mean = maximum(constraint_means)
                
                # Use exact scaling factors with bounds
                block_scalar = clamp(max_mean, min_scalar, max_scalar)
                info.blockScalings[blockID] = block_scalar
                # @info "Block $(blockID) global scaling factor: $(block_scalar) (maximum of $(length(constraint_means)) constraint means)"
            end
        end
    end
    
    # Step 3: Apply row scaling to constraints (if allowed)
    for constraint in mbp.constraints
        analysis = info.constraintAnalyses[constraint.id]
        
        if analysis.canScaleRows && haskey(constraint_row_scalars, constraint.id)
            row_scalars = constraint_row_scalars[constraint.id]
            current_row_scaling = info.constraintRowScalings[constraint.id]
            
            # Optimization: Vectorized multiplication instead of loop
            min_len = min(length(current_row_scaling), length(row_scalars))
            if min_len > 0
                @inbounds for i in 1:min_len
                    current_row_scaling[i] *= row_scalars[i]
                end
            end
            
            # @info "Applied row scaling to constraint $(constraint.id)"
        end
    end
    
    # @info "Logarithmic scaling completed with global block scaling"
end

"""
    allScalingFactorsAreOne(info::MultiblockScalingInfo) -> Bool

Check if all computed scaling factors are effectively 1.0 (no scaling needed).

# Arguments
- `info::MultiblockScalingInfo`: Scaling information with computed factors

# Returns
- `Bool`: true if all block scaling factors are 1.0, false otherwise

# Notes
Since scaling factors can be computed as powers of 2 or exact values, we check for exact equality to 1.0.
For aggressive strategies, we also check against the minimum scaling threshold.
"""
function allScalingFactorsAreOne(info::MultiblockScalingInfo)
    threshold = info.scalingOptions.minScalingThreshold
    
    # Check if any block has a scaling factor significantly different from 1.0
    for (blockID, scalingFactor) in info.blockScalings
        if abs(scalingFactor - 1.0) > threshold
            return false
        end
    end
    
    # Check if any constraint has row scaling factors significantly different from 1.0
    for (constraintID, rowScalings) in info.constraintRowScalings
        for rowScaling in rowScalings
            if abs(rowScaling - 1.0) > threshold
                return false
            end
        end
    end
    
    return true
end

"""
    getScalingStrategyInfo(strategy::ScalingStrategy) -> String

Get a description of the scaling strategy.

# Arguments
- `strategy::ScalingStrategy`: The scaling strategy

# Returns
- `String`: Description of the strategy characteristics
"""
function getScalingStrategyInfo(strategy::ScalingStrategy)
    if strategy == CONSERVATIVE
        return "CONSERVATIVE: Powers of 2 with well-conditioned checks, minimum aggregation (most conservative)"
    elseif strategy == MODERATE
        return "MODERATE: Exact scaling factors with relaxed well-conditioned checks, geometric mean aggregation (balanced)"
    elseif strategy == AGGRESSIVE
        return "AGGRESSIVE: Exact scaling factors without well-conditioned checks, maximum aggregation (most aggressive)"
    else
        return "UNKNOWN: Unrecognized scaling strategy"
    end
end

"""
    reportScalingResults(info::MultiblockScalingInfo)

Report scaling results with detailed information about applied scaling factors.

# Arguments
- `info::MultiblockScalingInfo`: Scaling information with computed factors
"""
function reportScalingResults(info::MultiblockScalingInfo)
    println("=== Scaling Results ===")
    println("Strategy: $(getScalingStrategyInfo(info.scalingOptions.strategy))")
    println("Globally scalable blocks: $(length(info.globallyScalableBlocks))")
    println("Scalable constraints: $(length(info.scalableConstraints))")
    
    # Report block scaling factors
    if !isempty(info.blockScalings)
        println("\nBlock scaling factors:")
        for (blockID, factor) in info.blockScalings
            if abs(factor - 1.0) > info.scalingOptions.minScalingThreshold
                println("  Block $blockID: $factor")
            end
        end
    end
    
    # Report constraint row scaling factors
    if !isempty(info.constraintRowScalings)
        println("\nConstraint row scaling factors:")
        for (constraintID, factors) in info.constraintRowScalings
            non_trivial = filter(f -> abs(f - 1.0) > info.scalingOptions.minScalingThreshold, factors)
            if !isempty(non_trivial)
                println("  Constraint $constraintID: $(length(non_trivial)) non-trivial factors")
                if length(non_trivial) <= 10  # Don't print too many
                    println("    Values: $non_trivial")
                end
            end
        end
    end
    
    if allScalingFactorsAreOne(info)
        println("\nResult: All scaling factors are effectively 1.0 - no scaling applied")
    else
        println("\nResult: Non-trivial scaling factors applied")
    end
end

# Convenience functions for creating scaling options
"""
    conservativeScaling() -> ScalingOptions

Create conservative scaling options (powers of 2, well-conditioned checks, minimum aggregation).
"""
conservativeScaling() = ScalingOptions(CONSERVATIVE)

"""
    moderateScaling() -> ScalingOptions

Create moderate scaling options (exact factors, relaxed checks, geometric mean aggregation).
"""
moderateScaling() = ScalingOptions(MODERATE)

"""
    aggressiveScaling() -> ScalingOptions

Create aggressive scaling options (exact factors, no well-conditioned checks, maximum aggregation).
"""
aggressiveScaling() = ScalingOptions(AGGRESSIVE)

"""
    customScaling(; strategy::ScalingStrategy = CONSERVATIVE,
                   maxBound::Float64 = 1000.0,
                   minBound::Float64 = 1.0/1000.0,
                   skipWellConditioned::Bool = false,
                   usePowersOfTwo::Bool = true,
                   minThreshold::Float64 = 1e-12) -> ScalingOptions

Create custom scaling options with specific parameters.

# Arguments
- `strategy::ScalingStrategy = CONSERVATIVE`: Base strategy to start from
- `maxBound::Float64 = 1000.0`: Maximum scaling factor bound
- `minBound::Float64 = 1.0/1000.0`: Minimum scaling factor bound
- `skipWellConditioned::Bool = false`: Whether to skip well-conditioned checks
- `usePowersOfTwo::Bool = true`: Whether to round to powers of 2
- `minThreshold::Float64 = 1e-12`: Minimum scaling threshold

# Returns
- `ScalingOptions`: Custom scaling configuration
"""
function customScaling(; strategy::ScalingStrategy = CONSERVATIVE,
                       maxBound::Float64 = 1000.0,
                       minBound::Float64 = 1.0/1000.0,
                       skipWellConditioned::Bool = false,
                       usePowersOfTwo::Bool = true,
                       minThreshold::Float64 = 1e-12)
    opts = ScalingOptions(strategy)
    opts.maxScalarBound = maxBound
    opts.minScalarBound = minBound
    opts.skipWellConditionedCheck = skipWellConditioned
    opts.usePowersOfTwo = usePowersOfTwo
    opts.minScalingThreshold = minThreshold
    return opts
end

"""
    scaleMultiblockProblem!(mbp::MultiblockProblem; 
                           maxIterations::Int=10,
                           tolerance::Float64=1e-8,
                           analyzeOnly::Bool=false,
                           scalingOptions::ScalingOptions=ScalingOptions()) -> MultiblockScalingInfo

Apply logarithmic scaling to multiblock problems.

# Arguments
- `mbp::MultiblockProblem`: The problem to scale
- `maxIterations::Int=10`: Maximum iterations for logarithmic scaling
- `tolerance::Float64=1e-8`: Convergence tolerance
- `analyzeOnly::Bool=false`: If true, only analyze without applying scaling
- `scalingOptions::ScalingOptions=ScalingOptions()`: Scaling strategy configuration

# Returns
- `MultiblockScalingInfo`: Complete scaling information

# Examples
```julia
# Apply conservative scaling (default)
info = scaleMultiblockProblem!(mbp)

# Apply aggressive scaling (with maximum aggregation)
info = scaleMultiblockProblem!(mbp, scalingOptions=ScalingOptions(AGGRESSIVE))

# Analyze only without applying
info = scaleMultiblockProblem!(mbp, analyzeOnly=true)
```
"""
function scaleMultiblockProblem!(mbp::MultiblockProblem; 
                                maxIterations::Int=20,
                                tolerance::Float64=1e-8,
                                analyzeOnly::Bool=false,
                                scalingOptions::ScalingOptions=ScalingOptions())
    
    @info "Starting multiblockProblem scaling analysis with $(scalingOptions.strategy) strategy..."                            
    startTime = time()

    info = MultiblockScalingInfo(mbp, maxIter=maxIterations, tol=tolerance, options=scalingOptions)
    
    while true 
        if analyzeOnly
            @info "Scaling analysis complete. Scaling not applied (analyzeOnly=true)."
            break 
        end 

        if isempty(info.scalableConstraints) && isempty(info.globallyScalableBlocks)
            @info "No scalable constraints or globally scalable blocks found. Scaling not applied."
            break 
        end 

        applyScaling!(mbp, info)
        
        if allScalingFactorsAreOne(info)
            @info "No scalable constraints and all scaling factors are 1.0. Scaling not applied."
            break 
        end
        
        applyScalingToProblem!(mbp, info)
        
        break 
    end

    msg = Printf.@sprintf("MultiblockProblem scaling analysis took %.2f seconds \n", time() - startTime)
    @info msg 
    
    # Report detailed scaling results
    # reportScalingResults(info)
    
    return info 
end

"""
    applyScalingToProblem!(mbp::MultiblockProblem, info::MultiblockScalingInfo)

Apply the computed scaling factors to the multiblock problem.

# Arguments
- `mbp::MultiblockProblem`: The problem to scale (modified in-place)
- `info::MultiblockScalingInfo`: Scaling information from optimization

# Effects
- Applies block variable scaling using WrapperScalingTranslationFunction
- Applies constraint row and column scaling according to enhanced rules
- Stores original data for unscaling
"""
function applyScalingToProblem!(mbp::MultiblockProblem, info::MultiblockScalingInfo)
    if info.isScaled
        @warn "applyScalingToProblem!: Problem is already scaled. "
        return
    end
    
    # @info "Applying optimized scaling to multiblock problem..."
    
    # Store original data
    for block in mbp.blocks
        info.originalFunctions[block.id] = (block.f, block.g)
        info.originalBlockValues[block.id] = copy(block.val)
    end
    
    for constraint in mbp.constraints
        info.originalMappings[constraint.id] = copy(constraint.mappings)
        info.originalRHS[constraint.id] = copy(constraint.rhs)
    end
    
    # Scale block variables and 
    minColScalar = 1.0
    maxColScalar = 1.0 
    for block in mbp.blocks
        if block.id in info.globallyScalableBlocks
            α = info.blockScalings[block.id]
            
            # Scale functions: f'(x') = f(x'/α) where x' = α*x
            block.f = WrapperScalingTranslationFunction(block.f, 1.0/α, zero(block.val))
            block.g = WrapperScalingTranslationFunction(block.g, 1.0/α, zero(block.val))
            
            # Scale initial values: x' = α * x
            block.val = α * block.val
            
            # @info "Scaled block $(block.id) with factor $(α)"
            minColScalar = min(minColScalar, α)
            maxColScalar = max(maxColScalar, α)
        end 
    end
    
    # Scale constraints according to enhanced rules
    minRowScalar = 1.0
    maxRowScalar = 1.0 
    for constraint in mbp.constraints
        analysis = info.constraintAnalyses[constraint.id]
        
        # Apply scaling to mappings
        for (blockID, mapping) in constraint.mappings
            if isa(mapping, LinearMappingMatrix)
                # Get scaling factors
                α = info.blockScalings[blockID]
                
                if analysis.canScaleRows
                    # Both row and column scaling: A' = D * A / α
                    D = Diagonal(info.constraintRowScalings[constraint.id])
                    scaled_matrix = D * mapping.A / α

                    minRowScalar = min(minRowScalar, minimum(info.constraintRowScalings[constraint.id]))
                    maxRowScalar = max(maxRowScalar, maximum(info.constraintRowScalings[constraint.id]))
                else
                    # Only column scaling: A' = A / α
                    scaled_matrix = mapping.A / α
                end
                
                constraint.mappings[blockID] = LinearMappingMatrix(scaled_matrix)
            end
            # LinearMappingIdentity and LinearMappingExtraction remain unchanged
        end
        
        # Scale RHS if rows can be scaled
        if analysis.canScaleRows
            if !isa(constraint.rhs, Number)
                D = Diagonal(info.constraintRowScalings[constraint.id])
                constraint.rhs = D * constraint.rhs
            else
                constraint.rhs = info.constraintRowScalings[constraint.id][1] * constraint.rhs
            end
            
            # @info "Applied row and column scaling to constraint $(constraint.id)"
        else
            # @info "Applied column-only scaling to constraint $(constraint.id)"
        end
    end
    
    info.isScaled = true
    @info "Scaling applied successfully! block scaling: [$(minColScalar), $(maxColScalar)], constraint scaling: [$(minRowScalar), $(maxRowScalar)]"
    # @info "applyScalingToProblem!: Scaling applied successfully!"
end

"""
    unscaleMultiblockProblem!(mbp::MultiblockProblem, info::MultiblockScalingInfo)

Remove scaling transformations and restore the original multiblock problem.

# Arguments
- `mbp::MultiblockProblem`: The scaled problem (modified in-place)
- `info::MultiblockScalingInfo`: Scaling information used to reverse transformations

# Effects
- Restores original block functions
- Restores original constraint mappings and right-hand sides
- Unscales block variable values to original space
- Resets scaling flag

# Mathematical Transformations
For scaled block i with scaling factor αᵢ:
- Variables: xᵢ = x'ᵢ / αᵢ (reverse the scaling)
"""
function unscaleMultiblockProblem!(mbp::MultiblockProblem, primalSol::Dict{BlockID, NumericVariable}, info::MultiblockScalingInfo)
    if info.isScaled == false 
        @warn "Problem is not scaled. Nothing to unscale."
        return
    end
    
    @info "Removing scaling from multiblock problem..."
    
    # Restore original functions
    for block in mbp.blocks
        if haskey(info.originalFunctions, block.id)
            block.f, block.g = info.originalFunctions[block.id]
        end
    end
    
    # Restore original constraint mappings and RHS
    for constraint in mbp.constraints
        if haskey(info.originalMappings, constraint.id)
            constraint.mappings = info.originalMappings[constraint.id]
        end
        if haskey(info.originalRHS, constraint.id)
            constraint.rhs = info.originalRHS[constraint.id]
        end
    end
    
    # Unscale variable values: x = x' / α
    for block in mbp.blocks
        if block.id in info.globallyScalableBlocks
            α = info.blockScalings[block.id]
            primalSol[block.id] ./= α
            # @info "Unscaled block $(block.id) with factor $(1.0/α)"
        end
    end
    
    info.isScaled = false
    @info "unscaleMultiblockProblem!: Scaling removed successfully!"
end

# Export main types and functions
export ScalingStrategy, CONSERVATIVE, MODERATE, AGGRESSIVE
export ScalingOptions, MultiblockScalingInfo, ConstraintScalingAnalysis
export scaleMultiblockProblem!, unscaleMultiblockProblem!
export conservativeScaling, moderateScaling, aggressiveScaling, customScaling
export reportScalingResults, getScalingStrategyInfo, allScalingFactorsAreOne

# Usage Examples:
#
# Basic Usage:
# info = scaleMultiblockProblem!(mbp)  # Conservative scaling (minimum aggregation)
# info = scaleMultiblockProblem!(mbp, scalingOptions=moderateScaling())  # Moderate scaling (geometric mean)
# info = scaleMultiblockProblem!(mbp, scalingOptions=aggressiveScaling())  # Aggressive scaling (maximum aggregation)
#
# Custom Scaling Configuration:
# opts = customScaling(strategy=AGGRESSIVE, maxBound=1e4, minBound=1e-4, 
#                      skipWellConditioned=true, usePowersOfTwo=false, minThreshold=1e-6)
# info = scaleMultiblockProblem!(mbp, scalingOptions=opts)
#
# Scaling Strategy Comparison:
# strategies = [("Conservative (Min)", conservativeScaling()), ("Moderate (GeoMean)", moderateScaling()),
#               ("Aggressive (Max)", aggressiveScaling())]
# for (name, opts) in strategies
#     println("\n=== $name Strategy ===")
#     mbp_copy = deepcopy(mbp)
#     info = scaleMultiblockProblem!(mbp_copy, scalingOptions=opts)
# end
#
# Analysis Only:
# info = scaleMultiblockProblem!(mbp, analyzeOnly=true, scalingOptions=aggressiveScaling())
# reportScalingResults(info)
