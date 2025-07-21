"""
    MultiblockProblem

A container for a multiblock optimization problem. This structure maintains a collection of block variables
and a collection of block constraints.
 
# Fields
- `blocks::Vector{BlockVariable}`: A vector of block variables.
- `constraints::Vector{BlockConstraint}`: A vector of block constraints.
 
A default constructor is provided that initializes both collections as empty.
"""
mutable struct MultiblockProblem
    blocks::Vector{BlockVariable}
    constraints::Vector{BlockConstraint}
    MultiblockProblem() = new(Vector{BlockVariable}(), Vector{BlockConstraint}())
end 

"""
    addBlockVariable!(mbp::MultiblockProblem, block::BlockVariable)

Add a block variable to the multiblock problem. If the block has a default ID, a new unique ID is assigned.
Otherwise, the ID is checked for uniqueness.
"""
function addBlockVariable!(mbp::MultiblockProblem, block::BlockVariable)
    if isa(block.id, String) && isempty(block.id) # the block has a default ID = ""
        baseId = "Block"
        newId = baseId * string(length(mbp.blocks))
        while any(b -> b.id == newId, mbp.blocks)
            newId = baseId * string(parse(Int, match(r"\d+", newId).match) + 1)
        end
        block.id = newId
        @warn("MultiblockProblem: a block with default ID is added; assigned ID = $newId")
    else # otherwise, check if the ID already exists
         for b in mbp.blocks
            if b.id == block.id
                error("MultiblockProblem: Block with ID '$(block.id)' already exists.")
            end
        end
    end
    push!(mbp.blocks, block)
    return block.id
end 

"""
    addBlockConstraint!(mbp::MultiblockProblem, constraint::BlockConstraint)

Add a block constraint to the multiblock problem. If the constraint has a default ID, a new unique ID is assigned.
Otherwise, the ID is checked for uniqueness.
"""
function addBlockConstraint!(mbp::MultiblockProblem, constraint::BlockConstraint)
    if isa(constraint.id, String) && isempty(constraint.id) # the constraint has a default ID = ""
        baseId = "Constraint"
        newId = baseId * string(length(mbp.constraints))
        while any(c -> c.id == newId, mbp.constraints)
            newId = baseId * string(parse(Int, match(r"\d+", newId).match) + 1)
        end
        constraint.id = newId
        @warn("MultiblockProblem: a constraint with default ID is added; assigned ID = $newId")
    else # otherwise, check if the ID already exists
         for c in mbp.constraints
            if c.id == constraint.id
                error("MultiblockProblem: Constraint with ID '$(constraint.id)' already exists.")
            end
        end
    end
    push!(mbp.constraints, constraint)
    return constraint.id
end 


"""
    checkMultiblockProblemValidity(mbp::MultiblockProblem; addWrapper::Bool=true)

Check whether the given multiblock problem is valid.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to check.
- `addWrapper::Bool=true`: Flag to add wrapper for scalar input functions.

# Returns
- `Bool`: `true` if the problem is valid, `false` otherwise.

# Implementation Details
The function checks the validity of each block variable and constraint. It also attempts
to evaluate the primal residuals of each constraint with the initial solution to ensure
the problem is well-formed.
"""
function checkMultiblockProblemValidity(mbp::MultiblockProblem; addWrapper::Bool=true)
    for block in mbp.blocks 
        if checkBlockVariableValidity(block; addScalarFunctionWrapper=addWrapper) == false 
            return false
        end 
    end 

    # Create an initial solution: a vector of the x fields from each block.
    initialSol = Dict{BlockID, NumericVariable}(block.id => block.val for block in mbp.blocks)

    # Validate each block constraint.
    for constr in mbp.constraints 
        if isa(constr.rhs, Number) && addWrapper == true 
            constr.rhs = Float64[constr.rhs]
        end 

        if checkBlockConstraintValidity(constr) == false 
            return false 
        end 

        try 
            pres = blockConstraintViolation(constr, initialSol)
        catch err  
            println("MultiblockProblem: error encountered while evaluating primal residuals of constraint $(constr.id): error = $err")
            return false 
        end
    end 

    return true 
end

"""
    checkMultiblockProblemFeasibility(mbp::MultiblockProblem)

Check the feasibility of the current solution in the multiblock problem.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to check.

# Returns
- `Float64`: The L2 norm of the constraint violations.
- `Float64`: The infinity norm of the constraint violations.

# Implementation Details
The function computes the constraint violations for each constraint using the current
values in the block variables, and returns both the L2 and infinity norms of these violations.
"""
function checkMultiblockProblemFeasibility(mbp::MultiblockProblem, primalSol::Dict{BlockID, NumericVariable})
    presL2 = 0.0
    presLInf = 0.0
    for constr in mbp.constraints 
        res = blockConstraintViolation(constr, primalSol)
        presL2 += dot(res, res)
        presLInf = max(presLInf, norm(res, Inf))
    end 
    presL2 = sqrt(presL2)
    return presL2, presLInf
end 

function checkMultiblockProblemFeasibility(mbp::MultiblockProblem)
    sol = Dict{BlockID, NumericVariable}(block.id => block.val for block in mbp.blocks)
    return checkMultiblockProblemFeasibility(mbp, sol)
end 


"""
    summary(mbp::MultiblockProblem)

Print a summary of the multiblock problem.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to summarize.

# Output
Prints information about the problem including:
- Number of block variables
- Number of block constraints
- Range of number of blocks involved in constraints
"""
function summary(mbp::MultiblockProblem)
    @info "Summary of Multiblock Problem: "
    println("    Number of block variables   = $(length(mbp.blocks))")
    println("    Number of block constraints = $(length(mbp.constraints))")
    
    min_row_blocks = length(mbp.blocks)
    max_row_blocks = 0
    for constr in mbp.constraints 
        number_blocks = length(constr.involvedBlocks)
        if number_blocks < min_row_blocks
            min_row_blocks = number_blocks
        end 
        if number_blocks > max_row_blocks 
            max_row_blocks = number_blocks
        end 
    end     
    println("    Range of num. blocks in row = [$min_row_blocks, $max_row_blocks]")
end 


""" 
    checkCompositeProblemValidity!(mbp::MultiblockProblem)

The input is a mbp with (p+1) blocks: 

    min sum_{i=1}^p (f_i(x_i) + g_i(x_i)) + g_{p+1}(x_{p+1})
    s.t. A1x1 + A2x2 + ... + Apxp - x_{p+1} = 0

The function checks that if the input mbp is of the form described above. If x_{p+1} is 
    identified, this block wil be moved to the end of mbp.blocks. 

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to check.

# Returns
- `Bool`: `true` if the problem is a valid composite form, `false` otherwise.

"""
function checkCompositeProblemValidity!(mbp::MultiblockProblem)
    if length(mbp.constraints) != 1
        println("checkCompositeProblemValidity: number of constraints must be 1")
        return false 
    end 

    constraint = mbp.constraints[1]

    if norm(constraint.rhs, Inf) > ZeroTolerance 
        println("checkCompositeProblemValidity: right-hand side of the constraint must be 0")
        return false 
    end 

    potentialProximalOnlyBlocks = BlockID[]
    for block in mbp.blocks
        mapping = constraint.mappings[block.id] 
        if isa(mapping, LinearMappingIdentity) && mapping.coe == -1.0 && isa(block.f, Zero) 
            push!(potentialProximalOnlyBlocks, block.id)
        end 

        if isa(block.f, WrapperScalarInputFunction)
            if isSmooth(block.f) == false || isConvex(block.f) == false ||
               isProximal(block.g) == false || isConvex(block.g) == false
                println("checkCompositeProblemValidity: block $(block.id) is not valid for AdaPDM.")
                return false 
            end 
        else 
            if isSmooth(typeof(block.f)) == false || isConvex(typeof(block.f)) == false ||
               isProximal(typeof(block.g)) == false || isConvex(typeof(block.g)) == false
                println("checkCompositeProblemValidity: block $(block.id) is not valid for AdaPDM.")
                return false 
            end 
        end 
    end 

    if isempty(potentialProximalOnlyBlocks)
        println("checkCompositeProblemValidity: no proximal-only block found")
        return false 
    end 

    proximalOnlyBlockID = pop!(potentialProximalOnlyBlocks) 
   
    idxToMove = findfirst(block -> block.id == proximalOnlyBlockID, mbp.blocks)
    blockToMove = mbp.blocks[idxToMove]
   
    deleteat!(mbp.blocks, idxToMove)
    push!(mbp.blocks, blockToMove)
    
    return true 
end 



