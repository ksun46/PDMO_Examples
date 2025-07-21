"""
    BlockVariable

A container representing a block variable in a larger optimization problem.

# Fields
- `id::Union{Int, String}`: A unique identifier for the block. This can be either an integer or a string.
- `f::AbstractFunction`: A function that should be smooth.
- `g::AbstractFunction`: A function that should be proximal.
- `val::NumericVariable`: The variable associated with this block.

A default constructor is provided with `BlockVariable(idx::Int64=0)`, which initializes `f` and `g` with `Zero()` 
and `val` with `0.0`. A block is considered uninitialized when `block_idx ≤ 0`.
"""
const BlockID = Union{Int64, String}

mutable struct BlockVariable 
    id::BlockID
    f::AbstractFunction
    g::AbstractFunction
    val::NumericVariable # variable shape --> numeric variables; variable numerics  
end 

"""
    BlockVariable(id::BlockID="")

Construct a new block variable with the specified ID.

# Arguments
- `id::BlockID=""`: The identifier for the block. Can be an integer or string. Default is an empty string.

# Returns
- `BlockVariable`: A new BlockVariable instance with the given ID and default Zero functions.

# Throws
- `ErrorException`: If `id` is a negative integer.

# Notes
- If `id` is an empty string, a warning is issued that it might be overwritten later.
- The block is initialized with `Zero()` functions for both `f` and `g`, and `0.0` for `val`.
"""
function BlockVariable(id::BlockID="")
    if isa(id, Int64) && id < 0
        error("BlockVariable: Block ID cannot be a negative integer.")
    end 
    if isa(id, String) && isempty(id)
        @warn("BlockVariable: initialized with default ID; this ID might be overwritten later.")
    end 
    return BlockVariable(id, Zero(), Zero(), 0.0)
end 

"""
    addWrapperForScalarInputFunction(block::BlockVariable)

Wrap scalar functions with `WrapperScalarInputFunction` for proper handling.

This function checks if the block's value is a scalar number. If so, it:
1. Wraps both `f` and `g` functions with `WrapperScalarInputFunction`
2. Converts the scalar value to a single-element Float64 array
3. Prints information about the conversion
""" 
function addWrapperForScalarInputFunction(block::BlockVariable)
    if isa(block.val, Number)
        if isa(block.f, Zero) == false 
            block.f = WrapperScalarInputFunction(block.f)
        end 

        if isa(block.g, Zero) || 
            isa(block.g, IndicatorNonnegativeOrthant) || 
            isa(block.g, IndicatorBallL2)
            # do nothing 
        elseif isa(block.g, IndicatorBox)
            block.g = IndicatorBox(ones(1) * block.g.lb, ones(1) * block.g.ub)
        else 
            block.g = WrapperScalarInputFunction(block.g)
        end 
        block.val = Float64[block.val]
        println("BlockVariable: added wrappers for scalar input functions to block $(block.id)")
    end 
end 


"""
    checkBlockVariableValidity(block::BlockVariable; addScalarFunctionWrapper::Bool=true) -> Bool

Check if a block variable is valid for use in optimization problems.

This function performs several validation steps:
1. Checks if the block ID is properly initialized (not negative for integer IDs)
2. Verifies that `f` is a smooth function
3. Verifies that `g` is a proximal function
4. Optionally adds scalar function wrappers if needed
5. Tests function evaluations, gradients, and proximal operators

# Arguments
- `block::BlockVariable`: The block variable to validate
- `addScalarFunctionWrapper::Bool=true`: Whether to automatically add scalar function wrappers if needed

# Returns
- `Bool`: `true` if the block is valid, `false` otherwise

# Notes
- Errors encountered during function evaluations are caught and reported, returning `false`
- Detailed error messages are printed to help diagnose issues
"""
function checkBlockVariableValidity(block::BlockVariable; addScalarFunctionWrapper::Bool=true)
    if (isa(block.id, Number) && block.id < 0)
        println("BlockVariable: block $(block.id) is not initialized.") 
        return false 
    end 
   
    # if (typeof(block.f) == WrapperScalarInputFunction && isSmooth(block.f) == false || 
    #     typeof(block.f) != WrapperScalarInputFunction && isSmooth(typeof(block.f)) == false)   
    if isSmooth(block.f) == false 
        println("BlockVariable: f of block $(block.id) is not smooth.") 
        return false 
    end 
    
    # if (typeof(block.g) == WrapperScalarInputFunction && isProximal(block.g) == false || 
    #     typeof(block.g) != WrapperScalarInputFunction && isProximal(typeof(block.g)) == false)
    if isProximal(block.g) == false 
        println("BlockVariable: g of block $(block.id) is not proximal.") 
        return false 
    end 

    if addScalarFunctionWrapper    
        addWrapperForScalarInputFunction(block)
    end 
    
    try
        val = (block.f)(block.val)
    catch error
        println("BlockVariable: Error enountered while evaluating funciton value of f of block $(block.id); error = $error") 
        return false 
    end
    
    try
        grad = gradientOracle(block.f, block.val)
    catch error
        println("BlockVariable: Error enountered while evaluating gradient oracle of f of block $(block.id); error = $error") 
        return false 
    end

    try 
        val = (block.g)(block.val)
    catch error
        println("BlockVariable: Error enountered while evaluating function value of g of block $(block.id); error =$error") 
        return false; 
    end 
    
    try 
        prox = proximalOracle(block.g, block.val, 1.0)
    catch error
        println("BlockVariable: Error enountered while evaluating proximal oracle of g of block $(block.id); error =$error") 
        return false; 
    end 

    return true 
end 

