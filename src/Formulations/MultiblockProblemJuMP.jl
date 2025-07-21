"""
    isSupportedObjectiveFunction(f::AbstractFunction) -> Bool

Check if a function type is supported for JuMP objective formulation, including wrapped functions.
"""
function isSupportedObjectiveFunction(f::AbstractFunction)
    if typeof(f) in [QuadraticFunction, 
                     AffineFunction, 
                     ComponentwiseExponentialFunction, 
                     Zero, 
                     UserDefinedSmoothFunction]
        return true
    elseif typeof(f) == WrapperScalingTranslationFunction
        return isSupportedObjectiveFunction(f.originalFunction)
    elseif typeof(f) == WrapperScalarInputFunction  
        return isSupportedObjectiveFunction(f.originalFunction)
    else 
        return false
    end
end

"""
    isSupportedProximalFunction(g::AbstractFunction) -> Bool

Check if a function type is supported for JuMP constraint formulation, including wrapped functions.
"""
function isSupportedProximalFunction(g::AbstractFunction)
    if typeof(g) in [IndicatorBox, 
                     IndicatorBallL2, 
                     IndicatorSumOfNVariables, 
                     ElementwiseL1Norm, 
                     IndicatorHyperplane, 
                     IndicatorSOC, 
                     IndicatorRotatedSOC,
                     IndicatorNonnegativeOrthant,
                     Zero, 
                    #  UserDefinedProximalFunction
                     ]
        return true
    elseif typeof(g) == WrapperScalingTranslationFunction
        return isSupportedProximalFunction(g.originalFunction)
    # elseif typeof(g) == WrapperScalarInputFunction  
        # return isSupportedConstraintFunction(g.originalFunction)
    else 
        return false
    end
end

"""
    unwrapFunction(f::AbstractFunction) -> (originalFunction, scaling, translation)

Unwrap a function, returning the original function and transformation parameters.
For non-wrapped functions, returns (f, 1.0, 0.0).
"""
function unwrapFunction(f::AbstractFunction)
    if typeof(f) == WrapperScalingTranslationFunction
        @assert(isa(f.translation, Number) == false, "unwrapFunction: the unwrapped function cannot have scalar input.")
        # The unwrapped function cannot have scalar input, cause a scalar input function has already been wrapped into 
        # a WrapperScalarInputFunction before scaling and translation. 
        return (f.originalFunction, f.coe, f.translation)
    else
        return (f, 1.0, 0.0)
    end
end

"""
    addBlockVariableToJuMPModel!(model::JuMP.Model, 
                               f::AbstractFunction,
                               g::AbstractFunction,
                               x::NumericVariable,
                               blockID::BlockID, 
                               var::Union{Dict{BlockID, Vector{JuMP.VariableRef}}, Dict{String, Vector{JuMP.VariableRef}}}, 
                               aux::Union{Dict{BlockID, Vector{JuMP.VariableRef}}, Dict{String, Vector{JuMP.VariableRef}}}, 
                               objExpressions::Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}})

Add a block variable to a JuMP model.

# Arguments
- `model::JuMP.Model`: The JuMP model to which the variable will be added.
- `f::AbstractFunction`: The smooth function component of the block variable.
- `g::AbstractFunction`: The nonsmooth function component of the block variable.
- `x::NumericVariable`: The current value of the block variable.
- `blockID::BlockID`: The ID of the block variable.
- `var::Union{Dict{BlockID, Vector{JuMP.VariableRef}}, Dict{String, Vector{JuMP.VariableRef}}}`: Dictionary to store the created JuMP variables.
- `aux::Union{Dict{BlockID, Vector{JuMP.VariableRef}}, Dict{String, Vector{JuMP.VariableRef}}}`: Dictionary to store auxiliary JuMP variables.
- `objExpressions::Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}}`: Vector to collect objective expressions.

# Returns
- `Bool`: `true` if the block has a nonlinear smooth function, `false` otherwise.

# Implementation Details
The function creates JuMP variables and constraints based on the types of `f` and `g`. 
It supports various function types including QuadraticFunction, AffineFunction, 
ComponentwiseExponentialFunction for smooth functions, and IndicatorBox, IndicatorBallL2, 
IndicatorSumOfNVariables, ElementwiseL1Norm, IndicatorHyperplane, IndicatorSOC for nonsmooth functions.
""" 
function addBlockVariableToJuMPModel!(model::JuMP.Model, 
    f::AbstractFunction,
    g::AbstractFunction,
    x::NumericVariable,
    blockID::BlockID, 
    var::Union{Dict{BlockID, Vector{JuMP.VariableRef}}, Dict{String, Vector{JuMP.VariableRef}}}, 
    aux::Union{Dict{BlockID, Vector{JuMP.VariableRef}}, Dict{String, Vector{JuMP.VariableRef}}}, 
    objExpressions::Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}})

    @assert(length(size(x)) == 1, "addBlockVariableToJuMPModel!: only support vector variables for now.")

    # Check if function types are supported (including unwrapped WrapperScalingTranslationFunction)
    @assert(isSupportedObjectiveFunction(f), 
        "addBlockVariableToJuMPModel!: unsupported type of f = $(typeof(f))")

    @assert(isSupportedProximalFunction(g),   
        "addBlockVariableToJuMPModel!: unsupported type of g = $(typeof(g))")

    dim = length(x)
    
    # Unwrap WrapperScalingTranslationFunction if possible
    f_unwrapped, f_scaling, f_translation = unwrapFunction(f)
    g_unwrapped, g_scaling, g_translation = unwrapFunction(g)

    g_type = typeof(g_unwrapped)
    g_not_scaled = g_scaling == 1.0 && all(g_translation .== 0.0)

    if g_type == IndicatorBox
        # Handle scaling and translation: lb <= g_scaling*x + g_translation <= ub
        # Becomes: (lb - g_translation)/g_scaling <= x <= (ub - g_translation)/g_scaling
        if g_not_scaled
            var[blockID] = JuMP.@variable(model, [k = 1:dim], 
                lower_bound = g_unwrapped.lb[k], 
                upper_bound = g_unwrapped.ub[k])
        else
            # Vector translation
            var[blockID] = JuMP.@variable(model, [k = 1:dim], 
                lower_bound = (g_unwrapped.lb[k] - g_translation[k]) / g_scaling, 
                upper_bound = (g_unwrapped.ub[k] - g_translation[k]) / g_scaling)
        end
    elseif g_type == IndicatorNonnegativeOrthant
        # Handle scaling and translation: g_scaling*x + g_translation >= 0
        # Becomes: x >= -g_translation / g_scaling
        if g_not_scaled 
            # No transformation needed
            var[blockID] = JuMP.@variable(model, [k = 1:dim], lower_bound = 0.0)
        else
            var[blockID] = JuMP.@variable(model, [k = 1:dim], lower_bound = -g_translation[k] / g_scaling)
        end
    elseif g_type == IndicatorBallL2
        # Handle scaling and translation: ||g_scaling*x + g_translation||_2 <= r
        # Need to add auxiliary variables for the transformed variables
        r = g_unwrapped.r
        if g_not_scaled 
            # No transformation needed
            var[blockID] = JuMP.@variable(model, [k = 1:dim], lower_bound = -r, upper_bound = r)
            JuMP.@constraint(model, sum(var[blockID][k]^2 for k in 1:dim) <= r^2)
        else
            # Create transformed variables: y = g_scaling*x + g_translation
            var[blockID] = JuMP.@variable(model, [k = 1:dim], lower_bound = -Inf, upper_bound = Inf)
            aux[blockID] = JuMP.@variable(model, [k = 1:dim], lower_bound = -r, upper_bound = r)
            JuMP.@constraint(model, [k in 1:dim], aux[blockID][k] == g_scaling * var[blockID][k] + g_translation[k])
            JuMP.@constraint(model, sum(aux[blockID][k]^2 for k in 1:dim) <= r^2)
        end
    elseif g_type == IndicatorSumOfNVariables
        # Handle scaling and translation for sum constraints
        rhs = g_unwrapped.rhs 
        subvectorDim = length(rhs)
        numberVariables = g_unwrapped.numberVariables
        var[blockID] = JuMP.@variable(model, [k=1:dim])
        
        if g_not_scaled 
            # No transformation needed
            JuMP.@constraint(model, [k in 1:subvectorDim],
                sum(var[blockID][(idx-1) * subvectorDim + k] for idx in 1:numberVariables) == rhs[k])
        else
            error("addBlockVariableToJuMPModel!: IndicatorSumOfNVariables with scaling and translation is not supported yet.")
        end
    elseif g_type == ElementwiseL1Norm
        var[blockID] = JuMP.@variable(model, [k=1:dim], lower_bound = -Inf, upper_bound = Inf)
        aux[blockID] = JuMP.@variable(model, [k=1:dim], lower_bound = 0.0)
        
        if g_not_scaled 
            # No transformation needed
            JuMP.@constraint(model, [k in 1:dim], var[blockID][k] <= aux[blockID][k])
            JuMP.@constraint(model, [k in 1:dim], -var[blockID][k] <= aux[blockID][k])
        else
            # For ||g_scaling*x + g_translation||_1, auxiliary variables represent |g_scaling*x + g_translation|
            JuMP.@constraint(model, [k in 1:dim], g_scaling * var[blockID][k] + g_translation[k] <= aux[blockID][k])
            JuMP.@constraint(model, [k in 1:dim], -(g_scaling * var[blockID][k] + g_translation[k]) <= aux[blockID][k])
        end
        obj_expr = JuMP.AffExpr(0.0)
        for k in 1:dim 
            JuMP.add_to_expression!(obj_expr, g_unwrapped.coefficient * aux[blockID][k])
        end  
        push!(objExpressions, obj_expr)
    elseif g_type == IndicatorHyperplane
        var[blockID] = JuMP.@variable(model, [k = 1:dim], lower_bound = -Inf, upper_bound = Inf)
        if g_not_scaled 
            # No transformation needed
            JuMP.@constraint(model, g_unwrapped.slope' * var[blockID] == g_unwrapped.intercept)
        else
            effective_intercept = (g_unwrapped.intercept - g_unwrapped.slope' * g_translation) / g_scaling
            JuMP.@constraint(model, g_unwrapped.slope' * var[blockID] == effective_intercept)
        end
    elseif g_type == IndicatorSOC 
        var[blockID] = JuMP.@variable(model, [k = 1:dim], lower_bound = -Inf, upper_bound = Inf)
        if g_not_scaled 
            # No transformation needed
            if g_unwrapped.radiusIndex == 1
                JuMP.@constraint(model, dot(var[blockID][2:end], var[blockID][2:end]) <= var[blockID][1]^2)
                JuMP.@constraint(model, var[blockID][1] >= 0.0)
            else
                JuMP.@constraint(model, dot(var[blockID][1:end-1], var[blockID][1:end-1]) <= var[blockID][end]^2)
                JuMP.@constraint(model, var[blockID][end] >= 0.0)
            end
        else
            # For SOC with transformation, create auxiliary variables
            aux[blockID] = JuMP.@variable(model, [k = 1:dim])
            JuMP.@constraint(model, [k in 1:dim], aux[blockID][k] == g_scaling * var[blockID][k] + g_translation[k])
            
            if g_unwrapped.radiusIndex == 1
                JuMP.@constraint(model, dot(aux[blockID][2:end], aux[blockID][2:end]) <= aux[blockID][1]^2)
                JuMP.@constraint(model, aux[blockID][1] >= 0.0)
            else
                JuMP.@constraint(model, dot(aux[blockID][1:end-1], aux[blockID][1:end-1]) <= aux[blockID][end]^2)
                JuMP.@constraint(model, aux[blockID][end] >= 0.0)
            end
        end  

    elseif g_type == IndicatorRotatedSOC
        var[blockID] = JuMP.@variable(model, [k = 1:dim], lower_bound = -Inf, upper_bound = Inf)
        
        if g_not_scaled 
            # No transformation needed - direct rotated SOC constraint
            # Rotated SOC: ||x[3:end]||² ≤ 2*x[1]*x[2], x[1] ≥ 0, x[2] ≥ 0
            JuMP.@constraint(model, sum(var[blockID][k]^2 for k in 3:dim) <= 2 * var[blockID][1] * var[blockID][2])
            JuMP.@constraint(model, var[blockID][1] >= 0.0)
            JuMP.@constraint(model, var[blockID][2] >= 0.0)
        else
            # For rotated SOC with transformation, create auxiliary variables
            aux[blockID] = JuMP.@variable(model, [k = 1:dim])
            JuMP.@constraint(model, [k in 1:dim], aux[blockID][k] == g_scaling * var[blockID][k] + g_translation[k])
            
            # Apply rotated SOC constraint to transformed variables
            # ||aux[3:end]||² ≤ 2*aux[1]*aux[2], aux[1] ≥ 0, aux[2] ≥ 0
            JuMP.@constraint(model, sum(aux[blockID][k]^2 for k in 3:dim) <= 2 * aux[blockID][1] * aux[blockID][2])
            JuMP.@constraint(model, aux[blockID][1] >= 0.0)
            JuMP.@constraint(model, aux[blockID][2] >= 0.0)
        end
    else # Zero
        var[blockID] = JuMP.@variable(model, [k = 1:dim], lower_bound = -Inf, upper_bound = Inf)
    end 

    # Handle different types of f (using unwrapped function)
    f_type = typeof(f_unwrapped)
    f_not_scaled = f_scaling == 1.0 && all(f_translation .== 0.0) 
    hasNonlinearSmoothFunction = false 

    if f_type == QuadraticFunction
        # For f(x) = (f_scaling*x + f_translation)' * Q * (f_scaling*x + f_translation) + q' * (f_scaling*x + f_translation) + r
        # = f_scaling^2 * x' * Q * x + 2*f_scaling*f_translation'*Q*x + f_translation'*Q*f_translation + f_scaling*q'*x + q'*f_translation + r
        obj_expr = JuMP.QuadExpr()
        
        if f_not_scaled 
            # No transformation needed
            JuMP.add_to_expression!(obj_expr, var[blockID]' * f_unwrapped.Q * var[blockID])
            JuMP.add_to_expression!(obj_expr, f_unwrapped.q' * var[blockID] + f_unwrapped.r)
        else
            # Apply scaling and translation transformations
            Q_scaled = f_scaling^2 * f_unwrapped.Q
            q_effective = f_scaling * f_unwrapped.q + 2 * f_scaling * f_unwrapped.Q * f_translation
            r_effective = f_unwrapped.r + f_scaling * f_unwrapped.q' * f_translation + 
                         f_translation' * f_unwrapped.Q * f_translation
            
            JuMP.add_to_expression!(obj_expr, var[blockID]' * Q_scaled * var[blockID])
            JuMP.add_to_expression!(obj_expr, q_effective' * var[blockID] + r_effective)
        end
        push!(objExpressions, obj_expr)
        
    elseif f_type == AffineFunction
        # For f(x) = A' * (f_scaling*x + f_translation) + r = f_scaling*A'*x + A'*f_translation + r
        obj_expr = JuMP.AffExpr()
        if f_not_scaled 
            # No transformation needed
            JuMP.add_to_expression!(obj_expr, dot(f_unwrapped.A, var[blockID]) + f_unwrapped.r)
        else
            A_scaled = f_scaling * f_unwrapped.A
            r_effective = f_unwrapped.r + f_unwrapped.A' * f_translation
            JuMP.add_to_expression!(obj_expr, dot(A_scaled, var[blockID]) + r_effective)
        end
        push!(objExpressions, obj_expr)
        
    elseif f_type == ComponentwiseExponentialFunction 
        hasNonlinearSmoothFunction = true 
    elseif f_type == UserDefinedSmoothFunction
        hasNonlinearSmoothFunction = true 
    elseif f_type == WrapperScalarInputFunction
        hasNonlinearSmoothFunction = true 
    end

    return hasNonlinearSmoothFunction
end

"""
    nonlinearExpressionFromSmoothFunction(f::AbstractFunction, var::Vector{JuMP.VariableRef})

Create a nonlinear expression from a smooth function for use in JuMP's nonlinear objective.

# Arguments
- `f::AbstractFunction`: The smooth function to convert (may be wrapped).
- `var::Vector{JuMP.VariableRef}`: The JuMP variables to which the function applies.

# Returns
- A nonlinear expression that can be used in JuMP's nonlinear objective.

# Implementation Details
Currently supports ComponentwiseExponentialFunction, which is converted to 
`f.coefficients' * exp.(var)`. Also handles WrapperScalingTranslationFunction wrapping 
ComponentwiseExponentialFunction. Throws an error for unsupported function types.
"""
function nonlinearExpressionFromSmoothFunction(f::AbstractFunction, var::Vector{JuMP.VariableRef})
    # Unwrap function if needed
    f_unwrapped, f_scaling, f_translation = unwrapFunction(f)
    f_not_scaled = f_scaling == 1.0 && all(f_translation .== 0.0)
    try 
        if typeof(f_unwrapped) == ComponentwiseExponentialFunction 
            if f_not_scaled 
                # No transformation needed
                return f_unwrapped.coefficients' * exp.(var)
            else
                # Simplified version (no conditional needed)
                return f_unwrapped.coefficients' * exp.(f_scaling .* var .+ f_translation)
            end
        elseif typeof(f_unwrapped) == UserDefinedSmoothFunction
            if f_not_scaled 
                return f_unwrapped.func(var)
            else
                return f_unwrapped.func(f_scaling .* var .+ f_translation)
            end
        elseif typeof(f_unwrapped) == WrapperScalarInputFunction
            if isa(f_unwrapped.originalFunction, Zero)
                return 0.0 * var[1]
            elseif isa(f_unwrapped.originalFunction, UserDefinedSmoothFunction)
                # Handle UserDefinedSmoothFunction wrapped in WrapperScalarInputFunction
                if f_not_scaled 
                    return f_unwrapped.originalFunction.func(var[1])
                else
                    return f_unwrapped.originalFunction.func(f_scaling * var[1] + f_translation[1])
                end
            else
                # Handle other types of functions
                if f_not_scaled 
                    return f_unwrapped.originalFunction(var[1])
                else
                    return f_unwrapped.originalFunction(f_scaling * var[1] + f_translation[1])
                end
            end
        else 
            error("nonlinearExpressionFromSmoothFunction: unsupported type of f = $(typeof(f_unwrapped))")
        end 
    catch error 
        println("nonlinearExpressionFromSmoothFunction: Error encountered adding nonlinear expression. ")
        rethrow(error)
    end 
end 


"""
    solveMultiblockProblemByJuMP(mbp::MultiblockProblem)

Solve a multiblock problem using JuMP and Ipopt.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to solve.

# Returns
- `Float64`: The optimal objective value found by the solver.

# Implementation Details
The function:
1. Converts the multiblock problem to a JuMP model
2. Sets up the Ipopt solver with HSL linear solver
3. Creates JuMP variables for each block variable
4. Adds constraints from the multiblock problem
5. Formulates the objective function (handling both linear/quadratic and nonlinear terms)
6. Solves the problem and returns the objective value

Currently, only LinearMappingMatrix and LinearMappingIdentity are supported for mappings.
"""
function solveMultiblockProblemByJuMP(mbp::MultiblockProblem)
    for constr in mbp.constraints
        for (id, L) in constr.mappings
            @assert(typeof(L) == LinearMappingMatrix || typeof(L) == LinearMappingIdentity, 
                "solveMultiblockProblemByJuMP: unsupported type of linear mapping = $(typeof(L))")
        end 
    end 
    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    if HSL_FOUND 
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma27")
    end 

    var = Dict{BlockID, Vector{JuMP.VariableRef}}() 
    aux = Dict{BlockID, Vector{JuMP.VariableRef}}()
    objExpressions = Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}}()

    blockHasNonlinearSmoothFunction = Dict{BlockID, Bool}() 
    for block in mbp.blocks 
        blockHasNonlinearSmoothFunction[block.id] = addBlockVariableToJuMPModel!(model, 
            block.f, 
            block.g, 
            block.val, 
            block.id, 
            var, 
            aux, 
            objExpressions)
    end 

    # Add constraints
    for constr in mbp.constraints
        JuMP.@constraint(model, 
                sum((typeof(L) == LinearMappingMatrix ? L.A * var[id] : L.coe * var[id]) for (id, L) in constr.mappings) .== constr.rhs)  
    end 

    # add objective 
    quadraticObj = JuMP.QuadExpr() 
    for expr in objExpressions 
        JuMP.add_to_expression!(quadraticObj, expr)
    end 
    
    if any(values(blockHasNonlinearSmoothFunction))
        nonlinearObj = 0 
        for block in mbp.blocks 
            if blockHasNonlinearSmoothFunction[block.id] 
                nonlinearObj += nonlinearExpressionFromSmoothFunction(block.f, var[block.id]) 
            end 
        end 
        nonlinearObj += quadraticObj 
        JuMP.@NLobjective(model, Min, nonlinearObj)
    else 
        JuMP.@objective(model, Min, quadraticObj)
    end 

    # # Print the model to see what JuMP is actually solving
    # println("="^80)
    # println("JuMP MODEL BEING SOLVED:")
    # println("="^80)
    # println(model)
    # println("="^80)

    JuMP.optimize!(model)

    JuMP.solution_summary(model)

    # # Print solution values to understand why JuMP thinks it's solved
    # println("SOLUTION VALUES:")
    # for (blockID, variables) in var
    #     vals = JuMP.value.(variables)
    #     println("Block $blockID: $vals")
    # end
    # println("="^80)

    obj = JuMP.objective_value(model)
    time = JuMP.solve_time(model)
    status = JuMP.termination_status(model)
    
    msg = Printf.@sprintf("MultiblockProblem: Objective value by JuMP = %.4e, time = %.2f seconds, status = %s\n", 
                         obj, time, status) 
    @info msg  

    return obj
end 
