"""
    QuadraticFunction(Q::SparseMatrixCSC{Float64, Int64}, q::Vector{Float64}, r::Float64)

Represents a quadratic function of the form f(x) = x'Qx + q'x + r.

# Mathematical Definition
f(x) = x'Qx + q'x + r

where:
- Q is a symmetric matrix (Hessian)
- q is a linear term vector
- r is a scalar offset

# Arguments
- `Q::SparseMatrixCSC{Float64, Int64}`: Quadratic coefficient matrix (should be symmetric)
- `q::Vector{Float64}`: Linear coefficient vector
- `r::Float64`: Scalar offset term

# Constructors
- `QuadraticFunction(Q, q, r)`: Full specification
- `QuadraticFunction(n::Int64)`: Zero quadratic function of dimension n

# Properties
- **Smooth**: Yes, quadratic functions are infinitely differentiable
- **Convex**: Yes, if Q is positive semidefinite
- **Proximal**: No, proximal operator not implemented (requires solving linear system)

# Mathematical Properties
- **Gradient**: ∇f(x) = (Q + Q')x + q
- **Hessian**: ∇²f(x) = Q + Q'

"""
struct QuadraticFunction <: AbstractFunction 
    Q::SparseMatrixCSC{Float64, Int64}
    q::Vector{Float64}
    r::Float64  

    function QuadraticFunction(Q::SparseMatrixCSC{Float64, Int64}, q::Vector{Float64}, r::Float64)
        rows, cols = size(Q) 
        if rows != cols || cols != length(q)
            error("QuadraticFunction: Dimension mismatch")
        end 
        new(Q, q, r)
    end 
end 

QuadraticFunction(n::Int64) = QuadraticFunction(spzeros(n,n), zeros(n), 0.0)

isSmooth(f::Type{<:QuadraticFunction}) = true
isConvex(f::Type{<:QuadraticFunction}) = true
isSupportedByJuMP(f::Type{<:QuadraticFunction}) = true

# function value
function (f::QuadraticFunction)(x::NumericVariable, enableParallel::Bool=false)
    if isa(x, Number)
        return f.Q[1,1] * x * x + f.q[1] * x + f.r
    end
    return dot(f.Q * x, x) + dot(f.q, x) + f.r
end

# gradient oracle
function gradientOracle!(y::Vector{Float64}, f::QuadraticFunction, x::Vector{Float64}, enableParallel::Bool=false)
    temp1 = f.Q * x
    temp2 = f.Q' * x
    y .= temp1 .+ temp2 .+ f.q
    return y
end

function gradientOracle(f::QuadraticFunction, x::NumericVariable, enableParallel::Bool=false)
    if isa(x, Number)
        return 2.0 * f.Q[1,1] * x + f.q[1]
    end
    y = similar(x)
    gradientOracle!(y, f, x, enableParallel)
    return y
end

# function JuMPAddProximableFunction(f::QuadraticFunction, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
#     @assert length(var) == size(f.Q, 1) == size(f.Q, 2) "QuadraticFunction: variable dimension must match Q dimension"
#     # Create quadratic expression: x'Qx + q'x + r
#     obj_expr = JuMP.QuadExpr()
#     JuMP.add_to_expression!(obj_expr, var' * f.Q * var)
#     JuMP.add_to_expression!(obj_expr, f.q' * var + f.r)
    
#     return obj_expr
# end

function JuMPAddSmoothFunction(f::QuadraticFunction, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    @assert length(var) == size(f.Q, 1) == size(f.Q, 2) "QuadraticFunction: variable dimension must match Q dimension"
    # Create quadratic expression: x'Qx + q'x + r
    obj_expr = JuMP.QuadExpr()
    JuMP.add_to_expression!(obj_expr, var' * f.Q * var)
    JuMP.add_to_expression!(obj_expr, f.q' * var + f.r)
    
    return obj_expr
end