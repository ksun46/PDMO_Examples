"""
    RBAdapter <: AbstractADMMAdapter

Residual Balance (RB) adapter for ADMM penalty parameter adaptation.

This adapter implements the residual balancing strategy
which adjusts the penalty parameter ρ based on the relative magnitudes of primal and
dual residuals to maintain balanced convergence rates.

**Mathematical Background**

The ADMM algorithm generates two types of residuals at each iteration:

1. **Primal residual**: `r^k = Ax^k + Bz^k - c`
2. **Dual residual**: `s^k = ρA^T B(z^k - z^{k-1})`

The optimal penalty parameter should balance the reduction of both residuals. The RB adapter
uses the following strategy:

- If `‖r^k‖₂ > μ ‖s^k‖₂`: Increase ρ to emphasize constraint satisfaction
- If `‖s^k‖₂ > μ ‖r^k‖₂`: Decrease ρ to emphasize optimality conditions
- Otherwise: Keep ρ unchanged

**Algorithm Details**

At each iteration, the adapter:
1. Computes the current primal residual norm: `‖r^k‖₂`
2. Computes the current dual residual norm: `‖s^k‖₂`
3. Compares their ratio against the threshold `testRatio`
4. Updates ρ by multiplying or dividing by `adapterRatio`
5. Clamps ρ to be within `[ADMM_MIN_RHO, ADMM_MAX_RHO]`

**Parameters**

- `testRatio::Float64`: Threshold for residual ratio comparison (default: 10.0)
- `adapterRatio::Float64`: Factor for ρ adjustment (default: 2.0)

**Parameter Guidelines**

- **testRatio**: Typical values 5-20. Higher values make adaptation less aggressive
- **adapterRatio**: Typical values 1.5-5. Higher values make larger adjustments

**Convergence Properties**

- **Balanced Convergence**: Maintains similar convergence rates for primal and dual residuals
- **Robust Performance**: Works well across different problem types and scales
- **Proven Theory**: Backed by convergence analysis

**Performance Characteristics**

- **Computational Cost**: O(1) per iteration (just ratio comparison)
- **Memory Usage**: O(1) - no additional storage required
- **Stability**: Generally stable due to moderate adjustment factors


"""
struct RBAdapter <: AbstractADMMAdapter 
    testRatio::Float64 
    adapterRatio::Float64 
    
    """
        RBAdapter(;testRatio::Float64=10.0, adapterRatio::Float64=2.0)

    Construct a residual balance adapter with specified parameters.

    **Arguments**
    - `testRatio::Float64=10.0`: Threshold ratio for comparing primal and dual residuals
    - `adapterRatio::Float64=2.0`: Factor by which to adjust ρ when imbalance is detected

    **Parameter Selection Guidelines**

    **testRatio**:
    - Larger values (15-20): More conservative, fewer adaptations
    - Smaller values (5-10): More aggressive, frequent adaptations
    - Very small values (<5): May cause oscillations

    **adapterRatio**:
    - Larger values (3-5): Faster adaptation, risk of overshooting
    - Smaller values (1.5-2): Gradual adaptation, more stable
    - Values close to 1: Minimal adaptation effect

    **Recommended Combinations**
    - Conservative: `testRatio=15.0, adapterRatio=1.5`
    - Balanced: `testRatio=10.0, adapterRatio=2.0` (default)
    - Aggressive: `testRatio=5.0, adapterRatio=3.0`


    """
    RBAdapter(;testRatio::Float64=10.0, adapterRatio::Float64=2.0) = new(testRatio, adapterRatio)
end 

"""
    initialize!(adapter::RBAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)

Initialize the RB adapter (no-operation).

The residual balance adapter requires no initialization since it operates purely
on residual information that is computed during ADMM iterations.

**Arguments**
- `adapter::RBAdapter`: The adapter instance
- `info::ADMMIterationInfo`: Current ADMM iteration information
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation

**Implementation**
This function performs no operations and returns immediately since the RB adapter
requires no internal state initialization.
"""
function initialize!(adapter::RBAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph)
    return 
end 

"""
    updatePenalty(adapter::RBAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, iter::Int64)

Update the penalty parameter using residual balancing strategy.

This function implements the core residual balancing algorithm that adjusts ρ based
on the relative magnitudes of primal and dual residuals.

**Arguments**
- `adapter::RBAdapter`: The adapter with balancing parameters
- `info::ADMMIterationInfo`: Current iteration information containing residuals
- `admmGraph::ADMMBipartiteGraph`: Bipartite graph representation
- `iter::Int64`: Current iteration number

**Returns**
- `Bool`: `true` if ρ was updated, `false` otherwise

**Algorithm Steps**

1. **Residual Extraction**: Get current primal (`pres`) and dual (`dres`) residual norms
2. **Ratio Testing**: Compare residual magnitudes against `testRatio`
3. **Parameter Update**: Adjust ρ based on the dominant residual type:
   - If `pres > testRatio × dres`: Multiply ρ by `adapterRatio`
   - If `dres > testRatio × pres`: Divide ρ by `adapterRatio`
   - Otherwise: Keep ρ unchanged
4. **Bounds Enforcement**: Clamp ρ to `[ADMM_MIN_RHO, ADMM_MAX_RHO]`
5. **Change Detection**: Update history only if change exceeds `FeasTolerance`

**Mathematical Intuition**

- **Large primal residual**: Constraints are poorly satisfied → Increase ρ
- **Large dual residual**: Optimality conditions are poorly satisfied → Decrease ρ
- **Balanced residuals**: Current ρ is appropriate → No change

**Convergence Impact**

- **Faster convergence**: When residuals are imbalanced
- **Stable convergence**: When residuals are already balanced
- **Robust performance**: Across different problem types and scales


"""
function updatePenalty(adapter::RBAdapter, info::ADMMIterationInfo, admmGraph::ADMMBipartiteGraph, iter::Int64)
    pres = info.presL2[end]
    dres = info.dresL2[end]
    oldRho = info.rhoHistory[end][1]
    newRho = oldRho 
    if (pres > adapter.testRatio * dres)
        newRho *= adapter.adapterRatio
    elseif (dres > adapter.testRatio * pres)
        newRho /= adapter.adapterRatio
    end 

    newRho = max(min(newRho, ADMM_MAX_RHO), ADMM_MIN_RHO)
    
    if (abs(newRho - oldRho) > FeasTolerance)    
        push!(info.rhoHistory, (newRho, iter))
        return true 
    end 
    
    return false 
end