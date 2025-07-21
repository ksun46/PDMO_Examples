using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

@testset "WrapperScalarInputFunction Tests" begin
    @testset "Constructor" begin
        # Test wrapping different types of functions
        
        # Wrap a Zero function
        zero_func = Zero()
        wrapper_zero = WrapperScalarInputFunction(zero_func)
        @test wrapper_zero isa WrapperScalarInputFunction
        @test wrapper_zero isa AbstractFunction
        @test wrapper_zero.originalFunction === zero_func
        
        # Wrap an ElementwiseL1Norm (scalar coefficient)
        l1_func = ElementwiseL1Norm(2.0)  # Single coefficient
        wrapper_l1 = WrapperScalarInputFunction(l1_func)
        @test wrapper_l1.originalFunction === l1_func
        
        # Wrap an IndicatorBox for scalar
        box_func = IndicatorBox([-1.0], [1.0])  # Scalar box [-1, 1]
        wrapper_box = WrapperScalarInputFunction(box_func)
        @test wrapper_box.originalFunction === box_func
    end

    @testset "Function Traits Delegation" begin
        # Test that traits are properly delegated to the original function
        
        # Test with ElementwiseL1Norm (not smooth, proximal, convex)
        l1_func = ElementwiseL1Norm(1.0)
        wrapper_l1 = WrapperScalarInputFunction(l1_func)
        
        @test isSmooth(wrapper_l1) == isSmooth(typeof(l1_func))
        @test isProximal(wrapper_l1) == isProximal(typeof(l1_func))
        @test isConvex(wrapper_l1) == isConvex(typeof(l1_func))
        @test isSet(wrapper_l1) == isSet(typeof(l1_func))
        
        # Test with IndicatorBox (not smooth, proximal, convex, is set)
        box_func = IndicatorBox([-2.0], [2.0])
        wrapper_box = WrapperScalarInputFunction(box_func)
        
        @test isSmooth(wrapper_box) == isSmooth(typeof(box_func))
        @test isProximal(wrapper_box) == isProximal(typeof(box_func))
        @test isConvex(wrapper_box) == isConvex(typeof(box_func))
        @test isSet(wrapper_box) == isSet(typeof(box_func))
    end

    @testset "Function Evaluation" begin
        # Test with ElementwiseL1Norm
        l1_func = ElementwiseL1Norm(2.0)  # f(x) = 2|x|
        wrapper_l1 = WrapperScalarInputFunction(l1_func)
        
        x_test = [3.0]
        expected = 2.0 * abs(3.0)  # 6.0
        @test wrapper_l1(x_test) ≈ expected
        @test wrapper_l1(x_test) == l1_func([3.0])  # Should match vector evaluation
        
        x_negative = [-2.0]
        expected_neg = 2.0 * abs(-2.0)  # 4.0
        @test wrapper_l1(x_negative) ≈ expected_neg
        
        # Test with IndicatorBox
        lower = -1.0
        upper = 1.0
        box_func = IndicatorBox(lower, upper)
        wrapper_box = WrapperScalarInputFunction(box_func)
        
        # Test points inside box
        x_inside = [0.5]
        @test wrapper_box(x_inside) ≈ 0.0
        
        # Test points outside box
        x_outside = [2.0]
        @test wrapper_box(x_outside) == Inf
        
        # Test error cases
        @test_throws AssertionError wrapper_l1([1.0, 2.0])  # Vector of length 2
        @test_throws AssertionError wrapper_l1(Float64[])   # Empty vector
        @test_throws AssertionError wrapper_l1(1.0)        # Scalar input (not vector)
    end

    @testset "Gradient Oracle" begin
        # Test with QuadraticFunction
        Q = sparse(reshape([2.0], 1, 1))  # 1x1 matrix
        q = [0.0]  # Linear term
        r = 0.0   # Constant term
        quad_func = QuadraticFunction(Q, q, r)
        wrapper_quad = WrapperScalarInputFunction(quad_func)
        
        x_grad = [1.0]
        grad = gradientOracle(wrapper_quad, x_grad)
        @test grad ≈ [4.0]  # Gradient of x'Qx is 2Qx = 2*2*1 = 4
        
        x_grad2 = [2.0]
        grad2 = gradientOracle(wrapper_quad, x_grad2)
        @test grad2 ≈ [8.0]  # Gradient at x = 2 is 2*2*2 = 8
        
        # Test in-place version
        grad_inplace = similar(x_grad)
        gradientOracle!(grad_inplace, wrapper_quad, x_grad)
        @test grad_inplace ≈ grad
    end

    @testset "Proximal Oracle" begin
        # Test with IndicatorBox
        lower = -1.0
        upper = 1.0
        box_func = IndicatorBox(lower, upper)
        wrapper_box = WrapperScalarInputFunction(box_func)
        
        # Test projection inside box
        x_inside = [0.5]
        prox_inside = proximalOracle(wrapper_box, x_inside)
        @test prox_inside ≈ x_inside
        
        # Test projection outside box
        x_outside = [2.0]
        prox_outside = proximalOracle(wrapper_box, x_outside)
        @test prox_outside ≈ [1.0]
        
        # Test in-place version
        prox_inplace = similar(x_outside)
        proximalOracle!(prox_inplace, wrapper_box, x_outside)
        @test prox_inplace ≈ prox_outside
    end

    @testset "Edge Cases" begin
        # Test with very small values
        l1_func = ElementwiseL1Norm(1.0)
        wrapper_l1 = WrapperScalarInputFunction(l1_func)
        
        x_tiny = [1e-15]
        @test wrapper_l1(x_tiny) ≈ 1e-15
        
        prox_tiny = proximalOracle(wrapper_l1, x_tiny, 1.0)
        @test prox_tiny[1] ≈ 0.0  # Should be thresholded to zero
        
        # Test with very large values
        x_large = [1e10]
        @test wrapper_l1(x_large) ≈ 1e10
        
        prox_large = proximalOracle(wrapper_l1, x_large, 1.0)
        @test prox_large[1] ≈ 1e10 - 1.0  # Should be thresholded by 1.0
        
        # Test with zero input
        x_zero = [0.0]
        @test wrapper_l1(x_zero) ≈ 0.0
        
        prox_zero_input = proximalOracle(wrapper_l1, x_zero, 1.0)
        @test prox_zero_input[1] ≈ 0.0
        
        # Test numerical stability
        x_stable = [1e-100]
        prox_stable = proximalOracle(wrapper_l1, x_stable, 1e-50)
        @test isfinite(prox_stable[1])
        
        # Test with different gamma values
        x_test = [5.0]
        γ_small = 0.1
        γ_large = 10.0
        
        prox_small_gamma = proximalOracle(wrapper_l1, x_test, γ_small)
        prox_large_gamma = proximalOracle(wrapper_l1, x_test, γ_large)
        
        # Larger gamma should give more shrinkage
        @test abs(prox_large_gamma[1]) ≤ abs(prox_small_gamma[1])
    end

    @testset "Mathematical Properties" begin
        # Test consistency with original function
        l1_func = ElementwiseL1Norm(3.0)
        wrapper_l1 = WrapperScalarInputFunction(l1_func)
        
        # Function values should match
        x_scalar = 2.0
        x_vector = [2.0]
        
        @test wrapper_l1(x_vector) ≈ l1_func([x_scalar])
        
        # Proximal operators should match
        γ = 0.5
        prox_wrapper = proximalOracle(wrapper_l1, x_vector, γ)
        prox_original = proximalOracle(l1_func, [x_scalar], γ)
        
        @test prox_wrapper[1] ≈ prox_original[1]
        
        # Test that wrapper preserves mathematical properties
        # For L1 norm, proximal operator should be non-expansive
        x1 = [1.0]
        x2 = [3.0]
        
        prox1 = proximalOracle(wrapper_l1, x1, γ)
        prox2 = proximalOracle(wrapper_l1, x2, γ)
        
        @test norm(prox1 - prox2) ≤ norm(x1 - x2) + 1e-10
        
        # Test that wrapper preserves convexity properties
        # For convex functions, proximal operator should reduce objective
        x_test = [4.0]
        prox_test = proximalOracle(wrapper_l1, x_test, γ)
        
        obj_original = wrapper_l1(x_test) + (1/(2*γ)) * norm(x_test - x_test)^2
        obj_prox = wrapper_l1(prox_test) + (1/(2*γ)) * norm(prox_test - x_test)^2
        
        @test obj_prox ≤ obj_original + 1e-10
        
        # Test with indicator function (projection properties)
        box_func = IndicatorBox(-2.0, 2.0)  # Use scalar bounds
        wrapper_box = WrapperScalarInputFunction(box_func)
        
        # Test idempotency: proj(proj(x)) = proj(x)
        x_outside = [5.0]
        proj1 = proximalOracle(wrapper_box, x_outside, γ)
        proj2 = proximalOracle(wrapper_box, proj1, γ)
        @test proj1 ≈ proj2 atol=1e-10
        
        # Test that projection gives feasible point
        @test wrapper_box(proj1) ≈ 0.0  # Should be in the feasible set
        
        # Test with smooth function (gradient consistency)
        Q = sparse(reshape([4.0], 1, 1))  # 1x1 matrix
        b = [0.0]
        c = 0.0
        quad_func = QuadraticFunction(Q, b, c)
        wrapper_quad = WrapperScalarInputFunction(quad_func)
        
        x_grad_test = [2.0]
        grad_wrapper = gradientOracle(wrapper_quad, x_grad_test)
        grad_original = gradientOracle(quad_func, [2.0])
        
        @test grad_wrapper[1] ≈ grad_original[1]
        
        # Test finite difference approximation for gradient
        h = 1e-8
        x_fd = [1.0]
        grad_analytical = gradientOracle(wrapper_quad, x_fd)
        
        x_plus = [x_fd[1] + h]
        x_minus = [x_fd[1] - h]
        grad_numerical = (wrapper_quad(x_plus) - wrapper_quad(x_minus)) / (2 * h)
        
        @test grad_analytical[1] ≈ grad_numerical atol=1e-6
    end
end 