struct UserDefinedIndicatorFunction 
    model::JuMP.AbstractModel
    var::Vector{JuMP.VariableRef}
    constraints::Vector{Function} 

    function UserDefinedIndicatorFunction(constraints::Vector{Function},
        variableSize::Int64)


        try 
            model = JuMP.Model(Ipopt.Optimizer)
            JuMP.set_silent(model)
            if HSL_FOUND 
                JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
                JuMP.set_attribute(model, "linear_solver", "ma27")
            end 

            var = JuMP.@variable(model, [k=1:variableSize])

            for f in constraints 
                JuMP.@constriant(model, f(var) <= 0.0)

            end 

        catch e 

            rethrow() 
        end 
        
    
        return new(model, var, constraints, variableSize)

    end 
end 