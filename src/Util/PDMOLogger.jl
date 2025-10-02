# PDMOLogger.jl - Simple logging wrapper for PDMO

using Logging

"""
    @PDMOInfo level msg [key=value ...]

Log info message if level >= 1. Wrapper around @info.
"""
macro PDMOInfo(level, msg, args...)
    quote
        if $(esc(level)) >= 1
            @info $(esc(msg)) $(esc.(args)...)
        end
    end
end

"""
    @PDMOWarn level msg [key=value ...]

Log warning message if level >= 2. Wrapper around @warn.
"""
macro PDMOWarn(level, msg, args...)
    quote
        if $(esc(level)) >= 2
            @warn $(esc(msg)) $(esc.(args)...)
        end
    end
end

"""
    @PDMOError level msg [key=value ...]

Log error message if level >= 3. Wrapper around @error.
"""
macro PDMOError(level, msg, args...)
    quote
        if $(esc(level)) >= 3
            @error $(esc(msg)) $(esc.(args)...)
        end
    end
end

"""
    @PDMODebug level msg [key=value ...]

Log debug message if level >= 3. Wrapper around @debug.
"""
macro PDMODebug(level, msg, args...)
    quote
        if $(esc(level)) >= 3
            @debug $(esc(msg)) $(esc.(args)...)
        end
    end
end
