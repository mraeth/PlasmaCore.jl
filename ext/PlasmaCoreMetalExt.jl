module PlasmaCoreMetalExt

using PlasmaCore
import Metal

PlasmaCore._backend_array_matches(exec::Metal.MetalBackend, x::Metal.MtlArray) = true

function metal_available_hook()
    try
        return Metal.functional()
    catch
        return false
    end
end

function set_metal_execution_space_hook()
    # Metal does not support Float64; downcast to Float32 on upload.
    PlasmaCore.set_execution_space!(;
        alloc = x -> Metal.MtlArray(Float32.(x)),
        exec  = Metal.MetalBackend(),
    )
    return nothing
end

function PlasmaCore._backend_synchronize!(exec::Metal.MetalBackend)
    Metal.synchronize()
    return nothing
end

function __init__()
    PlasmaCore.METAL_AVAILABLE_HOOK[] = metal_available_hook
    PlasmaCore.SET_METAL_EXECUTION_SPACE_HOOK[] = set_metal_execution_space_hook
    return nothing
end

end # module PlasmaCoreMetalExt
