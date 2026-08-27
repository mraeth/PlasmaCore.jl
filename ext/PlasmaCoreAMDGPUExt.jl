module PlasmaCoreAMDGPUExt

using PlasmaCore
import AMDGPU

PlasmaCore._backend_array_matches(exec::AMDGPU.ROCBackend, x::AMDGPU.ROCArray) = true

function amdgpu_available_hook()
    try
        return AMDGPU.functional()
    catch
        return false
    end
end

function set_amdgpu_execution_space_hook()
    PlasmaCore.set_execution_space!(;
        alloc = x -> AMDGPU.ROCArray(x),
        exec = AMDGPU.ROCBackend(),
    )
    return nothing
end

function PlasmaCore._backend_synchronize!(exec::AMDGPU.ROCBackend)
    AMDGPU.synchronize()
    return nothing
end

function __init__()
    PlasmaCore.AMDGPU_AVAILABLE_HOOK[] = amdgpu_available_hook
    PlasmaCore.SET_AMDGPU_EXECUTION_SPACE_HOOK[] = set_amdgpu_execution_space_hook
    return nothing
end

end # module PlasmaCoreAMDGPUExt
