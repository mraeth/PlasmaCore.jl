module PlasmaCoreCUDAExt

using PlasmaCore
import CUDA

PlasmaCore._backend_array_matches(exec::CUDA.CUDABackend, x::CUDA.CuArray) = true

function cuda_available_hook()
    try
        return CUDA.functional()
    catch
        return false
    end
end

function set_cuda_execution_space_hook()
    PlasmaCore.set_execution_space!(; alloc = x -> CUDA.CuArray(x), exec = CUDA.CUDABackend())
    return nothing
end

function PlasmaCore._backend_synchronize!(exec::CUDA.CUDABackend)
    CUDA.synchronize()
    return nothing
end

function __init__()
    PlasmaCore.CUDA_AVAILABLE_HOOK[] = cuda_available_hook
    PlasmaCore.SET_CUDA_EXECUTION_SPACE_HOOK[] = set_cuda_execution_space_hook
    return nothing
end

end # module PlasmaCoreCUDAExt
