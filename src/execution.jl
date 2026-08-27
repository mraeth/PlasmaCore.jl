function use_cpu!()
    set_execution_space!(; alloc = identity, exec = KernelAbstractions.CPU())
    return nothing
end

function cuda_available()
    return _cuda_available()
end

function use_cuda!()
    cuda_available() || error("CUDA is installed but no functional GPU is available")
    return _set_cuda_execution_space!()
end

function amdgpu_available()
    return _amdgpu_available()
end

function use_amdgpu!()
    amdgpu_available() || error("AMDGPU is installed but no functional GPU is available")
    return _set_amdgpu_execution_space!()
end

function metal_available()
    return _metal_available()
end

function use_metal!()
    metal_available() || error("Metal is installed but no functional GPU is available")
    return _set_metal_execution_space!()
end

function backend_copy(x::AbstractArray)
    return backend_array(Array(x))
end

function backend_copy(f::DistributionGrid)
    new_raw = backend_copy(f.data)
    new_sf = ScalarField(new_raw)
    # DistributionGridImpl parameters are now {NX, NV, ID, SF}
    NX, NV, ID = typeof(f).parameters[1:3]
    return DistributionGridImpl{NX, NV, ID, typeof(new_sf)}(new_sf)
end

function backend_copy(e::VectorField)
    return VectorField([backend_copy(component) for component in e])
end

function backend_copy(e::ScalarField)
    return ScalarField(backend_copy(e.data))
end

function backend_copy(m::TensorField{DT,N,AT,2,NF}) where {DT,N,AT,NF}
    NS = isqrt(NF)
    return MatrixField([backend_copy(m[i, j]) for i = 1:NS, j = 1:NS])
end


function backend_synchronize!()
    exec = backend()
    KernelAbstractions.synchronize(exec)
    _backend_synchronize!(exec)
    return nothing
end
