# Convert N-dimensional indices (1-based, column-major) to 1D index
function index_nd_to_1d(indices::NTuple{N,Int}, sizes::NTuple{N,Int}) where {N}
    i0 = 0
    stride = 1
    for k = 1:N
        i0 += (indices[k] - 1) * stride
        stride *= sizes[k]
    end
    return i0 + 1
end

@inline function _index_1d_to_nd(::Int, ::NTuple{0,Int})
    return ()
end

@inline function _index_1d_to_nd(i0::Int, sizes::NTuple{N,Int}) where {N}
    idx = i0 % sizes[1] + 1
    return (idx, _index_1d_to_nd(i0 ÷ sizes[1], Base.tail(sizes))...)
end

# Convert 1D index (1-based) to N-dimensional indices (column-major)
@inline function index_1d_to_nd(i::Int, sizes::NTuple{N,Int}) where {N}
    return _index_1d_to_nd(i - 1, sizes)
end

# Convert combined indices (X + Y) to 1D index
function index_combined_to_1d(
    indicesX::NTuple{M,Int},
    indicesY::NTuple{N,Int},
    sizesX::NTuple{M,Int},
    sizesY::NTuple{N,Int},
) where {M,N}
    iX = index_nd_to_1d(indicesX, sizesX)
    strideX = prod(sizesX)
    iY = index_nd_to_1d(indicesY, sizesY)
    return iX + (iY - 1) * strideX
end

# Convert 1D index back to (X, Y) indices
@inline function index_1d_to_combined(
    i::Int,
    sizesX::NTuple{M,Int},
    sizesY::NTuple{N,Int},
) where {M,N}
    strideX = prod(sizesX)
    iX = (i - 1) % strideX + 1
    iY = (i - 1) ÷ strideX + 1
    return (index_1d_to_nd(iX, sizesX), index_1d_to_nd(iY, sizesY))
end

@kernel function spectral_multiply_kernel!(buf, ctx)
    i = @index(Global)
    if i <= length(buf)
        @inbounds buf[i] *= ctx(i)
    end
end
