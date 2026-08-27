struct TensorField{DT,N,AT<:AbstractArray{DT,N},Rank,NF}
    data::NTuple{NF,AT}
    function TensorField{DT,N,AT,Rank,NF}(
        data::NTuple{NF,AT},
    ) where {DT,N,AT<:AbstractArray{DT,N},Rank,NF}
        return new{DT,N,AT,Rank,NF}(data)
    end
end

# Type aliases.
# For Rank=2: NF must be a perfect square; the matrix is NS×NS where NS = isqrt(NF).
const ScalarField{DT,N,AT} = TensorField{DT,N,AT,0,1}
const VectorField{DT,N,AT,NF} = TensorField{DT,N,AT,1,NF}
const MatrixField{DT,N,AT,NF} = TensorField{DT,N,AT,2,NF}

import Base: +, -, *, transpose, adjoint, getindex, iterate, length, size


# ScalarField: .data returns the raw array (unwrap single-element NTuple)
function Base.getproperty(tf::ScalarField, sym::Symbol)
    sym === :data && return getfield(tf, :data)[1]
    return getfield(tf, sym)
end


# Constructor functions

function ScalarField(data::AbstractArray{DT,N}) where {DT,N}
    allocated = backend_array(data)
    ET = eltype(allocated)
    AT = typeof(allocated)
    return TensorField{ET,N,AT,0,1}((allocated,))
end

function VectorField(data::AbstractVector{<:AbstractArray})
    isempty(data) && throw(ArgumentError("VectorField requires at least one component"))
    arrs = ntuple(i -> backend_array(data[i]), length(data))
    DT = eltype(first(arrs))
    N = ndims(first(arrs))
    AT = typeof(first(arrs))
    NF = length(arrs)
    ax = axes(first(arrs))
    for arr in arrs
        axes(arr) == ax ||
            throw(DimensionMismatch("VectorField components must share axes"))
        typeof(arr) == AT ||
            throw(ArgumentError("VectorField components must have the same array type"))
    end
    return TensorField{DT,N,AT,1,NF}(arrs)
end

function VectorField(data::AbstractVector{<:ScalarField})
    return VectorField([getfield(sf, :data)[1] for sf in data])
end

function VectorField(data::Tuple{Vararg{T}}) where {T<:ScalarField}
    return VectorField(collect(data))
end

function MatrixField(data::AbstractMatrix{<:ScalarField})
    NR, NC = size(data)
    NR == NC || throw(ArgumentError("MatrixField requires a square matrix (got $NR×$NC)"))
    NR >= 1 || throw(ArgumentError("MatrixField requires at least one component"))
    first_component = data[1, 1]
    ax = axes(getfield(first_component, :data)[1])
    AT = typeof(getfield(first_component, :data)[1])
    for component in data
        raw = getfield(component, :data)[1]
        axes(raw) == ax ||
            throw(DimensionMismatch("MatrixField components must share axes"))
        typeof(raw) == AT ||
            throw(ArgumentError("MatrixField components must have the same array type"))
    end
    DT = eltype(AT)
    N = ndims(AT)
    NF = NR * NR
    tup = ntuple(k -> getfield(data[(k-1)÷NR+1, (k-1)%NR+1], :data)[1], NF)
    return TensorField{DT,N,AT,2,NF}(tup)
end

function MatrixField(rows::AbstractVector{<:VectorField})
    isempty(rows) && throw(ArgumentError("MatrixField requires at least one row"))
    NR = length(rows)
    NC = length(getfield(rows[1], :data))
    NR == NC || throw(ArgumentError("MatrixField requires a square matrix (got $NR×$NC)"))
    return MatrixField([rows[i][j] for i = 1:NR, j = 1:NC])
end

function MatrixField(data::AbstractMatrix{<:AbstractArray})
    NR, NC = size(data)
    NR == NC || throw(ArgumentError("MatrixField requires a square matrix (got $NR×$NC)"))
    return MatrixField([
        ScalarField(backend_array(data[i, j])) for i = 1:NR, j = 1:NC
    ])
end


# Internal helpers: flat NTuple of raw arrays
_comps(tf::TensorField) = getfield(tf, :data)

# Reconstruct same field type from a new flat NTuple of raw arrays
_reconstruct(::TensorField{DT,N,AT,Rank,NF}, comps::NTuple{NF,AT}) where {DT,N,AT,Rank,NF} =
    TensorField{DT,N,AT,Rank,NF}(comps)


# Indexing and size.
# For MatrixField (Rank=2, NF=NS²): the matrix is NS×NS where NS = isqrt(NF).

getindex(vf::VectorField{DT,N,AT,NF}, i::Int) where {DT,N,AT,NF} =
    TensorField{DT,N,AT,0,1}((_comps(vf)[i],))

function getindex(mf::TensorField{DT,N,AT,2,NF}, i::Int, j::Int) where {DT,N,AT,NF}
    NS = isqrt(NF)
    return TensorField{DT,N,AT,0,1}((_comps(mf)[(i-1)*NS+j],))
end

function getindex(mf::TensorField{DT,N,AT,2,NF}, i::Int) where {DT,N,AT,NF}
    NS = isqrt(NF)
    return TensorField{DT,N,AT,1,NS}(ntuple(j -> _comps(mf)[(i-1)*NS+j], NS))
end

size(::TensorField{DT,N,AT,2,NF}) where {DT,N,AT,NF} =
    let NS = isqrt(NF);
        (NS, NS)
    end
length(tf::TensorField) = length(_comps(tf))

function iterate(tf::TensorField{DT,N,AT,Rank,NF}, state = 1) where {DT,N,AT,Rank,NF}
    state > NF && return nothing
    return (TensorField{DT,N,AT,0,1}((_comps(tf)[state],)), state + 1)
end


# ScalarField-specific arithmetic

function _sf_elemwise(op, a::ScalarField, b::ScalarField)
    axes(_comps(a)[1]) == axes(_comps(b)[1]) ||
        throw(DimensionMismatch("ScalarField axes must match for $(nameof(op))"))
    return ScalarField(op.(_comps(a)[1], _comps(b)[1]))
end

+(a::ScalarField{DT,N}, b::ScalarField{DT,N}) where {DT,N} = _sf_elemwise(+, a, b)
-(a::ScalarField{DT,N}, b::ScalarField{DT,N}) where {DT,N} = _sf_elemwise(-, a, b)
*(a::ScalarField{DT,N}, b::ScalarField{DT,N}) where {DT,N} = _sf_elemwise(*, a, b)

+(a::ScalarField, b::Number) = ScalarField(_comps(a)[1] .+ eltype(_comps(a)[1])(b))
+(a::Number, b::ScalarField) = ScalarField(eltype(_comps(b)[1])(a) .+ _comps(b)[1])
*(a::ScalarField, b::Number) = ScalarField(_comps(a)[1] .* eltype(_comps(a)[1])(b))
*(a::Number, b::ScalarField) = ScalarField(eltype(_comps(b)[1])(a) .* _comps(b)[1])


# Unified element-wise arithmetic for VectorField and MatrixField
const _CompoundTF = Union{VectorField,MatrixField}

+(a::TF, b::TF) where {TF<:_CompoundTF} =
    _reconstruct(a, map((x, y) -> x .+ y, _comps(a), _comps(b)))

-(a::TF, b::TF) where {TF<:_CompoundTF} =
    _reconstruct(a, map((x, y) -> x .- y, _comps(a), _comps(b)))

*(n::Number, a::TF) where {TF<:_CompoundTF} = _reconstruct(a, map(x -> eltype(x)(n) .* x, _comps(a)))
*(a::TF, n::Number) where {TF<:_CompoundTF} = _reconstruct(a, map(x -> x .* eltype(x)(n), _comps(a)))


# Matrix-specific operations (square NS×NS matrices)

function *(R::AbstractMatrix{<:Number}, v::VectorField{DT,N,AT,NF}) where {DT,N,AT,NF}
    NR_out = size(R, 1)
    size(R, 2) == NF || throw(
        DimensionMismatch("matrix columns $(size(R,2)) must match VectorField length $NF"),
    )
    result = [sum(R[i, j] * v[j] for j = 1:NF) for i = 1:NR_out]
    return VectorField(result)
end

function *(R::AbstractMatrix{<:Number}, mf::TensorField{DT,N,AT,2,NF}) where {DT,N,AT,NF}
    NS = isqrt(NF)
    size(R) == (NS, NS) ||
        throw(DimensionMismatch("rotation matrix must be $NS×$NS to match MatrixField"))
    result = [sum(R[i, k] * mf[k, j] for k = 1:NS) for i = 1:NS, j = 1:NS]
    return MatrixField(result)
end

function *(mf::TensorField{DT,N,AT,2,NF}, R::AbstractMatrix{<:Number}) where {DT,N,AT,NF}
    NS = isqrt(NF)
    size(R) == (NS, NS) ||
        throw(DimensionMismatch("rotation matrix must be $NS×$NS to match MatrixField"))
    result = [sum(mf[i, k] * R[k, j] for k = 1:NS) for i = 1:NS, j = 1:NS]
    return MatrixField(result)
end

transpose(mf::TensorField{DT,N,AT,2,NF}) where {DT,N,AT,NF} =
    let NS = isqrt(NF)
        MatrixField([mf[j, i] for i = 1:NS, j = 1:NS])
    end

adjoint(mf::MatrixField) = transpose(mf)

function *(mf::TensorField{DT,N,AT,2,NF}, v::VectorField{DT,N,AT,NS}) where {DT,N,AT,NF,NS}
    isqrt(NF) == NS || throw(
        DimensionMismatch(
            "MatrixField side $(isqrt(NF)) must match VectorField length $NS",
        ),
    )
    result = [sum(mf[i, j] * v[j] for j = 1:NS) for i = 1:NS]
    return VectorField(result)
end


# Allocation helpers

_grid_dims(grid::Grid) = Tuple(length(ax) for ax in grid.xaxes)

function empty_matrixfield(grid::Grid, NS::Int)
    dims = _grid_dims(grid)
    data = [ScalarField(zeros(Float64, dims)) for _ = 1:(NS*NS)]
    return MatrixField(reshape(data, NS, NS))
end

function empty_matrixfield(grid::Grid, NR::Int, NC::Int)
    NR == NC || throw(ArgumentError("MatrixField requires a square matrix (got $NR×$NC)"))
    return empty_matrixfield(grid, NR)
end

function empty_vectorfield(grid::Grid, ncomp::Int)
    dims = _grid_dims(grid)
    data = [ScalarField(zeros(Float64, dims)) for _ = 1:ncomp]
    return VectorField(data)
end

function empty_vectorfield(grid::Grid)
    return empty_vectorfield(grid, length(grid.vaxes))
end

function empty_scalarfield(grid::Grid)
    data = allocate(zeros(Float64, _grid_dims(grid)))
    return ScalarField(data)
end

function zero_vectorfield_like(vf::VectorField{DT,N,AT,NF}) where {DT,N,AT,NF}
    return TensorField{DT,N,AT,1,NF}(map(arr -> zero.(arr), _comps(vf)))
end


function Base.copyto!(dst::VectorField, src::VectorField)
    for (da, sa) in zip(_comps(dst), _comps(src))
        da .= sa
    end
    return dst
end

function Base.copyto!(dst::VectorField, src::AbstractVector{<:ScalarField})
    for (da, sf) in zip(_comps(dst), src)
        da .= getfield(sf, :data)[1]
    end
    return dst
end

function Base.materialize!(dst::VectorField, bc::Base.Broadcast.Broadcasted)
    copyto!(dst, Base.materialize(bc))
    return dst
end
