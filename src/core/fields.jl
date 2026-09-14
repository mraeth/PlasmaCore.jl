"""
    ScalarField{DT,N,AT}

Grid scalar field over an `N`-dimensional spatial grid, backed by array `AT`.
A genuine `AbstractArray{DT,N}`: `[]` indexes **spatially**, exactly like
indexing the underlying array (`sf[i,j]`, `sf[end,end]`, slices, `mean`,
broadcasting, `map`, comprehensions, ... all just work via the standard
`AbstractArray` machinery).
"""
struct ScalarField{DT,N,AT<:AbstractArray{DT,N}} <: AbstractArray{DT,N}
    data::AT
    ScalarField{DT,N,AT}(data::AT) where {DT,N,AT<:AbstractArray{DT,N}} = new{DT,N,AT}(data)
end

function ScalarField(data::AbstractArray{DT,N}) where {DT,N}
    allocated = backend_array(data)
    return ScalarField{eltype(allocated),N,typeof(allocated)}(allocated)
end

Base.size(sf::ScalarField) = size(sf.data)
Base.getindex(sf::ScalarField, I::Vararg{Int,N}) where {N} = sf.data[I...]
Base.setindex!(sf::ScalarField, v, I::Vararg{Int,N}) where {N} = (sf.data[I...] = v)
Base.IndexStyle(::Type{<:ScalarField{DT,N,AT}}) where {DT,N,AT} = IndexStyle(AT)
Base.similar(sf::ScalarField, ::Type{S}, dims::Dims) where {S} =
    ScalarField(similar(sf.data, S, dims))

Base.BroadcastStyle(::Type{<:ScalarField}) = Broadcast.ArrayStyle{ScalarField}()
Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ScalarField}},
    ::Type{ElType},
) where {ElType} = ScalarField(similar(Array{ElType}, axes(bc)))

# Lets mean(vf::VectorField) (sum of ScalarField components / n) work generically.
Base.:/(sf::ScalarField, n::Number) = ScalarField(sf.data ./ n)

# Base's generic `*(::AbstractArray,::AbstractArray)` means *matrix*
# multiplication for 2D arrays (via LinearAlgebra) — not what's wanted between
# two same-shape fields. Override with elementwise (Hadamard) multiplication,
# as used e.g. by `mf[i,j] * v[j]` inside the matrix/vector product below.
Base.:*(a::ScalarField{DT,N}, b::ScalarField{DT,N}) where {DT,N} =
    ScalarField(a.data .* b.data)


"""
    TensorField{SF,Rank,NF}

Tensor field of rank `Rank` (1 = vector, 2 = matrix) over `NF` [`ScalarField`](@ref)
components, stored as an `NTuple{NF,SF}`: `NF` separately allocated,
independently contiguous component fields (struct-of-arrays), not one field of
tuples/`SVector`s.

A genuine `AbstractArray{SF,Rank}` over the **tensor-component axis** — `[]`
selects a component, not a spatial point: `vf[i]` is the `i`-th component
`ScalarField`, `mf[i,j]` is the `(i,j)` component `ScalarField`. To index
spatially into a component, first select it (`vf[i][a,b]`) or use `Array(...)`
on the whole field.
"""
struct TensorField{SF<:ScalarField,Rank,NF} <: AbstractArray{SF,Rank}
    data::NTuple{NF,SF}
    TensorField{SF,Rank,NF}(data::NTuple{NF,SF}) where {SF<:ScalarField,Rank,NF} =
        new{SF,Rank,NF}(data)
end

# Type aliases.
# For Rank=2: NF must be a perfect square; the matrix is NS×NS where NS = isqrt(NF).
const VectorField{SF,NF} = TensorField{SF,1,NF}
const MatrixField{SF,NF} = TensorField{SF,2,NF}

Base.size(vf::VectorField{SF,NF}) where {SF,NF} = (NF,)
Base.size(mf::MatrixField{SF,NF}) where {SF,NF} = let NS = isqrt(NF)
    (NS, NS)
end
Base.IndexStyle(::Type{<:TensorField}) = IndexLinear()
Base.getindex(tf::TensorField, i::Int) = tf.data[i]  # vf[i], and linear access into mf's storage
Base.getindex(mf::MatrixField{SF,NF}, i::Int, j::Int) where {SF,NF} =
    mf.data[(i-1)*isqrt(NF)+j]  # mf[i,j]

# length/iterate come for free from AbstractArray via size+getindex, and agree
# with the flat NTuple storage order above (length(vf) == NF, for c in mf
# visits all NF components).


# Constructor functions

function VectorField(data::AbstractVector{<:AbstractArray})
    isempty(data) && throw(ArgumentError("VectorField requires at least one component"))
    return VectorField([ScalarField(d) for d in data])
end

function VectorField(data::AbstractVector{<:ScalarField})
    isempty(data) && throw(ArgumentError("VectorField requires at least one component"))
    SF = typeof(data[1])
    ax = axes(data[1])
    for c in data
        axes(c) == ax || throw(DimensionMismatch("VectorField components must share axes"))
        typeof(c) == SF ||
            throw(ArgumentError("VectorField components must have the same array type"))
    end
    NF = length(data)
    return TensorField{SF,1,NF}(ntuple(i -> data[i], NF))
end

function VectorField(data::Tuple{Vararg{T}}) where {T<:ScalarField}
    return VectorField(collect(data))
end

function MatrixField(data::AbstractMatrix{<:ScalarField})
    NR, NC = size(data)
    NR == NC || throw(ArgumentError("MatrixField requires a square matrix (got $NR×$NC)"))
    NR >= 1 || throw(ArgumentError("MatrixField requires at least one component"))
    SF = typeof(data[1, 1])
    ax = axes(data[1, 1])
    for component in data
        axes(component) == ax ||
            throw(DimensionMismatch("MatrixField components must share axes"))
        typeof(component) == SF ||
            throw(ArgumentError("MatrixField components must have the same array type"))
    end
    NF = NR * NR
    tup = ntuple(k -> data[(k-1)÷NR+1, (k-1)%NR+1], NF)
    return TensorField{SF,2,NF}(tup)
end

function MatrixField(rows::AbstractVector{<:VectorField})
    isempty(rows) && throw(ArgumentError("MatrixField requires at least one row"))
    NR = length(rows)
    NC = length(rows[1])
    NR == NC || throw(ArgumentError("MatrixField requires a square matrix (got $NR×$NC)"))
    return MatrixField([rows[i][j] for i = 1:NR, j = 1:NC])
end

function MatrixField(data::AbstractMatrix{<:AbstractArray})
    NR, NC = size(data)
    NR == NC || throw(ArgumentError("MatrixField requires a square matrix (got $NR×$NC)"))
    return MatrixField([ScalarField(data[i, j]) for i = 1:NR, j = 1:NC])
end


# Unified element-wise arithmetic for VectorField and MatrixField.
# ScalarField's own +/-/* (Number) come for free from Base's generic
# AbstractArray arithmetic (broadcast-backed, dispatching through the
# BroadcastStyle/similar pair above), so only the tuple-level combination
# needs to be spelled out here.
const _CompoundTF = Union{VectorField,MatrixField}

Base.:+(a::TF, b::TF) where {TF<:_CompoundTF} = _reconstruct(a, map(+, _comps(a), _comps(b)))
Base.:-(a::TF, b::TF) where {TF<:_CompoundTF} = _reconstruct(a, map(-, _comps(a), _comps(b)))
Base.:*(n::Number, a::TF) where {TF<:_CompoundTF} = _reconstruct(a, map(x -> n * x, _comps(a)))
Base.:*(a::TF, n::Number) where {TF<:_CompoundTF} = _reconstruct(a, map(x -> x * n, _comps(a)))

_comps(tf::TensorField) = getfield(tf, :data)
_reconstruct(::TensorField{SF,Rank,NF}, comps::NTuple{NF,SF}) where {SF,Rank,NF} =
    TensorField{SF,Rank,NF}(comps)


# Matrix-specific operations (square NS×NS matrices)

function Base.:*(R::AbstractMatrix{<:Number}, v::VectorField{SF,NF}) where {SF,NF}
    NR_out = size(R, 1)
    size(R, 2) == NF || throw(
        DimensionMismatch("matrix columns $(size(R,2)) must match VectorField length $NF"),
    )
    result = [sum(R[i, j] * v[j] for j = 1:NF) for i = 1:NR_out]
    return VectorField(result)
end

function Base.:*(R::AbstractMatrix{<:Number}, mf::MatrixField{SF,NF}) where {SF,NF}
    NS = isqrt(NF)
    size(R) == (NS, NS) ||
        throw(DimensionMismatch("rotation matrix must be $NS×$NS to match MatrixField"))
    result = [sum(R[i, k] * mf[k, j] for k = 1:NS) for i = 1:NS, j = 1:NS]
    return MatrixField(result)
end

function Base.:*(mf::MatrixField{SF,NF}, R::AbstractMatrix{<:Number}) where {SF,NF}
    NS = isqrt(NF)
    size(R) == (NS, NS) ||
        throw(DimensionMismatch("rotation matrix must be $NS×$NS to match MatrixField"))
    result = [sum(mf[i, k] * R[k, j] for k = 1:NS) for i = 1:NS, j = 1:NS]
    return MatrixField(result)
end

Base.transpose(mf::MatrixField{SF,NF}) where {SF,NF} = let NS = isqrt(NF)
    MatrixField([mf[j, i] for i = 1:NS, j = 1:NS])
end

Base.adjoint(mf::MatrixField) = transpose(mf)

function Base.:*(mf::MatrixField{SF,NF}, v::VectorField{SF,NS}) where {SF,NF,NS}
    isqrt(NF) == NS || throw(
        DimensionMismatch("MatrixField side $(isqrt(NF)) must match VectorField length $NS"),
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

function zero_vectorfield_like(vf::VectorField{SF,NF}) where {SF,NF}
    return VectorField([ScalarField(zero.(c.data)) for c in vf])
end


function Base.copyto!(dst::VectorField, src::VectorField)
    for (da, sa) in zip(dst, src)
        da.data .= sa.data
    end
    return dst
end

function Base.copyto!(dst::VectorField, src::AbstractVector{<:ScalarField})
    for (da, sf) in zip(dst, src)
        da.data .= sf.data
    end
    return dst
end

function Base.materialize!(dst::VectorField, bc::Base.Broadcast.Broadcasted)
    copyto!(dst, Base.materialize(bc))
    return dst
end
