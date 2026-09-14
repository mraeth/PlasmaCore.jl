"""
    TimeSeries{F,L}

Lazy, cached time series of fields (`ScalarField`/`TensorField`), generic over
*how* a timestep is fetched via a `loader::L` closure (`loader(step::Int)::F`).
A genuine `AbstractArray{F,1}` over an integer time axis — `mean`, `sum`,
broadcasting, `map`, slicing, and comprehensions all work directly.

# Indexing
- `ts[i]`      — i-th field (1-based, lazily loaded and cached)
- `ts[a:b]`    — sub-`TimeSeries` sharing the parent cache (no re-fetching)
- `ts[t, ...]` — spatial slice at a single timestep (composes with the field's
  own spatial indexing: `ts[t, ...] == ts[t][...]`)
- `ts[:, ...]` / `ts[a:b, ...]` — spatial slice across (a range of) timesteps,
  returned as a `TimeSeries`
"""
mutable struct TimeSeries{F,L} <: AbstractArray{F,1}
    timesteps::Vector{Int}
    label::String
    loader::L
    _cache::Dict{Int,F}
end

function TimeSeries(loader, timesteps::Vector{Int}; label::String = "")
    f0 = loader(first(timesteps))
    F = typeof(f0)
    return TimeSeries{F,typeof(loader)}(
        timesteps,
        label,
        loader,
        Dict(first(timesteps) => f0),
    )
end

Base.size(ts::TimeSeries) = (length(ts.timesteps),)
Base.IndexStyle(::Type{<:TimeSeries}) = IndexLinear()

function Base.getindex(ts::TimeSeries{F}, i::Int) where {F}
    step = ts.timesteps[i]
    haskey(ts._cache, step) && return ts._cache[step]
    f = ts.loader(step)::F
    ts._cache[step] = f
    return f
end
Base.setindex!(ts::TimeSeries{F}, v::F, i::Int) where {F} = (ts._cache[ts.timesteps[i]] = v)

Base.getindex(ts::TimeSeries{F}, r::AbstractVector{<:Integer}) where {F} = TimeSeries{
    F,
    typeof(ts.loader),
}(
    ts.timesteps[r],
    ts.label,
    ts.loader,
    Dict(k => ts._cache[k] for k in ts.timesteps[r] if haskey(ts._cache, k)),
)

Base.similar(ts::TimeSeries, ::Type{F2}, dims::Dims) where {F2<:Union{ScalarField,TensorField}} =
    TimeSeries{F2,Nothing}(copy(ts.timesteps), ts.label, nothing, Dict{Int,F2}())
Base.similar(ts::TimeSeries, ::Type{S}, dims::Dims) where {S} = similar(Array{S}, dims)

# Explicit convenience overloads, layered on top of the genuine
# AbstractArray{F,1} declaration above (more specific than, so preferred over,
# the generic AbstractArray fallback whenever more than one index is given).
Base.getindex(ts::TimeSeries, ::Colon) = ts
Base.getindex(ts::TimeSeries, t::Integer, spatial...) = ts[t][spatial...]
function Base.getindex(ts::TimeSeries, t_range, spatial...)
    r = t_range isa Colon ? Base.OneTo(length(ts)) : t_range
    return map(x -> x[spatial...], ts[r])
end
