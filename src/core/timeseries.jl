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

# Times
`timesteps` are integer step numbers. Physical times are optional: pass
`times = [t₁, t₂, …]` (one per step) to the constructor and read them back with
[`times`](@ref). They are the only correct time axis if `dt` changed, the run did
not start at 0, or the records are irregular. Slicing keeps them aligned with the
steps. Without `times`, `times(ts)` throws (a series does not know its `dt`).
"""
mutable struct TimeSeries{F,L} <: AbstractArray{F,1}
    timesteps::Vector{Int}
    times::Union{Nothing,Vector{Float64}}
    label::String
    loader::L
    _cache::Dict{Int,F}
end

function TimeSeries(
    loader,
    timesteps::Vector{Int};
    label::String = "",
    times::Union{Nothing,AbstractVector{<:Real}} = nothing,
)
    times === nothing ||
        length(times) == length(timesteps) ||
        throw(DimensionMismatch("times has $(length(times)) entries for $(length(timesteps)) timesteps"))
    f0 = loader(first(timesteps))
    F = typeof(f0)
    return TimeSeries{F,typeof(loader)}(
        timesteps,
        times === nothing ? nothing : Vector{Float64}(times),
        label,
        loader,
        Dict(first(timesteps) => f0),
    )
end

"""
    times(ts::TimeSeries) -> Vector{Float64}

Physical time of each record, as given at construction. Throws if the series was
built without `times`.
"""
function times(ts::TimeSeries)
    ts.times === nothing && throw(
        ArgumentError("TimeSeries \"$(ts.label)\" has no times; construct it with `times = ...`"),
    )
    return ts.times
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
    ts.times === nothing ? nothing : ts.times[r],
    ts.label,
    ts.loader,
    Dict(k => ts._cache[k] for k in ts.timesteps[r] if haskey(ts._cache, k)),
)

Base.similar(ts::TimeSeries, ::Type{F2}, dims::Dims) where {F2<:Union{ScalarField,TensorField}} =
    TimeSeries{F2,Nothing}(
        copy(ts.timesteps),
        ts.times === nothing ? nothing : copy(ts.times),
        ts.label,
        nothing,
        Dict{Int,F2}(),
    )
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
