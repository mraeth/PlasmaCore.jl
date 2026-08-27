import RecipesBase

mutable struct SimulationTime{T<:AbstractFloat} <: AbstractVector{T}
    dt::T
    fraction_dt::T
    current_T::T
    final_T::T
    step::Int
    phase::T
    nmax::Int
    wall_start_ns::UInt64
    _progress::Union{Progress,Nothing}
end


function SimulationTime(
    dt::T,
    final_T::T;
    nmax::Integer = typemax(Int),
    current_T = zero(T),
    step::Integer = 0,
    phase = zero(T),
) where {T<:AbstractFloat}
    return SimulationTime{T}(
        dt,
        one(T),
        T(current_T),
        final_T,
        Int(step),
        T(phase),
        Int(round(Int, final_T/dt)),
        time_ns(),
        nothing,
    )
end

# Aux function to indexing SimulationTime with [n] to get n*dt
Base.size(t::SimulationTime) = (t.nmax,)
Base.getindex(t::SimulationTime, i::Int) = t.dt * (i - 1)
Base.IndexStyle(::Type{<:SimulationTime}) = IndexLinear()

function SimulationTime(dt::Real, final_T::Real; kwargs...)
    T = promote_type(typeof(float(dt)), typeof(float(final_T)))
    return SimulationTime(T(dt), T(final_T); kwargs...)
end


function advance!(t::SimulationTime{T}, gyro_frequency::Real; wrap_phase::Bool = true) where {T<:AbstractFloat}
    t.current_T += t.dt
    t.step += 1
    t.phase += T(gyro_frequency) * t.dt

    if wrap_phase
        t.phase = mod(t.phase, T(2π))
    end

    return t
end

advance!(t::SimulationTime) = advance!(t, 0.0; wrap_phase = false)
function continue_advection(t::SimulationTime, show_progress::Bool = false)
    if show_progress
        isnothing(t._progress) && (
            t._progress = Progress(
                round(Int, t.final_T / t.dt);
                desc = "Simulating: ",
                showspeed = true,
            )
        )
        next!(t._progress; showvalues = [(:step, t.step), (:time, t.current_T)])
    end
    return t.current_T < t.final_T && t.step < t.nmax
end

function elapsed_seconds(t::SimulationTime)
    return Float64(time_ns() - t.wall_start_ns) * 1.0e-9
end

function reset_timer!(t::SimulationTime)
    t.wall_start_ns = time_ns()
    return t
end

Base.eltype(::Type{<:SimulationTime{T}}) where {T} = T
Base.length(t::SimulationTime) = min(
    floor(Int, t.final_T / t.dt) + 1,
    t.nmax == typemax(Int) ? typemax(Int) : t.nmax + 1,
)

function Base.iterate(t::SimulationTime, state::Int = 0)
    (state > t.nmax || t.dt * state > t.final_T) && return nothing
    return (t.dt * state, state + 1)
end

RecipesBase.@recipe function f(t::SimulationTime)
    collect(t)
end
